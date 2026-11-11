"""Gymnasium environment shared by training, evaluation, and replay."""

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from .reward import RewardSignals
from .reward import reward as student_reward
from .track import Track
from .vehicle import MAX_SPEED, MAX_STEER, MAX_WHEEL_ANGLE, WHEELBASE, Vehicle


class RaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    max_speed = MAX_SPEED
    max_steer = MAX_STEER
    time_limit = 48.
    control_dt = .05
    lookahead = np.array([.6, 1.3, 2.2, 3.5, 5.0])

    def __init__(self, reward_fn=student_reward, render_mode=None, camera="overview", layout="shanghai",
                 physics_dt=.01, diagnostics=False):
        self.track = Track(layout)
        self.gate_centers, indices = self.track.sample(self.track.gates)
        self.gate_tangents = self.track.tangents[indices]
        xml = Path(__file__).with_name("car.xml")
        if render_mode == "rgb_array":
            tree = ET.parse(xml)
            self.track.add_visuals(tree.getroot())
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.frame_skip = round(self.control_dt / physics_dt)
        if self.frame_skip < 1 or not math.isclose(self.frame_skip * physics_dt, self.control_dt):
            raise ValueError("physics_dt must evenly divide the control interval")
        self.model.opt.timestep = physics_dt
        self.data = mujoco.MjData(self.model)
        self.car = self.model.body("car").id
        self.vehicle = Vehicle(self.model, self.data, diagnostics)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = gym.spaces.Box(-10., 10., (20,), np.float32)
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.camera = camera
        self.renderer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.arc = -.65
        center, self.segment = self.track.sample(self.arc)
        tangent = self.track.tangents[self.segment]
        lateral, heading = self.np_random.uniform([-.25, -.14], [.25, .14])
        yaw = math.atan2(tangent[1], tangent[0]) + heading
        self.data.qpos[:2] = center + lateral * np.array([-tangent[1], tangent[0]])
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.next_gate = 0
        self.max_progress = 0.
        self.vehicle.reset()
        self.finish_time = None
        self.peak_speed = self.peak_grip = self.peak_slip = 0.
        self.tilt_time = self.overspeed_time = 0.
        obs, _, _, _ = self._observe()
        return obs, self._info("running")

    def step(self, action):
        try:
            action = np.asarray(action, dtype=float)
        except (TypeError, ValueError, OverflowError):
            action = np.empty(0)
        if action.shape != (2,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.000001:
            return self._observe()[0], -8., True, False, self._info("invalid_action")
        steer, speed = np.clip(action, -1., 1.)
        before_arc, before_time, before_progress = self.arc, self.data.time, self.max_progress
        before_steer = math.atan(WHEELBASE * self.vehicle.curvature)
        clearance = self.track.half_width
        grip_risk = normal_load = 0.
        status = "running"

        for _ in range(self.frame_skip):
            dt = self.model.opt.timestep
            start = self.data.qpos[:2].copy()
            start_time = self.data.time
            self.vehicle.apply(float(steer) * self.max_steer, float(speed + 1.) * .5 * self.max_speed)
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= start_time or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                status = "physics_error"
                break

            risk, load, peak_grip, peak_slip = self.vehicle.contacts()
            grip_risk += risk * dt
            normal_load += load * dt
            self.peak_grip = max(self.peak_grip, peak_grip)
            self.peak_slip = max(self.peak_slip, peak_slip)
            self.peak_speed = max(self.peak_speed, math.hypot(*self.data.qvel[:2]))
            mujoco.mj_kinematics(self.model, self.data)

            segments, arcs, distances = self.track.project(self.vehicle.footprint(), self.segment)
            clearance = min(clearance, self.track.half_width - math.sqrt(distances.max()))
            self.tilt_time = self.tilt_time + dt if self.data.xmat[self.car, 8] < .5 else 0.
            overspeed = math.hypot(*self.data.qvel[:2]) > 5.10
            self.overspeed_time = self.overspeed_time + dt if overspeed else 0.
            if clearance < -1e-6:
                status = "off_track"
                break
            if self.tilt_time >= .10 - 1e-9:
                status = "rollover"
                break
            if self.overspeed_time >= .10 - 1e-9:
                status = "overspeed"
                break

            self.segment = segments[0]
            lap = self.track.length
            self.arc += (arcs[0] - self.arc + lap / 2.) % lap - lap / 2.
            fraction = self._crossing_fraction(start, self.data.qpos[:2])
            if fraction is not None:
                self.next_gate += 1
            if self.next_gate == len(self.track.gates):
                self.max_progress = 1.
                self.finish_time = start_time + fraction * dt
                status = "finished"
                break
            self.max_progress = max(self.max_progress, float(min(self.arc, self.track.gates[self.next_gate]) / lap))

        if status == "running" and self.data.time >= self.time_limit - 1e-9:
            status = "timeout"
        if status == "physics_error":
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0., True, False, self._info(status)
        obs, lateral, heading, forward = self._observe()
        virtual_steer = math.atan(WHEELBASE * self.vehicle.curvature)
        signals = RewardSignals(
            progress_delta_m=float(self.arc - before_arc),
            new_progress_m=(self.max_progress - before_progress) * self.track.length,
            lateral_error_m=float(lateral), heading_error_rad=heading, speed_mps=forward,
            steering=virtual_steer / self.max_steer,
            steering_change=(virtual_steer - before_steer) / self.max_steer,
            clearance_m=clearance, grip_risk_sq=grip_risk / normal_load if normal_load else 0.,
            dt_s=float(self.data.time - before_time), finished=status == "finished",
            off_track=status == "off_track", timed_out=status == "timeout",
            failed=status in ("off_track", "rollover", "overspeed"),
        )
        terminated = status not in ("running", "timeout")
        return obs, float(self.reward_fn(signals)), terminated, status == "timeout", self._info(status)

    def _crossing_fraction(self, start, end):
        center = self.gate_centers[self.next_gate]
        tangent = self.gate_tangents[self.next_gate]
        a = float((start - center) @ tangent)
        b = float((end - center) @ tangent)
        if a < 0. <= b:
            fraction = -a / (b - a)
            offset = start + fraction * (end - start) - center
            if np.linalg.norm(offset) <= self.track.half_width:
                return fraction
        return None

    def _yaw(self):
        mat = self.data.xmat[self.car]
        return math.atan2(mat[3], mat[0])

    def _observe(self):
        yaw = self._yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        position = self.data.qpos[:2]
        center, idx = self.track.sample(self.arc)
        tangent = self.track.tangents[idx]
        lateral = (position - center) @ np.array([-tangent[1], tangent[0]])
        heading = (yaw - math.atan2(tangent[1], tangent[0]) + math.pi) % (2 * math.pi) - math.pi
        forward, sideways = self.data.qvel[:2] @ rotation
        targets, _ = self.track.sample(self.arc + self.lookahead)
        local = (targets - position) @ rotation / self.lookahead[-1]
        observation = np.r_[local.ravel(), lateral / self.track.half_width,
                            math.sin(heading), math.cos(heading),
                            forward / self.max_speed, sideways / self.max_speed,
                            self.data.qvel[5] / 5.,
                            self.data.qpos[self.vehicle.steer_qpos] / MAX_WHEEL_ANGLE,
                            math.atan(WHEELBASE * self.vehicle.curvature) / self.max_steer,
                            self.vehicle.speed_target / self.max_speed]
        return np.clip(observation, -10., 10.).astype(np.float32), lateral, heading, float(forward)

    def _info(self, status):
        return {"status": status, "progress": self.max_progress,
                "gates": self.next_gate,
                "time_s": self.finish_time if status == "finished" else float(self.data.time),
                "peak_speed_mps": self.peak_speed, "peak_grip": self.peak_grip,
                "peak_slip_mps": self.peak_slip if self.vehicle.diagnostics else None}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=600, width=960)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
