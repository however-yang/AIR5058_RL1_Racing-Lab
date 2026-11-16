"""Race grading from uniform first-pass arc samples and MuJoCo wheel forces."""

import math
from statistics import mean

import mujoco
import numpy as np


CURVATURE_MIN = .14  # 1/m; applied to the centerline, never to the student's path
PACE_GAIN_FOR_FULL_CREDIT = .15
MIN_FINISH_RATE = .95
SLIP_FULL = math.radians(8)
SLIP_ZERO = math.radians(16)


class BendGrip:
    def __init__(self, env, friction):
        track = env.track
        count = math.ceil(track.length / .10)
        arcs = (np.arange(count) + .5) * track.length / count
        _, before = track.sample(arcs - .55)
        _, after = track.sample(arcs + .55)
        a, b = track.tangents[before], track.tangents[after]
        turn = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))
        bend = np.abs(turn) / 1.10 >= CURVATURE_MIN
        self.grid = arcs
        self.bend = bend
        self.straight = np.abs(turn) / 1.10 <= .05
        self.arcs = arcs[bend]
        _, at = track.sample(self.arcs)
        forward = track.tangents[at]
        self.road_forward = forward
        self.inward = np.sign(turn[bend, None]) * np.column_stack((-forward[:, 1], forward[:, 0]))
        if len(self.arcs) == 0:
            raise ValueError("No grading bends for this layout")
        gravity = float(np.linalg.norm(env.model.opt.gravity))
        self.capacity_floor = .75 * friction * mujoco.mj_getTotalmass(env.model) * gravity
        self.force = np.zeros(6)
        self.reset()

    def reset(self, capture=False):
        self.next_sample = 0
        self.total = 0.
        self.speed_index = 0
        self.speed = [None] * len(self.grid) if capture else None
        self.target = [None] * len(self.grid) if capture else None
        self.effective = [None] * len(self.arcs) if capture else None
        self.raw = [None] * len(self.arcs) if capture else None

    def profile(self, max_speed):
        if self.speed is None:
            return None
        straight_speed = [v for v, straight in zip(self.speed, self.straight)
                          if straight and v is not None]
        bends = [v for v in self.effective if v is not None]
        return {
            "max_speed_mps": max_speed,
            "straight_peak_mps": max(straight_speed, default=None),
            "straight_at_95pct_fraction": (
                mean(v >= .95 * max_speed for v in straight_speed) if straight_speed else None),
            "bend_peak_effective_fraction": max(bends, default=None),
            "bend_at_90pct_fraction": mean(v >= .90 for v in bends) if bends else None,
            "straight_samples_seen": len(straight_speed),
            "straight_samples_total": int(self.straight.sum()),
            "bend_samples_seen": len(bends),
            "bend_samples_total": len(self.arcs),
            "arc_m": self.grid.tolist(),
            "straight": self.straight.tolist(),
            "bend": self.bend.tolist(),
            "speed_mps": self.speed,
            "target_mps": self.target,
            "bend_arc_m": self.arcs.tolist(),
            "effective_grip": self.effective,
            "inward_grip_before_slip_discount": self.raw,
        }

    def record(self, env):
        if self.speed is not None:
            end = int(np.searchsorted(self.grid, env.arc, side="right"))
            if end > self.speed_index:
                speed = math.hypot(*env.data.qvel[:2])
                target = float(env.vehicle.speed_target)
                self.speed[self.speed_index:end] = [speed] * (end - self.speed_index)
                self.target[self.speed_index:end] = [target] * (end - self.speed_index)
                self.speed_index = end
        # Samples are consumed even if the car was sliding, reversing, or unloaded.
        stop = int(np.searchsorted(self.arcs, env.arc, side="right"))
        if stop <= self.next_sample:
            return
        indices = slice(self.next_sample, stop)
        inward = self.inward[indices]
        self.next_sample = stop
        vx, vy = env.data.qvel[:2]
        yaw = env._yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        forward, lateral = vx * c + vy * s, -vx * s + vy * c
        slip = abs(math.atan2(lateral, forward))
        quality = (max(0., min(1., (SLIP_ZERO - slip) / (SLIP_ZERO - SLIP_FULL)))
                   if forward > 0. else 0.)
        if quality == 0. and self.effective is None:
            return
        lateral_force = np.zeros(2)
        capacity = 0.
        floor, tires = env.vehicle.floor, env.vehicle.tire_body
        for index, contact in enumerate(env.data.contact):
            if contact.efc_address < 0:
                continue
            if contact.geom1 == floor and contact.geom2 in tires:
                direction = 1.
            elif contact.geom2 == floor and contact.geom1 in tires:
                direction = -1.
            else:
                continue
            mujoco.mj_contactForce(env.model, env.data, index, self.force)
            normal = self.force[0]
            if normal <= .10:
                continue
            axes = contact.frame.reshape(3, 3)
            lateral_force += direction * (self.force[1:3] @ axes[1:, :2])
            capacity += normal * contact.friction[0]
        forward_progress = self.road_forward[indices] @ np.array([vx, vy]) > 0.
        useful = (inward @ lateral_force) * forward_progress
        raw = np.clip(useful / max(self.capacity_floor, capacity), 0., 1.)
        self.total += quality * float(raw.sum())
        if self.effective is not None:
            self.effective[indices] = (quality * raw).tolist()
            self.raw[indices] = raw.tolist()

    def fraction(self):
        return self.total / len(self.arcs)


def grade(reference, policy):
    """Main event only. Every assigned seed is included, failures contribute zero."""
    if not reference or [lap["seed"] for lap in reference] != [lap["seed"] for lap in policy]:
        raise ValueError("Grading requires non-empty paired episodes in the same seed order")
    if any(lap["status"] != "finished" for lap in reference):
        return {"eligible": False, "reason": "official_baseline_invalid", "total": None}
    if mean(lap["status"] == "finished" for lap in policy) < MIN_FINISH_RATE:
        return {"eligible": False, "reason": "completion_below_95_percent",
                "pace_points": 0., "grip_points": 0., "total": 0.}
    pace = []
    grip = []
    for base, lap in zip(reference, policy):
        if lap["status"] == "finished":
            saved = (base["time_s"] - lap["time_s"]) / base["time_s"]
            pace.append(min(1., max(0., saved / PACE_GAIN_FOR_FULL_CREDIT)))
            grip.append(lap["bend_grip_fraction"])
        else:
            pace.append(0.)
            grip.append(0.)
    pace_points = 70. * mean(pace)
    grip_points = 30. * mean(grip)
    return {"eligible": True, "pace_points": pace_points,
            "grip_points": grip_points, "total": pace_points + grip_points}
