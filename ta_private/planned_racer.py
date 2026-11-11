"""Private map-aware residual driver. The MuJoCo action interface stays two-dimensional."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import SAC, TD3

from reference import load_policy
from racing import RaceEnv
from time_line import TimeLine


RESIDUAL_SCALE = np.array([.29, .32], dtype=np.float32)
LANE_RATE = 1.25


class PlannedRace(gym.Wrapper):
    def __init__(self, env, line, residual_scale=RESIDUAL_SCALE, lane_limit=0., lane_gain=1.):
        super().__init__(env)
        if abs(env.unwrapped.track.length - line.length) > .01:
            raise ValueError("Reference line is for a different circuit")
        self.line = line
        self.residual_scale = np.asarray(residual_scale, dtype=np.float32)
        self.lane_limit = lane_limit
        self.lane_gain = lane_gain
        self.lane_fraction = 0.
        self.observation_space = gym.spaces.Box(-10., 10., (36 + bool(lane_limit),), np.float32)
        if lane_limit:
            self.action_space = gym.spaces.Box(-1., 1., (3,), np.float32)

    def _augment(self, obs):
        self.base_action, context = self.line.reference(self.env.unwrapped, self.lane_fraction, self.lane_limit)
        if self.lane_limit:
            context = np.append(context, self.lane_fraction)
        return np.clip(np.concatenate((obs, context)), -10., 10.).astype(np.float32)

    def reset(self, **kwargs):
        self.lane_fraction = 0.
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != self.action_space.shape or not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1.000001:
            command = [np.nan, np.nan]
        else:
            if self.lane_limit:
                change = LANE_RATE * self.env.unwrapped.control_dt
                request = float(residual[2]) * self.lane_gain
                self.lane_fraction += float(np.clip(request - self.lane_fraction, -change, change))
                self.base_action, _ = self.line.reference(self.env.unwrapped, self.lane_fraction, self.lane_limit)
            command = np.clip(self.base_action + self.residual_scale * residual[:2], -1., 1.)
        self.last_command = command
        obs, reward, terminated, truncated, info = self.env.step(command)
        return self._augment(obs), reward, terminated, truncated, info


class SectorClock:
    def __init__(self, track, sectors):
        self.sectors = tuple(sectors)
        self.arcs = np.unique([arc for _, entry, exit in self.sectors for arc in (entry, exit)])
        self.points, indices = track.sample(self.arcs)
        self.tangents = track.tangents[indices]
        self.width = track.half_width
        self.times = []

    def record(self, before, after, time_before, time_after, arc):
        i = len(self.times)
        if i >= len(self.arcs) or abs(arc - self.arcs[i]) > 1.5:
            return
        center, tangent = self.points[i], self.tangents[i]
        first = float((before - center) @ tangent)
        last = float((after - center) @ tangent)
        if first < 0. <= last:
            fraction = -first / (last - first)
            crossing = before + fraction * (after - before)
            if np.linalg.norm(crossing - center) <= self.width + .02:
                self.times.append(time_before + fraction * (time_after - time_before))

    def results(self):
        crossings = dict(zip(self.arcs, self.times))
        return {name: crossings[exit] - crossings[entry] for name, entry, exit in self.sectors
                if entry in crossings and exit in crossings}


def run_cases(mode, seeds, *, layout="shanghai", line=None, checkpoint=None, mu=None,
              replay=None, trace=False, camera="overview", residual_scale=RESIDUAL_SCALE, sectors=(), lane_limit=0., lane_gain=1.):
    env = RaceEnv(layout=layout, render_mode="rgb_array" if replay else None, camera=camera)
    if mu is not None:
        env.model.pair_friction[:, :2] = mu
    if mode in ("planner", "sac", "td3"):
        env = PlannedRace(env, line, residual_scale, lane_limit, lane_gain)
    if mode in ("sac", "td3"):
        algorithm = SAC if mode == "sac" else TD3
        model = algorithm.load(checkpoint, device="cpu") if isinstance(checkpoint, (str, Path)) else checkpoint
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    elif mode == "ppo":
        policy = load_policy(checkpoint, "ppo")
    elif mode == "planner":
        policy = lambda _: np.zeros(2, np.float32)
    else:
        raise ValueError(f"Unknown driver: {mode}")

    episodes, frames, telemetry = [], [], []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            car = env.unwrapped
            clock = SectorClock(car.track, sectors) if sectors else None
            first = not episodes
            step = 0
            while info["status"] == "running":
                action = policy(obs)
                if first and trace:
                    car = env.unwrapped
                    if mode in ("planner", "sac", "td3"):
                        command = np.clip(env.base_action + env.residual_scale * action[:2], -1., 1.)
                        planned_speed = (env.base_action[1] + 1.) * .5 * car.max_speed
                    else:
                        command = action
                        planned_speed = None
                    center, index = car.track.sample(car.arc)
                    tangent = car.track.tangents[index]
                    lateral = float((car.data.qpos[:2] - center) @ (tangent[::-1] * [-1., 1.]))
                    telemetry.append({"lateral_m": lateral, "t": float(car.data.time), "s": float(car.arc), "x": float(car.data.qpos[0]),
                                      "y": float(car.data.qpos[1]),
                                      "v": float(np.linalg.norm(car.data.qvel[:2])),
                                      "requested_speed": float((command[1] + 1.) * .5 * car.max_speed),
                                      "planned_speed": float(planned_speed) if planned_speed is not None else None,
                                      "steering": float(command[0])})
                if replay and first and step % 2 == 0:
                    frame = Image.fromarray(env.unwrapped.render()).resize((540, 338))
                    frames.append(frame.quantize(colors=80, dither=Image.Dither.NONE))
                if clock is not None:
                    time_before = car.data.time
                    before = car.data.qpos[:2].copy()
                obs, _, _, _, info = env.step(action)
                if first and trace and lane_limit:
                    telemetry[-1].update(lane_request=float(action[2]), lane_fraction=env.lane_fraction,
                                         requested_speed=float((env.last_command[1] + 1.) * .5 * car.max_speed),
                                         steering=float(env.last_command[0]))
                if clock is not None:
                    clock.record(before, car.data.qpos[:2], time_before, car.data.time, car.arc)
                step += 1
                if info["status"] == "physics_error":
                    raise RuntimeError(f"MuJoCo failed on {layout}, seed {seed}")
            if clock is not None:
                info["sectors_s"] = clock.results()
            episodes.append({"seed": seed, **info})
    finally:
        env.close()

    if replay:
        frames[0].save(replay, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    times = [ep["time_s"] for ep in episodes if ep["status"] == "finished"]
    result = {"finish_rate": len(times) / len(episodes),
              "mean_progress": float(np.mean([ep["progress"] for ep in episodes])),
              "median_lap_s": float(np.median(times)) if times else None, "episodes": episodes}
    if trace:
        result["telemetry"] = telemetry
    return result


def summary(result):
    return {k: v for k, v in result.items() if k not in ("episodes", "telemetry")}
