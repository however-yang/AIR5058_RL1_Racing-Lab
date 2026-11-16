"""Student interface for the unchanged MuJoCo racing environment."""

import math

import numpy as np

from racing import RaceEnv
from racing.vehicle import MAX_STEER, WHEELBASE
from scoring import BendGrip
from timing import SPLIT_SECTORS, SectorClock


SURFACES = {"dry": .90, "low-grip": .82}
BASELINE_SPEED = 3.40


class InvalidAction(ValueError):
    """The policy returned an out-of-range or malformed action."""


class Lab:
    """Actions are (lateral offset in m, target speed in m/s)."""

    def __init__(self, surface="dry", reward_fn=None, measure_grip=False):
        kwargs = {} if reward_fn is None else {"reward_fn": reward_fn}
        self._env = RaceEnv(diagnostics=True, **kwargs)
        self._env.model.pair_friction[:, :2] = SURFACES[surface]
        self._grip = BendGrip(self._env, SURFACES[surface]) if measure_grip else None
        if self._grip is not None:
            self._env.step_observer = self._grip.record
        self.half_width = self._env.track.half_width
        self.max_speed = self._env.max_speed
        self.control_dt = self._env.control_dt

    def reset(self, seed=None, *, capture_profile=False):
        obs, info = self._env.reset(seed=seed)
        self._clock = SectorClock(self._env.track, SPLIT_SECTORS)
        if self._grip is not None:
            self._grip.reset(capture=capture_profile)
        return obs, info

    def step(self, action):
        try:
            offset, speed = map(float, action)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidAction("Expected two numeric action fields") from exc
        if not math.isfinite(offset) or not -self.half_width <= offset <= self.half_width:
            raise InvalidAction("Offset must be finite and within the road half-width")
        if not math.isfinite(speed) or not 0. <= speed <= self.max_speed:
            raise InvalidAction("Speed must be finite and within [0, 5] m/s")
        env = self._env
        before, start_time = env.data.qpos[:2].copy(), env.data.time
        # Same geometric preview executor for every policy and for the baseline.
        speed_now = math.hypot(*env.data.qvel[:2])
        point, index = env.track.sample(env.arc + .95 + .18 * speed_now)
        tx, ty = env.track.tangents[index]
        delta = point + offset * np.array([-ty, tx]) - before
        yaw = env._yaw()
        across = delta @ np.array([-math.sin(yaw), math.cos(yaw)])
        steer = math.atan2(2. * WHEELBASE * across, delta @ delta)
        command = (float(np.clip(steer / MAX_STEER, -1., 1.)), 2. * speed / self.max_speed - 1.)
        obs, reward, term, trunc, info = env.step(command)
        self._clock.record(before, env.data.qpos[:2], start_time, env.data.time, env.arc)
        if term or trunc:
            info["sectors_s"] = self._clock.results()
            if self._grip is not None:
                info["bend_grip_fraction"] = self._grip.fraction() if info["status"] == "finished" else 0.
        return obs, reward, term, trunc, info

    def performance(self):
        return self._grip.profile(self.max_speed) if self._grip is not None else None

    def close(self):
        self._env.close()


def baseline(_observation):
    return 0., BASELINE_SPEED
