"""A bounded SAC correction trained with a time-to-finish potential."""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from planned_racer import PlannedRace, run_cases, summary
from multi_line_potential import MultiLineTimeToGo, with_speed_envelope
from racecraft_reward import driving_risk_seconds, straight_time_weights
from time_line import TimeLine
from train_racing_line import SegmentTrainingEnv

GAMMA = .9995
FAILURE_SECONDS = 55.
POTENTIAL_WEIGHT = .45
REWARD_SCALE = 5.
SCALE = (.10, .14)


class TimeToGo:
    """Roll out the calibrated speed model on the selected line, backwards from finish."""

    def __init__(self, line, curvature=None, step_weights=None):
        self.arcs = line.arcs
        self.speed_grid = np.linspace(0., 5.15, 36)
        self.times = np.zeros((len(self.arcs), len(self.speed_grid)))
        distance = np.linalg.norm(np.diff(line.closed_points, axis=0), axis=1)
        v = self.speed_grid
        drive = np.interp(v, line.pace.drive[:, 0], line.pace.drive[:, 1])
        brake = np.interp(v, line.pace.brake[:, 0], line.pace.brake[:, 1])
        for i in range(len(distance) - 1, -1, -1):
            reserve = (1. if curvature is None else
                       np.sqrt(1. - np.minimum(.96, .87 * v**2 * abs(curvature[i]) / line.lateral_limit)**2))
            faster = np.sqrt(v**2 + 2. * distance[i] * drive * reserve)
            slower = np.sqrt(np.maximum(0., v**2 - 2. * distance[i] * brake * reserve))
            after = np.clip(line.closed_speeds[i + 1], slower, faster)
            duration = 2. * distance[i] / np.maximum(.15, v + after)
            if step_weights is not None:
                duration *= step_weights[i]
            self.times[i] = duration + np.interp(after, v, self.times[i + 1])

    def potential(self, env, obs):
        last_gate = env.track.gates[min(env.next_gate, len(env.track.gates) - 1)]
        arc = float(np.clip(env.arc, 0., last_gate))
        i = min(int(np.searchsorted(self.arcs, arc, side="right")) - 1, len(self.arcs) - 2)
        fraction = (arc - self.arcs[i]) / (self.arcs[i + 1] - self.arcs[i])
        ground_speed = math.hypot(*env.data.qvel[:2])
        directed_speed = ground_speed * max(0., float(obs[12]))
        low = np.interp(directed_speed, self.speed_grid, self.times[i])
        high = np.interp(directed_speed, self.speed_grid, self.times[i + 1])
        return -float(low + fraction * (high - low))


def time_reward(elapsed, failed, current, following, risk_seconds=0.):
    return (-elapsed - FAILURE_SECONDS * failed - risk_seconds
            + POTENTIAL_WEIGHT * (GAMMA * following - current)) / REWARD_SCALE


class TimeRewardRace(PlannedRace):
    def __init__(self, line, multi_line=False, residual_scale=SCALE, lane_limit=0., env=None, racecraft=False):
        super().__init__(SegmentTrainingEnv() if env is None else env, line, residual_scale, lane_limit)
        self.racecraft = racecraft
        track = self.env.unwrapped.track
        if racecraft:
            self.step_weights = straight_time_weights(track, line.arcs)
            self.grip_estimates = {}
            self.env.unwrapped.reward_fn = driving_risk_seconds
        else:
            self.estimate = MultiLineTimeToGo(track, line, TimeToGo) if multi_line else TimeToGo(line)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        car = self.env.unwrapped
        if self.racecraft:
            # The acceleration curves were measured on the published dry surface.
            grip = min(1., float(car.model.pair_friction[:, :2].min()) / .90)
            if grip not in self.grip_estimates:
                line = self.line if grip == 1. else with_speed_envelope(
                    self.line, self.line.points, grip * self.line.lateral_limit)
                self.grip_estimates[grip] = MultiLineTimeToGo(
                    car.track, line, partial(TimeToGo, step_weights=self.step_weights), self.step_weights)
            self.estimate = self.grip_estimates[grip]
        self.phi = self.estimate.potential(car, obs)
        self.clock = info["time_s"]
        return obs, info

    def step(self, action):
        obs, measured_cost, terminated, truncated, info = super().step(action)
        if info["status"] == "physics_error":
            raise RuntimeError("Physics error during time-reward training")
        finished = info["status"] == "finished"
        end = terminated or truncated
        phi = 0. if end else self.estimate.potential(self.env.unwrapped, obs)
        # The base environment returns a negative sentinel on invalid actions.
        risk = max(0., measured_cost) if self.racecraft else 0.
        reward = time_reward(info["time_s"] - self.clock, end and not finished, self.phi, phi, risk)
        self.phi, self.clock = phi, info["time_s"]
        # The 48 s race deadline is a failure in this finite-horizon objective.
        return obs, reward, end, False, info


def make_env(path, monitor, multi_line=False, racecraft=False):
    torch.set_num_threads(1)
    return Monitor(TimeRewardRace(TimeLine.load(path), multi_line, racecraft=racecraft), str(monitor),
                   info_keywords=("status", "progress", "time_s"))


class Snapshots(BaseCallback):
    def __init__(self, output, line, residual_scale=SCALE, lane_limit=0., screen_seed=6200):
        super().__init__()
        self.residual_scale, self.lane_limit, self.screen_seed = residual_scale, lane_limit, screen_seed
        self.output, self.line = output, line
        self.next_at = 32768
        self.rows = []

    def measure(self, name):
        dry = run_cases("sac", range(self.screen_seed, self.screen_seed + 8), line=self.line, checkpoint=self.model,
                        residual_scale=self.residual_scale, lane_limit=self.lane_limit)
        stress = run_cases("sac", range(self.screen_seed + 50, self.screen_seed + 56), line=self.line, checkpoint=self.model,
                           mu=.82, residual_scale=self.residual_scale, lane_limit=self.lane_limit)
        self.model.save(self.output / name)
        row = {"name": name, "steps": self.num_timesteps, "dry": summary(dry), "stress": summary(stress)}
        self.rows.append(row)
        (self.output / "validation.json").write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        print(row, flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_at:
            self.measure(f"step_{self.num_timesteps}")
            self.next_at += 32768
        return True

    def _on_training_end(self):
        self.measure("final")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=196608)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    rewards = parser.add_mutually_exclusive_group()
    rewards.add_argument("--multi-line", action="store_true")
    rewards.add_argument("--racecraft", action="store_true", help="Emphasize corner exits and measured instability")
    args = parser.parse_args()
    torch.set_num_threads(1)
    line = TimeLine.load(args.line)
    args.out.mkdir(parents=True)
    config = {"layout": "shanghai", "algorithm": "SAC", "gamma": GAMMA, "seed": args.seed,
              "steps": args.steps, "residual_scale": SCALE, "multi_line": args.multi_line or args.racecraft,
              "racecraft": args.racecraft, "failure_seconds": FAILURE_SECONDS,
              "reward_scale": REWARD_SCALE, "potential_weight": POTENTIAL_WEIGHT, "line": str(args.line)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    vec = SubprocVecEnv([partial(make_env, args.line, args.out / str(i), args.multi_line, args.racecraft) for i in range(args.envs)])
    start = perf_counter()
    try:
        model = SAC("MlpPolicy", vec, learning_rate=1.5e-4, buffer_size=250000, learning_starts=4096,
                    batch_size=256, gamma=GAMMA, train_freq=4, gradient_steps=4,
                    ent_coef="auto_0.003", device="cpu", seed=args.seed,
                    policy_kwargs={"net_arch": [96, 96]}, verbose=0)
        # Start from the calibrated controller and learn bounded corrections.
        with torch.no_grad():
            model.actor.mu.weight.zero_()
            model.actor.mu.bias.zero_()
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(math.log(.20))
        print("Starting time-potential SAC", flush=True)
        model.learn(total_timesteps=args.steps, callback=Snapshots(args.out, line))
        (args.out / "runtime.json").write_text(
            json.dumps({"steps": model.num_timesteps, "wall_seconds": perf_counter() - start}, indent=2),
            encoding="utf-8")
    finally:
        vec.close()


if __name__ == "__main__":
    main()
