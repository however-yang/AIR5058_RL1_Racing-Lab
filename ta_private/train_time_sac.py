"""Train a Shanghai residual SAC driver around an independently planned racing line."""

import argparse
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from planned_racer import PlannedRace, run_cases, summary
from evaluate import ranking
from racing import RaceEnv
from reference import progress_reward
from train_racing_line import SegmentTrainingEnv
from time_line import TimeLine


SEEDS = tuple(range(110, 118))


def time_reward(signals):
    edge = max(0., min(1., (.13 - signals.clearance_m) / .13))
    return (progress_reward(signals) - signals.dt_s * (1.0 + .8 * edge**2)
            - 4. * signals.failed - .005 * signals.steering_change**2)


def make_env(line_path, monitor, algorithm):
    env = SegmentTrainingEnv()
    if algorithm == "td3":
        env.reward_fn = time_reward
    return Monitor(PlannedRace(env, TimeLine.load(line_path)), str(monitor),
                   info_keywords=("status", "progress", "time_s"))


class ShanghaiCheckpoints(BaseCallback):
    def __init__(self, path, line, every, algorithm):
        super().__init__()
        self.path, self.line, self.every = path, line, every
        self.algorithm = algorithm
        self.next = every
        self.records = []
        self.best = None

    def measure(self):
        result = run_cases(self.algorithm, SEEDS, line=self.line, checkpoint=self.model)
        row = {"steps": self.num_timesteps, **summary(result)}
        self.records.append(row)
        # Keep all robust checkpoints, so selection can use independent starts.
        if row["finish_rate"] >= .875:
            self.model.save(self.path / f"checkpoint_{self.num_timesteps}")
        if self.best is None or ranking(row) > ranking(self.best):
            self.best = row
            self.model.save(self.path / "best")
        (self.path / "validation.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        print("Shanghai:", row, flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.measure()
            self.next += self.every
        return True

    def _on_training_end(self):
        self.measure()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--line", type=Path)
    parser.add_argument("--steps", type=int, default=327680)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--algo", choices=("sac", "td3"), default="sac")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lateral-limit", type=float, default=7.1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True)
    if args.line:
        line = TimeLine.load(args.line)
    else:
        env = RaceEnv()
        line = TimeLine(env.track, args.lateral_limit)
        env.close()
    path = args.out / "shanghai_line.npz"
    line.save(path)
    control = run_cases("planner", range(6), line=line)
    (args.out / "planner.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    print("Planner estimated:", line.estimated_time, "actual:", summary(control), flush=True)
    if args.planner_only:
        return

    cfg = {"layout": "shanghai", "algorithm": args.algo.upper(), "seed": args.seed, "steps": args.steps,
           "device": args.device, "envs": args.envs, "net_arch": [96, 96],
           "lateral_limit": line.lateral_limit, "observation": 36,
           "reward": "time" if args.algo == "td3" else "line_risk"}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    vec = SubprocVecEnv([partial(make_env, path, args.out / str(i), args.algo) for i in range(args.envs)])
    start = perf_counter()
    try:
        settings = dict(learning_rate=3e-4, buffer_size=350000, learning_starts=4096,
                        batch_size=256, gamma=.995, train_freq=4, gradient_steps=4,
                        device=args.device, seed=args.seed, policy_kwargs={"net_arch": [96, 96]},
                        verbose=0)
        if args.algo == "sac":
            model = SAC("MlpPolicy", vec, ent_coef="auto_0.02", **settings)
        else:
            settings.update(gamma=.998, learning_rate=2.5e-4)
            noise = NormalActionNoise(mean=np.zeros(2), sigma=np.array([.17, .21]))
            model = TD3("MlpPolicy", vec, action_noise=noise, target_policy_noise=.12,
                        target_noise_clip=.25, **settings)
            with torch.no_grad():
                model.actor.mu[-2].weight.zero_()
                model.actor.mu[-2].bias.zero_()
                model.actor_target.load_state_dict(model.actor.state_dict())
        print(f"Starting residual {args.algo.upper()}", flush=True)
        model.learn(total_timesteps=args.steps, callback=ShanghaiCheckpoints(args.out, line, 32768, args.algo))
        model.save(args.out / "final")
        (args.out / "runtime.json").write_text(
            json.dumps({"steps": model.num_timesteps, "wall_seconds": perf_counter() - start}, indent=2),
            encoding="utf-8")
    finally:
        vec.close()


if __name__ == "__main__":
    main()
