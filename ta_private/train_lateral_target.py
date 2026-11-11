"""Fine-tune matched multi-line SAC policies with and without a lateral target."""

import argparse
from functools import partial
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from train_remaining_time import GAMMA, Snapshots, TimeRewardRace
from experiments.evaluation import model_fingerprint, sha
from time_line import TimeLine
from train_racing_line import SegmentTrainingEnv


class MixedGripShanghai(SegmentTrainingEnv):
    def __init__(self, phase):
        super().__init__()
        self.episode = phase
        self.dry = self.model.pair_friction[:, :2].copy()

    def reset(self, **kwargs):
        self.model.pair_friction[:, :2] = .82 if self.episode % 4 == 0 else self.dry
        self.episode += 1
        return super().reset(**kwargs)


def worker(path, monitor, residual_scale, lane_limit, phase):
    torch.set_num_threads(1)
    env = TimeRewardRace(TimeLine.load(path), multi_line=True, residual_scale=residual_scale,
                         lane_limit=lane_limit, env=MixedGripShanghai(phase))
    return Monitor(env, str(monitor), info_keywords=("status", "progress", "time_s"))


def inherit_policy(model, previous):
    old_width = previous.observation_space.shape[0]
    with torch.no_grad():
        original = previous.policy.state_dict()
        for name, target in model.policy.state_dict().items():
            old = original[name]
            if target.shape == old.shape:
                target.copy_(old)
            elif name == "actor.latent_pi.0.weight":
                target.zero_()
                target[:, :old_width].copy_(old)
            elif name in ("actor.mu.weight", "actor.mu.bias", "actor.log_std.weight", "actor.log_std.bias"):
                target.zero_()
                target[:2].copy_(old)
                if name == "actor.log_std.bias":
                    target[2] = math.log(.18)
            elif name.startswith(("critic.", "critic_target.")) and name.endswith(".0.weight"):
                target.zero_()
                target[:, :old_width].copy_(old[:, :old_width])
                target[:, old_width + 1:old_width + 3].copy_(old[:, old_width:])
            else:
                raise ValueError(f"Unexpected parameter shape for {name}: {target.shape}")

    # Existing controls must initially produce the same deterministic commands.
    observations = np.random.default_rng(219).normal(size=(8, old_width)).astype(np.float32)
    with_target = model.action_space.shape != previous.action_space.shape
    inputs = np.pad(observations, ((0, 0), (0, 1))) if with_target else observations
    before = previous.predict(observations, deterministic=True)[0]
    after = model.predict(inputs, deterministic=True)[0]
    np.testing.assert_allclose(after[:, :2], before, atol=2e-6, rtol=0.)
    if with_target:
        np.testing.assert_array_equal(after[:, 2], 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=131072)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument("--lateral-target", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    frozen = json.loads((args.previous / "frozen_selection.json").read_text(encoding="utf-8"))
    choice = frozen["selected"]["multi_line"]
    line_path = args.previous / "multi_line" / "shanghai_line.npz"
    checkpoint = args.previous / "multi_line" / "policy.zip"
    fingerprints = frozen["sha256"]["multi_line"]
    if (sha(line_path) != fingerprints["line"] or sha(checkpoint) != fingerprints["checkpoint"]
            or model_fingerprint() != frozen["physics"]):
        raise RuntimeError("The prior controller or vehicle files differ from the frozen run")
    scale = choice["residual_scale"]
    lane_limit = .65 if args.lateral_target else 0.
    config = {"steps": args.steps, "envs": args.envs, "seed": args.seed, "lane_limit": lane_limit,
              "residual_scale": scale, "low_grip_every": 4, "low_grip_mu": .82,
              "source": fingerprints, "learning_rate": 7.5e-5}
    args.out.mkdir(parents=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    line = TimeLine.load(line_path)
    previous = SAC.load(checkpoint, device="cpu")
    vec = SubprocVecEnv([partial(worker, line_path, args.out / str(i), scale, lane_limit, i)
                         for i in range(args.envs)])
    start = perf_counter()
    try:
        model = SAC("MlpPolicy", vec, learning_rate=config["learning_rate"], buffer_size=200000,
                    learning_starts=1024, batch_size=256, gamma=GAMMA, train_freq=4, gradient_steps=4,
                    ent_coef="auto_0.0015", target_entropy=-2., device="cpu", seed=args.seed,
                    policy_kwargs={"net_arch": [96, 96]}, verbose=0)
        inherit_policy(model, previous)
        print("Matched starting policy verified", "lateral target" if lane_limit else "fixed line", flush=True)
        model.learn(total_timesteps=args.steps, callback=Snapshots(args.out, line, scale, lane_limit, 8100))
        (args.out / "runtime.json").write_text(
            json.dumps({"steps": model.num_timesteps, "wall_seconds": perf_counter() - start}, indent=2),
            encoding="utf-8")
    finally:
        vec.close()


if __name__ == "__main__":
    main()
