"""Train a policy with the reward you implemented in racing/reward.py."""

import argparse
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from evaluate import ALGORITHMS, evaluate, ranking
from racing import RaceEnv
from racing.reward import reward


class SaveBest(BaseCallback):
    def __init__(self, output, seeds, every):
        super().__init__()
        self.output, self.seeds, self.every = output, seeds, every
        self.next_eval = every
        self.best = (-float("inf"),)
        self.history = []

    def measure(self):
        result = evaluate(lambda obs: self.model.predict(obs, deterministic=True)[0], self.seeds)
        result = {"steps": self.num_timesteps, **result}
        self.history.append(result)
        if ranking(result) > self.best:
            self.best = ranking(result)
            self.model.save(self.output / "best")
        (self.output / "validation.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print({k: v for k, v in result.items() if k != "episodes"}, flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.measure()
            self.next_eval += self.every
        return True

    def _on_training_end(self):
        self.measure()


def make_env(reward_fn, monitor_file):
    return Monitor(RaceEnv(reward_fn=reward_fn), str(monitor_file),
                   info_keywords=("status", "progress", "time_s"))


def train(args, reward_fn=reward, seeds=tuple(range(20, 25)), overrides=None):
    if min(args.steps, args.envs, args.eval_every) < 1:
        raise ValueError("steps, envs and eval-every must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    torch.set_num_threads(1)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    settings = dict(policy_kwargs=dict(net_arch=[64, 64]), learning_rate=3e-4, gamma=.99)
    if args.algo == "ppo":
        settings.update(n_steps=512, batch_size=256, n_epochs=6)
    else:
        settings.update(buffer_size=200_000, learning_starts=2048, batch_size=256,
                        train_freq=4, gradient_steps=4)
    settings.update(overrides or {})
    args.out.mkdir(parents=True)
    config = {**vars(args), "out": str(args.out), "device_used": device, "settings": settings,
              "validation_seeds": list(seeds)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    factories = [partial(make_env, reward_fn, args.out / str(i)) for i in range(args.envs)]
    vec_env = SubprocVecEnv(factories) if args.parallel else DummyVecEnv(factories)
    try:
        model = ALGORITHMS[args.algo]("MlpPolicy", vec_env, seed=args.seed,
                                      device=device, verbose=0, **settings)
        callback = SaveBest(args.out, seeds, args.eval_every)
        print(f"{args.algo.upper()} on {model.device}; {args.envs} environments", flush=True)
        start = perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(args.out / "final")
        run = {"steps": model.num_timesteps, "wall_seconds": perf_counter() - start}
        (args.out / "runtime.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(run, flush=True)
    finally:
        vec_env.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=ALGORITHMS, default="ppo")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--parallel", action="store_true", help="Run the physics in separate worker processes")
    p.add_argument("--steps", type=int, default=204_800)
    p.add_argument("--eval-every", type=int, default=20_480)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, required=True, help="New run directory")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
