"""Imitate a Shanghai racing-line teacher, then refine a standalone PPO driver."""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from reference import VALIDATION_SEEDS, progress_reward
from racing import RaceEnv
from train import SaveBest
from racing.vehicle import TIRE_RADIUS, WHEELS
from racing_teacher import RacingTeacher


class SegmentTrainingEnv(RaceEnv):
    def __init__(self):
        super().__init__(reward_fn=line_reward, layout="shanghai")
        self.wheel_dofs = [self.model.joint(wheel).dofadr.item() for wheel in WHEELS]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.np_random.random() >= .40:
            return obs, info
        self.arc = self.np_random.uniform(5., self.track.length - 10.)
        center, self.segment = self.track.sample(self.arc)
        tangent = self.track.tangents[self.segment]
        offset = self.np_random.uniform(-.35, .35)
        yaw = math.atan2(tangent[1], tangent[0]) + self.np_random.uniform(-.16, .16)
        speed = self.np_random.uniform(0., 2.2)
        self.data.qpos[:2] = center + offset * np.array([-tangent[1], tangent[0]])
        self.data.qpos[3:7] = [math.cos(yaw / 2.), 0., 0., math.sin(yaw / 2.)]
        self.data.qvel[:2] = speed * np.array([math.cos(yaw), math.sin(yaw)])
        self.data.qvel[self.wheel_dofs] = speed / TIRE_RADIUS
        self.vehicle.speed_target = speed
        self.next_gate = int(np.searchsorted(self.track.gates, self.arc, side="right"))
        self.max_progress = self.arc / self.track.length
        mujoco.mj_forward(self.model, self.data)
        return self._observe()[0], self._info("running")


def line_reward(s):
    edge = max(0., min(1., (.24 - s.clearance_m) / .24))
    return progress_reward(s) - s.dt_s * (1.5 * edge**2 + .45 * s.grip_risk_sq) - .010 * s.steering_change**2


def make_env(log):
    return Monitor(SegmentTrainingEnv(), str(log), info_keywords=("status", "progress", "time_s"))


def check_teacher(teacher, seeds):
    env = RaceEnv()
    episodes = []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(teacher.action(env))
                done = term or trunc
            episodes.append({"seed": seed, **info})
    finally:
        env.close()
    return episodes


def demonstrations(model, teacher, env, rng, samples, learner_fraction):
    obs, _ = env.reset(seed=int(rng.integers(2**31)))
    inputs, labels = [], []
    for _ in range(samples):
        label = teacher.action(env)
        inputs.append(obs.copy())
        labels.append(label)
        action = model.predict(obs, deterministic=True)[0] if rng.random() < learner_fraction else label
        action = np.clip(action + rng.normal(0., [.06, .075]), -1., 1.)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.asarray(inputs), np.asarray(labels)


def imitate(model, teacher, seed):
    generator = np.random.default_rng(seed)
    env = SegmentTrainingEnv()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.001)
    weights = torch.tensor([3., 1.], device=model.device)
    datasets = []
    try:
        for samples, learner_fraction in [(14000, 0.), (8000, .35), (8000, .60)]:
            datasets.append(demonstrations(model, teacher, env, generator, samples, learner_fraction))
            x = torch.as_tensor(np.concatenate([row[0] for row in datasets]), device=model.device)
            y = torch.as_tensor(np.concatenate([row[1] for row in datasets]), device=model.device)
            for _ in range(15):
                for idx in generator.permutation(len(x)).reshape(-1, 200):
                    means = model.policy.get_distribution(x[idx]).distribution.mean
                    loss = (weights * torch.nn.functional.smooth_l1_loss(means, y[idx], reduction="none")).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print(f"Imitation samples: {len(x)}; loss={loss.item():.5f}", flush=True)
    finally:
        env.close()

    with torch.no_grad():
        model.policy.log_std[:] = torch.tensor([math.log(.26), math.log(.34)], device=model.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--lateral-budget", type=float, default=5.9)
    parser.add_argument("--steps", type=int, default=409600)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prototype = RaceEnv()
    teacher = RacingTeacher(prototype.track, args.lateral_budget)
    prototype.close()
    baseline = check_teacher(teacher, range(6))
    print("Teacher:", [(ep["status"], round(ep["time_s"], 2), round(ep["progress"], 3)) for ep in baseline], flush=True)
    if args.teacher_only:
        return
    if args.out is None:
        parser.error("--out is required for training")
    args.out.mkdir(parents=True)
    config = {**vars(args), "out": str(args.out), "algo": "ppo", "profile": "imitation_racing",
              "layout": "shanghai", "network": [96, 96], "gamma": .995,
              "curriculum_fraction": .40, "validation_seeds": list(VALIDATION_SEEDS)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "teacher.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    workers = SubprocVecEnv([partial(make_env, args.out / str(i)) for i in range(args.envs)])
    start = perf_counter()
    try:
        model = PPO("MlpPolicy", workers, device="cpu", seed=args.seed, learning_rate=1.4e-4,
                    gamma=.995, n_steps=512, batch_size=256, n_epochs=6, clip_range=.15,
                    policy_kwargs={"net_arch": [96, 96]}, verbose=0)
        imitate(model, teacher, args.seed)
        model.save(args.out / "imitation")
        print("Starting PPO refinement", flush=True)
        model.learn(total_timesteps=args.steps, callback=SaveBest(args.out, VALIDATION_SEEDS, 40960))
        model.save(args.out / "final")
        (args.out / "runtime.json").write_text(
            json.dumps({"steps": model.num_timesteps, "wall_seconds": perf_counter() - start}, indent=2),
            encoding="utf-8",
        )
    finally:
        workers.close()


if __name__ == "__main__":
    main()
