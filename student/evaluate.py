"""Evaluate a policy using the fixed course rules and common start seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO, SAC

from racing import RaceEnv

ALGORITHMS = {"ppo": PPO, "sac": SAC}
PUBLIC_SEEDS = tuple(range(20))


def evaluate(policy, seeds, *, replay=None, camera="overview", layout="shanghai", physics_dt=.01, diagnostics=False):
    env = RaceEnv(reward_fn=lambda _: 0., render_mode="rgb_array" if replay else None,
                  camera=camera, layout=layout, physics_dt=physics_dt, diagnostics=diagnostics)
    episodes = []
    frames = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            record = replay is not None and len(episodes) == 0
            done = False
            while not done:
                if record:
                    frames.append(Image.fromarray(env.render()).resize((720, 450)))
                obs, _, terminated, truncated, info = env.step(policy(obs))
                if info["status"] == "physics_error":
                    raise RuntimeError(f"MuJoCo physics failure on {layout}, seed {seed}")
                done = terminated or truncated
            if record:
                frames.append(Image.fromarray(env.render()).resize((720, 450)))
            episodes.append({"seed": seed, **info})
    finally:
        env.close()

    if replay:
        replay.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(replay, save_all=True, append_images=frames[1:], duration=50, loop=0)

    times = [ep["time_s"] for ep in episodes if ep["status"] == "finished"]
    return {
        "episodes": episodes,
        "finish_rate": len(times) / len(episodes),
        "mean_progress": float(np.mean([ep["progress"] for ep in episodes])),
        "median_lap_s": float(np.median(times)) if times else None,
        "off_track_rate": sum(ep["status"] == "off_track" for ep in episodes) / len(episodes),
        "timeout_rate": sum(ep["status"] == "timeout" for ep in episodes) / len(episodes),
        "overspeed_rate": sum(ep["status"] == "overspeed" for ep in episodes) / len(episodes),
        "rollover_rate": sum(ep["status"] == "rollover" for ep in episodes) / len(episodes),
        "invalid_action_rate": sum(ep["status"] == "invalid_action" for ep in episodes) / len(episodes),
    }


def ranking(metrics):
    lap = metrics["median_lap_s"]
    return metrics["finish_rate"], metrics["mean_progress"], -lap if lap is not None else -float("inf")


def load_policy(model, algo):
    checkpoint = ALGORITHMS[algo].load(model, device="cpu")
    expected = checkpoint.observation_space.shape

    def predict(obs):
        if obs.shape != expected:
            raise ValueError(f"Checkpoint expects {expected}, current vehicle provides {obs.shape}. Use a matching course version.")
        return checkpoint.predict(obs, deterministic=True)[0]

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algo", choices=ALGORITHMS, default="ppo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path, help="Save the first episode as a GIF")
    parser.add_argument("--camera", choices=("overview", "chase"), default="overview")
    parser.add_argument("--layout", choices=("shanghai", "spa"), default="shanghai")
    args = parser.parse_args()
    result = evaluate(load_policy(args.model, args.algo), PUBLIC_SEEDS,
                      replay=args.replay, camera=args.camera, layout=args.layout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
