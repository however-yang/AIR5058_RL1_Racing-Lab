"""Choose by Shanghai pace and robustness, then test the frozen model on Spa."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from reference import evaluate, load_policy
from racing import RaceEnv


DRY_SEEDS = tuple(range(2200, 2240))
STRESS_SEEDS = tuple(range(2300, 2320))
SPA_SEEDS = tuple(range(2500, 2540))
STRESS_GRIP = .82


def select_score(row):
    dry, stress = row["shanghai_dry"], row["shanghai_lower_grip"]
    pace = -(dry["median_lap_s"] if dry["median_lap_s"] is not None else float("inf"))
    if dry["finish_rate"] >= .95:
        return 1, stress["finish_rate"], dry["finish_rate"], pace
    return 0, dry["finish_rate"], dry["mean_progress"], stress["finish_rate"]


def lower_grip(policy):
    env = RaceEnv(layout="shanghai")
    env.model.pair_friction[:, :2] = STRESS_GRIP
    episodes = []
    try:
        for seed in STRESS_SEEDS:
            obs, info = env.reset(seed=seed)
            while info["status"] == "running":
                obs, _, _, _, info = env.step(policy(obs))
            if info["status"] == "physics_error":
                raise RuntimeError(f"MuJoCo stress evaluation failed on seed {seed}")
            episodes.append({"seed": seed, **info})
    finally:
        env.close()
    times = [ep["time_s"] for ep in episodes if ep["status"] == "finished"]
    return {"finish_rate": len(times) / len(episodes),
            "median_lap_s": float(np.median(times)) if times else None, "episodes": episodes}


def assess(candidate):
    torch.set_num_threads(1)
    policy = load_policy(Path(candidate["model"]), "ppo")
    candidate["shanghai_dry"] = evaluate(policy, DRY_SEEDS, layout="shanghai")
    candidate["shanghai_lower_grip"] = lower_grip(policy)
    print(candidate["label"], candidate["shanghai_dry"]["finish_rate"],
          candidate["shanghai_dry"]["median_lap_s"],
          candidate["shanghai_lower_grip"]["finish_rate"], flush=True)
    return candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    candidates = [{"label": "previous", "model": str(args.previous)}]
    for directory in args.runs:
        cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        run = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        history = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        best = max(history, key=lambda m: (m["finish_rate"], m["mean_progress"],
                                           -(m["median_lap_s"] or float("inf"))))
        checkpoints = [("best", best["steps"]), ("final", run["steps"])]
        saved = directory / "early_best.json"
        if saved.is_file():
            checkpoints.append(("early_best", json.loads(saved.read_text(encoding="utf-8"))["steps"]))
        for name, steps in checkpoints:
            candidates.append({"label": f"{directory.name}/{name}", "model": str(directory / f"{name}.zip"),
                               "training_steps_at_checkpoint": steps, "run": run, "config": cfg})

    with ProcessPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        scored = list(pool.map(assess, candidates))
    winner = max(scored, key=select_score)
    checkpoint = args.out / "reference_policy.zip"
    shutil.copyfile(winner["model"], checkpoint)
    frozen = {"winner": winner["label"], "checkpoint_sha256": digest(checkpoint),
              "selection": "Shanghai dry >=95%; then lower-grip completion, dry completion and pace",
              "dry_seeds": list(DRY_SEEDS), "stress_seeds": list(STRESS_SEEDS), "stress_mu": STRESS_GRIP}
    (args.out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Frozen selection:", frozen, flush=True)

    torch.set_num_threads(1)
    spa = evaluate(load_policy(checkpoint, "ppo"), SPA_SEEDS, layout="spa")
    old_spa = evaluate(load_policy(args.previous, "ppo"), SPA_SEEDS, layout="spa")
    assert digest(checkpoint) == frozen["checkpoint_sha256"]
    report = {**frozen, "candidates": scored, "winner_details": winner,
              "spa_transfer": spa, "previous_spa_transfer": old_spa}
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Spa winner and previous:", [(r["finish_rate"], r["median_lap_s"]) for r in (spa, old_spa)])


if __name__ == "__main__":
    main()
