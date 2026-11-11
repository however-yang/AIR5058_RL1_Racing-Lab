"""Select the fastest reliable Shanghai driver and independently test Spa."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import torch

from planned_racer import run_cases, summary
from racing import RaceEnv
from time_line import TimeLine


DRY_SEEDS = tuple(range(3100, 3140))
STRESS_SEEDS = tuple(range(3200, 3220))
SPA_SEEDS = tuple(range(3500, 3540))
STRESS_MU = .82


def lap_key(row):
    lap = row["median_lap_s"]
    return row["finish_rate"], row["mean_progress"], -(lap if lap is not None else float("inf"))


def score(candidate):
    dry, stress = candidate["dry"], candidate["stress"]
    median = dry["median_lap_s"]
    pace = -(median if median is not None else float("inf"))
    if min(dry["finish_rate"], stress["finish_rate"]) >= .95:
        return 1, pace, dry["finish_rate"], stress["finish_rate"]
    return 0, min(dry["finish_rate"], stress["finish_rate"]), dry["finish_rate"], pace


def assess(candidate):
    torch.set_num_threads(1)
    line = TimeLine.load(candidate["line"]) if candidate.get("line") else None
    mode = candidate["mode"]
    checkpoint = candidate.get("checkpoint")
    candidate["dry"] = run_cases(mode, DRY_SEEDS, line=line, checkpoint=checkpoint)
    candidate["stress"] = run_cases(mode, STRESS_SEEDS, line=line, checkpoint=checkpoint, mu=STRESS_MU)
    print(candidate["label"], summary(candidate["dry"]), summary(candidate["stress"]), flush=True)
    return candidate


def spa_case(job):
    torch.set_num_threads(1)
    mode, line, checkpoint = job
    return run_cases(mode, SPA_SEEDS, layout="spa", line=line, checkpoint=checkpoint)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--planners", type=Path, nargs="+", required=True)
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    candidates = [{"label": "previous PPO", "mode": "ppo", "checkpoint": str(args.ppo)}]
    for folder in args.planners:
        line = folder / "shanghai_line.npz"
        candidates.append({"label": folder.name, "mode": "planner", "line": str(line),
                           "lateral_limit": TimeLine.load(line).lateral_limit})
    for folder in args.runs:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        runtime = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
        history = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
        selected = {}
        for row in sorted(history, key=lap_key, reverse=True):
            model = folder / f"checkpoint_{row['steps']}.zip"
            if model.is_file():
                selected.setdefault(str(model), row["steps"])
            if len(selected) == 2:
                break
        selected[str(folder / "final.zip")] = runtime["steps"]
        for checkpoint, step in selected.items():
            candidates.append({"label": f"{folder.name}/{Path(checkpoint).stem}", "mode": config["algorithm"].lower(),
                               "line": str(folder / "shanghai_line.npz"), "checkpoint": checkpoint,
                               "step": step, "lateral_limit": config["lateral_limit"]})

    with ProcessPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        assessed = list(pool.map(assess, candidates))
    winner = max(assessed, key=score)
    baseline = assessed[0]
    plan_baseline = max((row for row in assessed if row["mode"] == "planner"), key=score)
    actor = args.out / "reference_policy.zip"
    if winner.get("checkpoint"):
        shutil.copyfile(winner["checkpoint"], actor)
    if winner.get("line"):
        shutil.copyfile(winner["line"], args.out / "shanghai_line.npz")
    manifest = {"label": winner["label"], "mode": winner["mode"], "lateral_limit": winner.get("lateral_limit"),
                "checkpoint_sha256": sha(actor) if actor.exists() else None,
                "shanghai_line_sha256": sha(args.out / "shanghai_line.npz") if winner.get("line") else None,
                "selection_rule": "Shanghai dry and low-grip finish >=95%, then fastest dry median",
                "dry_seeds": list(DRY_SEEDS), "stress_seeds": list(STRESS_SEEDS), "stress_mu": STRESS_MU}
    (args.out / "frozen_selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Frozen Shanghai winner:", winner["label"], flush=True)

    # The Spa geometry is first instantiated after the candidate and parameters are frozen.
    torch.set_num_threads(1)
    comparison = {"winner": winner, "previous": baseline, "planner": plan_baseline}
    for mode in ("sac", "td3"):
        choices = [row for row in assessed if row["mode"] == mode]
        if choices:
            comparison[mode] = max(choices, key=score)
    spa = RaceEnv(layout="spa")
    try:
        levels = {row["lateral_limit"] for row in comparison.values() if row.get("line")}
        plans = {level: TimeLine(spa.track, level) for level in sorted(levels)}
    finally:
        spa.close()
    if winner.get("line"):
        plans[winner["lateral_limit"]].save(args.out / "spa_line.npz")

    jobs, names = {}, {}
    for name, row in comparison.items():
        key = row["label"]
        names[name] = key
        model = actor if key == winner["label"] and actor.exists() else row.get("checkpoint")
        jobs[key] = (row["mode"], plans.get(row.get("lateral_limit")), model)
    with ProcessPoolExecutor(max_workers=min(5, len(jobs))) as pool:
        evaluated = dict(zip(jobs, pool.map(spa_case, jobs.values())))
    results = {name: evaluated[key] for name, key in names.items()}
    if actor.exists():
        assert sha(actor) == manifest["checkpoint_sha256"]
    report = {"frozen": manifest, "winner": winner, "candidates": assessed,
              "spa_transfer": results["winner"], "spa_comparison": results,
              "comparison_labels": names}
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Spa comparison:", {k: summary(v) for k, v in results.items()}, flush=True)



if __name__ == "__main__":
    main()
