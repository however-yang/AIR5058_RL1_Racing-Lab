"""Select a reference using validation only, then score the winner on holdout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from reference import HOLDOUT_SEEDS, load_policy
from check_env import adaptive_pursuit, pure_pursuit
from evaluate import evaluate, ranking

SELECTION_SEEDS = tuple(range(200, 280))


def summarize(result):
    return {key: value for key, value in result.items() if key != "episodes"}


def score_run(folder):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    runtime = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
    candidates = []
    for filename in ("best.zip", "final.zip"):
        if not (folder / filename).exists():
            continue
        policy = load_policy(folder / filename, config["algo"])
        metrics = summarize(evaluate(policy, SELECTION_SEEDS))
        candidates.append({"run": str(folder), "checkpoint": filename, "config": config,
                           "runtime": runtime, "best_validation": metrics})
        print(folder.name, filename, metrics, flush=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--camera", choices=("overview", "chase"), default="overview")
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=min(4, len(args.runs))) as workers:
        candidates = [candidate for batch in workers.map(score_run, args.runs) for candidate in batch]
    winner = max(candidates, key=lambda candidate: ranking(candidate["best_validation"]))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "reference_policy.zip"
    shutil.copyfile(Path(winner["run"]) / winner["checkpoint"], checkpoint)
    policy = load_policy(checkpoint, winner["config"]["algo"])
    replay = args.out / "reference_lap.gif" if args.replay else None
    held_out = evaluate(policy, HOLDOUT_SEEDS, replay=replay, camera=args.camera)
    baselines = {}
    for speed in (3.5, 4.5, 5.0):
        baselines[str(speed)] = evaluate(lambda obs: pure_pursuit(obs, speed), HOLDOUT_SEEDS)
    baselines["adaptive"] = evaluate(adaptive_pursuit, HOLDOUT_SEEDS)
    report = {"selection_seeds": list(SELECTION_SEEDS), "candidates": candidates, "winner": winner, "holdout": held_out,
              "scripted_holdout": baselines}
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner["run"], "holdout": summarize(held_out),
                      "scripted_holdout": {k: summarize(v) for k, v in baselines.items()}}, indent=2))


if __name__ == "__main__":
    main()
