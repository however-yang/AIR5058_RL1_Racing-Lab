"""Select only on Shanghai, then evaluate the frozen winner on Spa."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from reference import HOLDOUT_SEEDS, evaluate, load_policy
from evaluate import ranking


def summary(result):
    return {key: value for key, value in result.items() if key != "episodes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for folder in args.runs:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        runtime = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
        history = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
        checkpoint = max(history, key=ranking)
        candidates.append({"run": str(folder), "config": config, "runtime": runtime,
                           "shanghai_validation": summary(checkpoint),
                           "shared_budget": summary(next(row for row in history if row["steps"] == 81920))})

    winner = max(candidates, key=lambda row: ranking(row["shanghai_validation"]))
    args.out.mkdir(parents=True)
    model = args.out / "reference_policy.zip"
    shutil.copyfile(Path(winner["run"]) / "best.zip", model)
    policy = load_policy(model, winner["config"]["algo"])
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    print("Frozen Shanghai selection:", winner["run"], winner["shanghai_validation"], flush=True)
    shanghai = evaluate(policy, HOLDOUT_SEEDS, layout="shanghai")
    spa = evaluate(policy, HOLDOUT_SEEDS, layout="spa")
    assert hashlib.sha256(model.read_bytes()).hexdigest() == checksum
    report = {"physics_version": "grip-ackermann-v1", "training_layout": "shanghai",
              "transfer_layout": "spa", "checkpoint_sha256": checksum,
              "winner": winner, "candidates": candidates, "shanghai_holdout": shanghai, "spa_transfer": spa}
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"shanghai_holdout": summary(shanghai), "spa_transfer": summary(spa)}, indent=2))


if __name__ == "__main__":
    main()
