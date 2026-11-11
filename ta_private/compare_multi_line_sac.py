"""Compare the frozen time-reward SAC with a matched multi-line reward run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.evaluation import (SPLIT_SECTORS as SECTORS, admitted, evaluate as evaluate_laps,
                                    freeze_policy_assets, load_json, model_fingerprint,
                                    prepare_spa_reference, rank, save_json, sha, timing, verify_frozen_assets)
from planned_racer import summary


DRY_SEEDS = tuple(range(7300, 7340))
STRESS_SEEDS = tuple(range(7400, 7440))
SPA_SEEDS = tuple(range(7700, 7740))
STRESS_MU = .82


def evaluate(row, seeds, layout, mu=None):
    return evaluate_laps({**row, "mode": "sac"}, layout, seeds, mu)


def shanghai(row):
    row["dry"] = evaluate(row, DRY_SEEDS, "shanghai")
    row["stress"] = evaluate(row, STRESS_SEEDS, "shanghai", STRESS_MU)
    print(row["label"], summary(row["dry"]), summary(row["stress"]), "admitted", admitted(row), flush=True)
    return row


def spa(row):
    return evaluate(row, SPA_SEEDS, "spa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--sac", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    physics = model_fingerprint()
    args.out.mkdir(parents=True, exist_ok=True)
    previous = load_json(args.previous / "results.json")
    old = previous["groups"]["time_sac"]
    shanghai_line = args.previous / "time_sac" / "shanghai_line.npz"
    baseline = {
        "label": old["label"], "display_name": "Previous SAC", "mode": "sac",
        "shanghai_line": str(shanghai_line), "checkpoint": str(args.previous / "time_sac" / "policy.zip"),
        "steps": old["steps"], "residual_scale": old["residual_scale"],
    }
    config = load_json(args.sac / "config.json")
    history = load_json(args.sac / "validation.json")
    runtime = load_json(args.sac / "runtime.json")
    if not config["multi_line"] or runtime["steps"] != previous["sac_training"]["steps"]:
        raise ValueError("Expected the new reward and the same training budget as the previous SAC")
    settings = ("layout", "algorithm", "gamma", "seed", "steps", "residual_scale",
                "failure_seconds", "reward_scale", "potential_weight")
    if any(config[k] != previous["sac_config"][k] for k in settings):
        raise ValueError("SAC training or reward settings differ from the previous run")
    old_hash = previous["frozen"]["sha256"]["time_sac"]
    if sha(shanghai_line) != old_hash["line"] or sha(Path(config["line"])) != old_hash["line"]:
        raise RuntimeError("The two runs must use the same Shanghai reference line")
    if sha(Path(baseline["checkpoint"])) != old_hash["checkpoint"] or physics != previous["frozen"]["physics"]:
        raise RuntimeError("The baseline weights or vehicle files differ from the previous comparison")

    # Match the previous evaluation gain. The training-end weights include the last gradient update.
    candidates = [baseline] + [
        {**baseline, "label": "multi_line/" + row["name"], "display_name": "Multi-line SAC",
         "checkpoint": str(args.sac / (row["name"] + ".zip")), "steps": row["steps"]}
        for row in history
    ]
    with ProcessPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        evaluated = list(pool.map(shanghai, candidates))
    groups = {"baseline": evaluated[0], "multi_line": max(evaluated[1:], key=rank)}
    qualified = {name: row for name, row in groups.items() if admitted(row)}
    fingerprints = freeze_policy_assets(args.out, groups)
    frozen = {
        "winner": max(qualified, key=lambda name: rank(qualified[name])) if qualified else None,
        "physics": physics, "sectors_m": SECTORS, "dry_seeds": DRY_SEEDS,
        "stress_seeds": STRESS_SEEDS, "stress_mu": STRESS_MU, "threshold": .95,
        "admitted": {name: name in qualified for name in groups},
        "selected": {name: {"label": row["label"], "residual_scale": row["residual_scale"]}
                     for name, row in groups.items()},
        "sha256": fingerprints,
    }
    save_json(args.out / "frozen_selection.json", frozen)
    print("Frozen before Spa:", frozen["winner"], frozen["admitted"], flush=True)

    spa_results = {}
    spa_hashes = {}
    if qualified:
        expected_hash = previous["spa_line_sha256"]["time_sac"]
        target, spa_hashes = prepare_spa_reference(args.out, args.previous / "measured_spa_line.npz",
                                                    expected_hash, qualified)
        with ProcessPoolExecutor(max_workers=len(qualified)) as pool:
            spa_results = dict(zip(qualified, pool.map(spa, qualified.values())))
        verify_frozen_assets(target, expected_hash, qualified, fingerprints)

    reliable_spa = {name: row for name, row in spa_results.items() if row["finish_rate"] >= .95}
    fastest = min(reliable_spa, key=lambda name: reliable_spa[name]["median_lap_s"]) if reliable_spa else None
    shanghai_timing = {name: timing(row["dry"], "shanghai", groups["baseline"]["dry"], SECTORS["shanghai"])
                       for name, row in groups.items()}
    spa_timing = {name: timing(row, "spa", spa_results.get("baseline"), SECTORS["spa"])
                  for name, row in spa_results.items()}
    save_json(args.out / "results.json", {
        "frozen": frozen, "groups": groups, "all_shanghai_candidates": evaluated,
        "sac_training": runtime, "sac_config": config, "spa_seeds": SPA_SEEDS,
        "spa": spa_results, "spa_line_sha256": spa_hashes, "fastest_observed_spa": fastest,
        "shanghai_timing": shanghai_timing, "spa_timing": spa_timing,
    })
    print("Shanghai:", shanghai_timing, flush=True)
    print("Spa:", spa_timing, flush=True)


if __name__ == "__main__":
    main()
