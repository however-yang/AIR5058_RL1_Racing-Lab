"""Match the multi-line and racecraft rewards, with a frozen faster-line benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.evaluation import (SPLIT_SECTORS as SECTORS, admitted, evaluate,
                                    freeze_policy_assets, load_json, model_fingerprint,
                                    prepare_spa_reference, rank, save_json, sha, timing, verify_frozen_assets)
from planned_racer import summary


DRY_SEEDS = tuple(range(9300, 9340))
LOW_GRIP_SEEDS = tuple(range(9400, 9440))
SPA_SEEDS = tuple(range(9700, 9740))
LOW_GRIP = .82


def screening(row):
    row["dry"] = evaluate(row, "shanghai", range(6200, 6208))
    row["stress"] = evaluate(row, "shanghai", range(6250, 6256), LOW_GRIP)
    return row


def shanghai(row):
    row["dry"] = evaluate(row, "shanghai", DRY_SEEDS)
    row["stress"] = evaluate(row, "shanghai", LOW_GRIP_SEEDS, LOW_GRIP)
    print(row["label"], summary(row["dry"]), summary(row["stress"]), "admitted", admitted(row), flush=True)
    return row


def spa(row):
    return evaluate(row, "spa", SPA_SEEDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--racecraft", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--skip-fixed-benchmark", action="store_true",
                        help="Compare the two newly trained rewards without the archived fixed policy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    physics = model_fingerprint()
    prior = load_json(args.previous / "results.json")
    if physics != prior["frozen"]["physics"]:
        raise RuntimeError("The vehicle differs from the previous comparison")
    runs = {"baseline": args.original, "racecraft": args.racecraft}
    configs = {name: load_json(folder / "config.json") for name, folder in runs.items()}
    runtimes = {name: load_json(folder / "runtime.json") for name, folder in runs.items()}
    old, new = configs["baseline"], configs["racecraft"]
    settings = ("layout", "algorithm", "gamma", "seed", "steps", "residual_scale",
                "failure_seconds", "reward_scale", "potential_weight")
    expected = prior["frozen"]["sha256"]["fixed"]
    if (not old["multi_line"] or old.get("racecraft") or not new["racecraft"]
            or any(old[k] != new[k] for k in settings)
            or any(run["steps"] != old["steps"] for run in runtimes.values())
            or any(sha(Path(config["line"])) != expected["line"] for config in configs.values())):
        raise ValueError("Expected matching Shanghai runs with only the reward mode changed")

    scale = prior["groups"]["fixed"]["residual_scale"]
    common = {"mode": "sac", "residual_scale": scale, "lane_limit": 0., "lane_gain": 1.}
    shortlist = []
    if not args.skip_fixed_benchmark:
        benchmark = {**common, "arm": "fixed", "label": "previous_fixed_fine_tuned",
                     "display_name": "Previous fixed-line fine-tuned SAC",
                     "shanghai_line": str(args.previous / "fixed" / "shanghai_line.npz"),
                     "checkpoint": str(args.previous / "fixed" / "policy.zip"),
                     "steps": prior["groups"]["fixed"]["steps"]}
        if (sha(Path(benchmark["checkpoint"])) != expected["checkpoint"]
                or sha(Path(benchmark["shanghai_line"])) != expected["line"]):
            raise RuntimeError("Previous benchmark weights or line have changed")
        shortlist.append(benchmark)
    candidates = []
    for name, folder in runs.items():
        for row in load_json(folder / "validation.json"):
            candidates.append({
                **common, "arm": name, "label": name + "/" + row["name"], "steps": row["steps"],
                "display_name": "Original multi-line reward" if name == "baseline" else "Racecraft reward",
                "shanghai_line": configs[name]["line"], "checkpoint": str(folder / (row["name"] + ".zip")),
            })
    with ProcessPoolExecutor(max_workers=8) as pool:
        screened = list(pool.map(screening, candidates))
        for name in runs:
            rows = sorted((row for row in screened if row["arm"] == name), key=rank, reverse=True)[:3]
            print("Shanghai shortlist:", name, [row["label"] for row in rows], flush=True)
            shortlist.extend(dict(row) for row in rows)
        evaluated = list(pool.map(shanghai, shortlist))
    arm_order = (*runs,) if args.skip_fixed_benchmark else (*runs, "fixed")
    groups = {name: max((row for row in evaluated if row["arm"] == name), key=rank)
              for name in arm_order}
    eligible = {name: row for name, row in groups.items() if admitted(row)}
    fingerprints = freeze_policy_assets(args.out, groups)

    frozen = {"winner": max(eligible, key=lambda name: rank(eligible[name])) if eligible else None,
              "physics": physics, "sectors_m": SECTORS, "dry_seeds": DRY_SEEDS,
              "stress_seeds": LOW_GRIP_SEEDS, "stress_mu": LOW_GRIP, "threshold": .95,
              "admitted": {name: name in eligible for name in groups}, "sha256": fingerprints,
              "selected": {name: {"label": row["label"], "steps": row.get("steps"), "residual_scale": scale}
                           for name, row in groups.items()}}
    save_json(args.out / "frozen_selection.json", frozen)
    print("Frozen before Spa:", frozen["winner"], frozen["admitted"], flush=True)

    spa_results, spa_hashes = {}, {}
    if eligible:
        expected_hash = prior["spa_line_sha256"]["fixed"]
        target, spa_hashes = prepare_spa_reference(args.out, args.previous / "spa_line.npz",
                                                    expected_hash, eligible)
        with ProcessPoolExecutor(max_workers=len(eligible)) as pool:
            spa_results = dict(zip(eligible, pool.map(spa, eligible.values())))
        verify_frozen_assets(target, expected_hash, eligible, fingerprints)

    reliable = {name: row for name, row in spa_results.items() if row["finish_rate"] >= .95}
    fastest = min(reliable, key=lambda name: reliable[name]["median_lap_s"]) if reliable else None
    shanghai_timing = {
        condition: {name: timing(row[condition], "shanghai", groups["baseline"][condition], SECTORS["shanghai"])
                    for name, row in groups.items()}
        for condition in ("dry", "stress")
    }
    spa_timing = {name: timing(row, "spa", spa_results.get("baseline"), SECTORS["spa"])
                  for name, row in spa_results.items()}
    versus_fixed = {}
    if "fixed" in groups:
        versus_fixed = {condition: timing(groups["racecraft"][condition], "shanghai",
                                          groups["fixed"][condition], SECTORS["shanghai"])
                        for condition in ("dry", "stress")}
    if "racecraft" in spa_results and "fixed" in spa_results:
        versus_fixed["spa"] = timing(spa_results["racecraft"], "spa", spa_results["fixed"], SECTORS["spa"])
    save_json(args.out / "results.json", {
        "frozen": frozen, "groups": groups, "screened": screened, "all_shanghai_candidates": evaluated,
        "sac_training": runtimes, "sac_config": configs, "spa_seeds": SPA_SEEDS,
        "spa": spa_results, "spa_line_sha256": spa_hashes, "fastest_observed_spa": fastest,
        "shanghai_timing": shanghai_timing, "spa_timing": spa_timing, "racecraft_paired_with_fixed": versus_fixed,
    })
    print("Spa:", {name: summary(row) for name, row in spa_results.items()}, flush=True)


if __name__ == "__main__":
    main()
