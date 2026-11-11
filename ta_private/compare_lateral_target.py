"""Evaluate the frozen policy and both fine-tuning arms before Spa transfer."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.evaluation import (SPLIT_SECTORS as SECTORS, admitted, evaluate,
                                    freeze_policy_assets, load_json, model_fingerprint,
                                    prepare_spa_reference, rank, save_json, sha, timing, verify_frozen_assets)
from planned_racer import summary


DRY_SEEDS = tuple(range(8300, 8340))
LOW_GRIP_SEEDS = tuple(range(8400, 8440))
SPA_SEEDS = tuple(range(8700, 8740))
LOW_GRIP = .82


def screen_lateral(row):
    row["dry"] = evaluate(row, "shanghai", range(8100, 8108))
    row["stress"] = evaluate(row, "shanghai", range(8150, 8156), LOW_GRIP)
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
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--lateral", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    physics = model_fingerprint()
    prior = load_json(args.previous / "results.json")
    expected = prior["frozen"]["sha256"]["multi_line"]
    line = args.previous / "multi_line" / "shanghai_line.npz"
    weight = args.previous / "multi_line" / "policy.zip"
    if (sha(line) != expected["line"] or sha(weight) != expected["checkpoint"]
            or physics != prior["frozen"]["physics"]):
        raise RuntimeError("Previous policy or vehicle differs from the frozen run")
    common = {"mode": "sac", "shanghai_line": str(line),
              "residual_scale": prior["groups"]["multi_line"]["residual_scale"]}
    baseline = {**common, "arm": "baseline", "label": "frozen_multi_line",
                "display_name": "Frozen multi-line SAC", "checkpoint": str(weight),
                "lane_limit": 0., "lane_gain": 1., "steps": 0}
    arms = {"fixed": args.fixed, "lateral": args.lateral}
    configs = {name: load_json(folder / "config.json") for name, folder in arms.items()}
    runs = {name: load_json(folder / "runtime.json") for name, folder in arms.items()}
    fixed, lateral = configs["fixed"], configs["lateral"]
    if (fixed["lane_limit"] != 0. or lateral["lane_limit"] <= 0.
            or {**fixed, "lane_limit": lateral["lane_limit"]} != lateral
            or any(r["steps"] != fixed["steps"] for r in runs.values())
            or fixed["source"] != expected or fixed["residual_scale"] != common["residual_scale"]):
        raise ValueError("Expected matched fine-tuning runs from the frozen multi-line policy")

    candidates = [baseline]
    lateral_screen = []
    for name, folder in arms.items():
        history = load_json(folder / "validation.json")
        rows = [{**common, "arm": name, "label": name + "/" + row["name"],
                 "display_name": "Learned lateral target" if name == "lateral" else "Fixed line fine-tuned",
                 "checkpoint": str(folder / (row["name"] + ".zip")),
                 "lane_limit": configs[name]["lane_limit"], "lane_gain": 1., "steps": row["steps"],
                 "dry": row["dry"], "stress": row["stress"]} for row in history]
        if name == "fixed":
            candidates.extend(sorted(rows, key=rank, reverse=True)[:3])
            continue
        # Calibrate the deployment gain before any of the new selection starts are evaluated.
        reduced = [{**row, "label": f'{row["label"]}/gain_{gain:.2f}', "lane_gain": gain}
                   for gain in (.45, .20) for row in rows]
        with ProcessPoolExecutor(max_workers=8) as pool:
            lateral_screen = rows + list(pool.map(screen_lateral, reduced))
        for gain in (1., .45, .20):
            best = max((row for row in lateral_screen if row["lane_gain"] == gain), key=rank)
            print("Screening lateral gain", gain, best["label"], summary(best["dry"]),
                  summary(best["stress"]), flush=True)
            candidates.append(dict(best))
    with ProcessPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        evaluated = list(pool.map(shanghai, candidates))
    groups = {"baseline": evaluated[0]}
    for name in arms:
        groups[name] = max((row for row in evaluated if row["arm"] == name), key=rank)
    qualified = {name: row for name, row in groups.items() if admitted(row)}
    fingerprints = freeze_policy_assets(args.out, groups)

    frozen = {"winner": max(qualified, key=lambda name: rank(qualified[name])) if qualified else None,
              "physics": physics, "sectors_m": SECTORS, "dry_seeds": DRY_SEEDS,
              "stress_seeds": LOW_GRIP_SEEDS, "stress_mu": LOW_GRIP, "threshold": .95,
              "admitted": {name: name in qualified for name in groups}, "sha256": fingerprints,
              "selected": {name: {key: row[key] for key in ("label", "residual_scale", "lane_limit", "lane_gain", "steps")}
                           for name, row in groups.items()}}
    save_json(args.out / "frozen_selection.json", frozen)
    print("Frozen before Spa:", frozen["winner"], frozen["admitted"], flush=True)

    spa_results = {}
    spa_hashes = {}
    if qualified:
        expected_hash = prior["spa_line_sha256"]["multi_line"]
        target, spa_hashes = prepare_spa_reference(args.out, args.previous / "spa_line.npz",
                                                    expected_hash, qualified)
        with ProcessPoolExecutor(max_workers=len(qualified)) as pool:
            spa_results = dict(zip(qualified, pool.map(spa, qualified.values())))
        verify_frozen_assets(target, expected_hash, qualified, fingerprints)

    reliable = {name: row for name, row in spa_results.items() if row["finish_rate"] >= .95}
    fastest = min(reliable, key=lambda name: reliable[name]["median_lap_s"]) if reliable else None
    shanghai_timing = {
        condition: {name: timing(row[condition], "shanghai", groups["baseline"][condition], SECTORS["shanghai"])
                    for name, row in groups.items()}
        for condition in ("dry", "stress")
    }
    spa_timing = {name: timing(row, "spa", spa_results.get("baseline"), SECTORS["spa"])
                  for name, row in spa_results.items()}
    paired_with_fixed = {
        condition: timing(groups["lateral"][condition], "shanghai", groups["fixed"][condition], SECTORS["shanghai"])
        for condition in ("dry", "stress")
    }
    if "lateral" in spa_results and "fixed" in spa_results:
        paired_with_fixed["spa"] = timing(spa_results["lateral"], "spa", spa_results["fixed"], SECTORS["spa"])
    save_json(args.out / "results.json", {
        "frozen": frozen, "groups": groups, "all_shanghai_candidates": evaluated,
        "sac_training": runs, "sac_config": configs, "spa_seeds": SPA_SEEDS, "lateral_screen": lateral_screen,
        "spa": spa_results, "spa_line_sha256": spa_hashes, "fastest_observed_spa": fastest,
        "shanghai_timing": shanghai_timing, "spa_timing": spa_timing,
        "lateral_paired_with_fixed": paired_with_fixed,
    })
    print("Spa:", {name: summary(row) for name, row in spa_results.items()}, flush=True)


if __name__ == "__main__":
    main()
