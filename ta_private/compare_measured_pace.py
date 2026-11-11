"""Three-way comparison with fixed timing lines and a hard Shanghai transfer gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import torch

from experiments.evaluation import SECTORS, admitted, artifact_path, model_fingerprint, rank, sha, timing
from planned_racer import run_cases, summary
from racing import RaceEnv
from time_line import TimeLine


DRY_SEEDS = tuple(range(6300, 6340))
STRESS_SEEDS = tuple(range(6400, 6420))
SPA_SEEDS = tuple(range(6700, 6740))
STRESS_MU = .82
def run(row, seeds, layout, mu=None):
    source = row["shanghai_line"] if layout == "shanghai" else row["spa_line"]
    return run_cases(row["mode"], seeds, layout=layout, line=TimeLine.load(artifact_path(source)),
                     checkpoint=artifact_path(row["checkpoint"]) if row.get("checkpoint") else None, mu=mu, sectors=SECTORS[layout],
                     **({"residual_scale": row["residual_scale"]} if "residual_scale" in row else {}))


def shanghai(row):
    torch.set_num_threads(1)
    row["dry"] = run(row, DRY_SEEDS, "shanghai")
    row["stress"] = run(row, STRESS_SEEDS, "shanghai", STRESS_MU)
    print(row["label"], summary(row["dry"]), summary(row["stress"]), "admitted", admitted(row), flush=True)
    return row


def spa(row):
    torch.set_num_threads(1)
    return run(row, SPA_SEEDS, "spa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--sac", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    physics = model_fingerprint()
    args.out.mkdir(parents=True)
    cfg = json.loads((args.sac / "config.json").read_text(encoding="utf-8"))
    history = json.loads((args.sac / "validation.json").read_text(encoding="utf-8"))
    training = json.loads((args.sac / "runtime.json").read_text(encoding="utf-8"))
    candidates = [
        {"label": "baseline", "mode": "planner", "shanghai_line": str(args.baseline / "shanghai_line.npz")},
        {"label": "measured", "mode": "planner", "shanghai_line": str(args.calibrated / "shanghai_line.npz")},
    ]
    for row in history:
        candidates.append({"label": "time_sac/" + row["name"], "mode": "sac",
                           "shanghai_line": cfg["line"], "checkpoint": str(args.sac / (row["name"] + ".zip")),
                           "residual_scale": cfg["residual_scale"], "steps": row["steps"]})
    # One conservative deployment setting, chosen before evaluating any of the fresh starts.
    screen = sorted(history, key=lambda row: (min(row["dry"]["finish_rate"], row["stress"]["finish_rate"]),
                                             -(row["dry"]["median_lap_s"] or float("inf"))),
                    reverse=True)
    for item in screen[:2]:
        original = next(row for row in candidates if row["label"] == "time_sac/" + item["name"])
        candidates.append({**original, "label": original["label"] + "/45pct",
                           "residual_scale": [.45 * s for s in original["residual_scale"]]})
    candidates.append({"label": "measured/same_parameters", "mode": "planner",
                       "shanghai_line": str(args.calibrated / "lat7.10_preview0.52.npz")})
    with ProcessPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        evaluated = list(pool.map(shanghai, candidates))
    groups = {"baseline": evaluated[0], "measured": evaluated[1],
              "time_sac": max((row for row in evaluated if row["mode"] == "sac"), key=rank)}
    eligible = {name: row for name, row in groups.items() if admitted(row)}
    if not eligible:
        raise RuntimeError("No Shanghai candidate passed both 95% transfer gates")
    winner = max(eligible, key=lambda name: rank(eligible[name]))
    for name, row in eligible.items():
        bundle = args.out / name
        bundle.mkdir()
        for field, filename in (("shanghai_line", "shanghai_line.npz"), ("checkpoint", "policy.zip")):
            if row.get(field):
                row["source_" + field] = row[field]
                target = bundle / filename
                shutil.copyfile(artifact_path(row[field]), target)
                row[field] = target.as_posix()
    frozen = {
        "winner": winner, "physics": physics, "sectors_m": SECTORS,
        "dry_seeds": list(DRY_SEEDS), "stress_seeds": list(STRESS_SEEDS), "stress_mu": STRESS_MU,
        "threshold": .95,
        "admitted": {name: name in eligible for name in groups},
        "selected": {name: {"label": row["label"], "residual_scale": row.get("residual_scale")}
                     for name, row in groups.items()},
        "sha256": {name: {"line": sha(Path(row["shanghai_line"])),
                          "checkpoint": sha(Path(row["checkpoint"])) if row.get("checkpoint") else None}
                   for name, row in groups.items()},
    }
    (args.out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Frozen before Spa:", winner, frozen["admitted"], flush=True)

    torch.set_num_threads(1)
    if "baseline" in eligible:
        target = args.out / "baseline" / "spa_line.npz"
        shutil.copyfile(args.baseline / "spa_line.npz", target)
        eligible["baseline"]["spa_line"] = str(target)
    if any(name != "baseline" for name in eligible):
        source = TimeLine.load(args.calibrated / "shanghai_line.npz")
        env = RaceEnv(layout="spa")
        try:
            line = TimeLine(env.track, source.lateral_limit, pace=source.pace,
                            brake_preview=source.brake_preview)
        finally:
            env.close()
        path = args.out / "measured_spa_line.npz"
        line.save(path)
        for name in ("measured", "time_sac"):
            if name in eligible:
                eligible[name]["spa_line"] = str(path)
    with ProcessPoolExecutor(max_workers=len(eligible)) as pool:
        spa_results = dict(zip(eligible, pool.map(spa, eligible.values())))

    dry_timing = {name: timing(row["dry"], "shanghai", groups["baseline"]["dry"])
                  for name, row in groups.items()}
    aligned = timing(evaluated[-1]["dry"], "shanghai", groups["baseline"]["dry"])
    spa_timing = {name: timing(result, "spa", spa_results.get("baseline"))
                  for name, result in spa_results.items()}
    reliable_spa = {name: result for name, result in spa_results.items() if result["finish_rate"] >= .95}
    fastest = min(reliable_spa, key=lambda name: reliable_spa[name]["median_lap_s"]) if reliable_spa else None
    for name, row in eligible.items():
        if sha(Path(row["shanghai_line"])) != frozen["sha256"][name]["line"]:
            raise RuntimeError("Shanghai line changed during the transfer test")
        if row.get("checkpoint") and sha(Path(row["checkpoint"])) != frozen["sha256"][name]["checkpoint"]:
            raise RuntimeError("Checkpoint changed during the transfer test")
    report = {"frozen": frozen, "spa_line_sha256": {name: sha(Path(row["spa_line"]))
                                                    for name, row in eligible.items()}, "groups": groups, "all_shanghai_candidates": evaluated,
              "sac_training": training, "sac_config": cfg, "spa_seeds": list(SPA_SEEDS),
              "spa": spa_results, "shanghai_timing": dry_timing, "spa_timing": spa_timing,
              "shanghai_same_parameter_ablation": aligned,
              "fastest_observed_spa": fastest}
    (args.out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Shanghai:", dry_timing, flush=True)
    print("Spa:", spa_timing, flush=True)


if __name__ == "__main__":
    main()
