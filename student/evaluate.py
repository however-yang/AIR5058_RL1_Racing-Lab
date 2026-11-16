"""Run the baseline and optionally compare a frozen student policy on the same seeds."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import mean, median

from lab import InvalidAction, Lab, SURFACES, baseline
from scoring import grade


def run_lap(lab, seed, act, *, capture=False, keep_profile=False):
    obs, _ = lab.reset(seed, capture_profile=capture)
    while True:
        obs.setflags(write=False)
        try:
            action = act(obs)
        except Exception:
            info = {"status": "policy_error"}
            break
        try:
            obs, _, term, trunc, info = lab.step(action)
        except InvalidAction:
            info = {"status": "invalid_action"}
            break
        if term or trunc:
            break
    result = {"seed": seed, **info}
    if capture:
        profile = lab.performance()
        result["performance"] = {k: v for k, v in profile.items() if not isinstance(v, list)}
        if keep_profile:
            result["profile"] = profile
    return result


def summarize(episodes):
    finished = [ep for ep in episodes if ep["status"] == "finished"]
    return {"trials": len(episodes), "finished": len(finished),
            "completion_rate": len(finished) / len(episodes),
            "median_lap_s": median(ep["time_s"] for ep in finished) if finished else None,
            "episodes": episodes}


def paired_times(reference, student):
    if min(reference["completion_rate"], student["completion_rate"]) < .95:
        return None
    pairs = [(a, b) for a, b in zip(reference["episodes"], student["episodes"])
             if a["status"] == b["status"] == "finished"]
    names = set.intersection(*(set(ep["sectors_s"]) for pair in pairs for ep in pair))
    return {"common_finishes": len(pairs),
            "lap_saved_s": median(a["time_s"] - b["time_s"] for a, b in pairs),
            "lap_saved_fraction": median((a["time_s"] - b["time_s"]) / a["time_s"] for a, b in pairs),
            "sector_saved_s": {name: median(a["sectors_s"][name] - b["sectors_s"][name] for a, b in pairs)
                               for name in sorted(names)}}


def evaluate_surface(surface, seeds, policy_factory=None, plot_seed=None):
    lab = Lab(surface=surface, measure_grip=True)
    record = {"layout": "shanghai", "surface": surface}

    def run_batch(factory):
        return summarize([run_lap(lab, seed, factory(), capture=plot_seed is not None,
                                  keep_profile=seed == plot_seed) for seed in seeds])

    try:
        record["baseline"] = run_batch(lambda: baseline)
        if policy_factory is not None:
            record["student"] = run_batch(policy_factory)
            record["paired"] = paired_times(record["baseline"], record["student"])
            record["grade"] = grade(record["baseline"]["episodes"], record["student"]["episodes"])
    finally:
        lab.close()
    return record


def combined_grade(surfaces):
    grades = {surface: result["grade"] for surface, result in surfaces.items()}
    if any(result["total"] is None for result in grades.values()):
        return {"eligible": False, "reason": "official_baseline_invalid", "total": None}
    failed = [surface for surface, result in grades.items() if not result["eligible"]]
    if failed:
        return {"eligible": False, "reason": "completion_below_95_percent", "surfaces_below_gate": failed,
                "pace_points": 0., "grip_points": 0., "total": 0.}
    return {"eligible": True, **{key: mean(result[key] for result in grades.values())
                                  for key in ("pace_points", "grip_points", "total")}}


def summarize_report(report):
    return {key: ({k: v for k, v in value.items() if k != "episodes"}
                  if key in ("baseline", "student") else value)
            for key, value in report.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("both", *SURFACES), default="both",
                        help="Evaluate both Shanghai surfaces by default")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path, help="Write speed and grip PNGs, one per surface")
    parser.add_argument("--plot-seed", type=int, help="Seed within this batch, default: first seed")
    args = parser.parse_args()
    if args.count < 1 or args.start_seed < 0:
        parser.error("count must be positive and start-seed nonnegative")
    if args.plot_seed is not None and (not args.plot or
                                       not args.start_seed <= args.plot_seed < args.start_seed + args.count):
        parser.error("--plot-seed requires --plot and must be within this batch")
    plot_seed = None
    if args.plot:
        plot_seed = args.start_seed if args.plot_seed is None else args.plot_seed
    policy_factory = None
    if args.checkpoint is not None:
        from student_policy import load_policy

        policy_factory = partial(load_policy, args.checkpoint)

    seeds = range(args.start_seed, args.start_seed + args.count)
    selected = tuple(SURFACES) if args.surface == "both" else (args.surface,)
    surfaces = {surface: evaluate_surface(surface, seeds, policy_factory, plot_seed) for surface in selected}
    if args.plot:
        from performance_plot import plot_comparison

        for surface, result in surfaces.items():
            path = (args.plot.with_name(f"{args.plot.stem}-{surface}{args.plot.suffix or '.png'}")
                    if args.surface == "both" else args.plot)
            plot_comparison(result, plot_seed, path)
    if args.surface == "both":
        record = {"layout": "shanghai", "surfaces": surfaces}
        summary = {"layout": "shanghai", "surfaces": {k: summarize_report(v) for k, v in surfaces.items()}}
        if policy_factory is not None:
            record["grade"] = summary["grade"] = combined_grade(surfaces)
    else:
        record = surfaces[args.surface]
        summary = summarize_report(record)
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
