"""Replay the fastest reliable Spa result and verify paired timing against the report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image
import torch

from experiments.evaluation import SECTORS, artifact_path, model_fingerprint, sha
from planned_racer import run_cases
from time_line import TimeLine


def replay(row, seed, camera, gif=None, sectors=SECTORS["spa"]):
    return run_cases(row["mode"], [seed], layout="spa", line=TimeLine.load(artifact_path(row["spa_line"])),
                     checkpoint=artifact_path(row["checkpoint"]) if row.get("checkpoint") else None, replay=gif, trace=True, camera=camera,
                     sectors=sectors, lane_limit=row.get("lane_limit", 0.), lane_gain=row.get("lane_gain", 1.),
                     **({"residual_scale": row["residual_scale"]} if "residual_scale" in row else {}))


def plot(path, baseline, fastest, before, after, baseline_label, winner_label):
    def unpack(result):
        rows = result["telemetry"]
        data = np.asarray([[r[k] for k in ("s", "t", "v", "lateral_m")] for r in rows])
        advancing = np.r_[True, np.diff(np.maximum.accumulate(data[:, 0])) > 0.]
        return data[advancing].T

    arc0, time0, speed0, lateral0 = unpack(baseline)
    arc1, time1, speed1, lateral1 = unpack(fastest)
    end = min(arc0[-1], arc1[-1])
    grid = np.linspace(0., end, 650)
    saving = np.interp(grid, arc0, time0) - np.interp(grid, arc1, time1)
    show_lateral = "lane_fraction" in fastest["telemetry"][0]
    fig, axes = plt.subplots(3 if show_lateral else 2, 1,
                             figsize=(10.5, 8.4 if show_lateral else 6.4), sharex=True,
                             layout="constrained", facecolor="#f6f8fc")
    fig.suptitle(f"Spa paired lap  |  {before['time_s']:.2f} s → {after['time_s']:.2f} s  |  Seed {after['seed']}",
                 x=.08, ha="left", fontsize=16, fontweight="bold", color="#15263a")
    labels = ("Eau Rouge to Kemmel end", "Bruxelles to next straight end")
    colors = ("#7ec6c4", "#edb78b")
    for ax in axes:
        for (_, a, b), label, color in zip(SECTORS["spa"], labels, colors):
            ax.axvspan(a, b, color=color, alpha=.21, label=label if ax is axes[0] else None)
        ax.grid(axis="y", alpha=.18)

    axes[0].plot(grid, saving, lw=2.1, color="#076d77", label="Time gained vs baseline")
    axes[0].axhline(0., color="#777f89", lw=.8)
    axes[0].set_ylabel("Cumulative gain (s)")
    axes[0].legend(loc="best", frameon=False, ncols=2, fontsize=9)
    axes[1].plot(arc0, speed0, color="#6e7c8a", lw=1.15, label=baseline_label)
    axes[1].plot(arc1, speed1, color="#076d77", lw=1.7,
                 label=winner_label)
    axes[1].set(xlim=(0, end), ylim=(0, 5.1), ylabel="Actual speed (m/s)")
    if show_lateral:
        axes[2].plot(arc0, lateral0, color="#6e7c8a", lw=1.15)
        axes[2].plot(arc1, lateral1, color="#076d77", lw=1.7)
        axes[2].axhline(0., color="#777f89", lw=.8)
        axes[2].set_ylabel("Lateral position (m)")
    axes[-1].set_xlabel("Centerline progress (m)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.savefig(path, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--group", help="Replay a specific group that raced in Spa")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads((args.bundle / "results.json").read_text(encoding="utf-8"))
    fastest = args.group or report["fastest_observed_spa"]
    if fastest is None:
        raise RuntimeError("No Spa group reached 95% finish rate")
    if fastest not in report["spa"]:
        raise ValueError(f"No Spa evaluation for group {fastest}")
    group = report["groups"][fastest]
    assert model_fingerprint() == report["frozen"]["physics"]
    fingerprints = report["frozen"]["sha256"][fastest]
    assert sha(Path(group["shanghai_line"])) == fingerprints["line"]
    assert sha(Path(group["spa_line"])) == report["spa_line_sha256"][fastest]
    if group.get("checkpoint"):
        assert sha(Path(group["checkpoint"])) == fingerprints["checkpoint"]

    old = {ep["seed"]: ep for ep in report["spa"].get("baseline", {"episodes": []})["episodes"] if ep["status"] == "finished"}
    final = report["spa"][fastest]
    complete = [ep for ep in final["episodes"] if ep["status"] == "finished"]
    matched = [ep for ep in complete if ep["seed"] in old]
    expected = min(matched or complete, key=lambda ep: abs(ep["time_s"] - final["median_lap_s"]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1., -1.25, 0.]
    camera.distance = 39.
    camera.azimuth = 112.
    camera.elevation = -30.
    sectors = report["frozen"]["sectors_m"]["spa"]
    output = replay(group, expected["seed"], camera, args.gif, sectors=sectors)
    assert output["episodes"][0] == expected, (output["episodes"][0], expected)
    with Image.open(args.gif) as gif:
        assert gif.n_frames > 50 and gif.size == (540, 338)
    if args.plot and expected["seed"] in old:
        seed = expected["seed"]
        prior = report["groups"]["baseline"]
        baseline = output if fastest == "baseline" else replay(prior, seed, camera, sectors=sectors)
        assert baseline["episodes"][0] == old[seed]
        other = (next((name for name in ("racecraft", "lateral", "multi_line") if name in report["spa"]), fastest)
                 if fastest == "baseline" else fastest)
        comparison = report["groups"][other]
        paired = next((ep for ep in report["spa"][other]["episodes"]
                       if ep["seed"] == seed and ep["status"] == "finished"), None)
        if paired is not None:
            second = output if other == fastest else replay(comparison, seed, camera, sectors=sectors)
            assert second["episodes"][0] == paired
            label = comparison.get("display_name", {"baseline": "Selected baseline",
                                                    "measured": "Measured-curve planner",
                                                    "time_sac": "Time-reward SAC"}.get(other, other))
            plot(args.plot, baseline, second, old[seed], paired,
                 prior.get("display_name", "Previous planner"), label)
    name = "replay_lap.json" if args.group is None else f"replay_lap_{args.group}.json"
    (args.bundle / name).write_text(json.dumps({"group": fastest, **output}, indent=2), encoding="utf-8")
    print("Verified Spa replay:", fastest, expected)


if __name__ == "__main__":
    main()
