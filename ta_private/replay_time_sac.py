"""Replay and measure the frozen lap-time winner in the project Conda environment."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mujoco
import numpy as np
from PIL import Image
import torch

from planned_racer import run_cases
from racing import RaceEnv
from time_line import TimeLine


def draw(rows, track, line, report, path, circuit="Spa"):
    data = np.asarray([[r["s"], r["x"], r["y"], r["v"], r["requested_speed"]] for r in rows])
    distance, x, y, speed, request = data.T
    center, indices = track.sample(distance)
    normals = track.tangents[:, ::-1] * [-1, 1]
    offset = np.sum((np.c_[x, y] - center) * normals[indices], axis=1)
    tangents = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
    signed_curve = (np.roll(tangents, -1) - np.roll(tangents, 1) + np.pi) % (2. * np.pi) - np.pi
    signed_curve /= 2. * np.median(track.lengths)
    curve = sum(np.roll(signed_curve, i) for i in range(-4, 5)) / 9.
    peaks = np.flatnonzero((np.abs(curve) > .28) & (np.abs(curve) >= np.abs(np.roll(curve, 1)))
                           & (np.abs(curve) >= np.abs(np.roll(curve, -1))))
    increasing = np.r_[True, np.diff(np.maximum.accumulate(distance)) > 0.]
    examples = []
    for i in peaks:
        middle = track.arc[i]
        if middle < 7. or middle + 3.4 > distance.max():
            continue
        inside_sign = np.sign(curve[i])
        middle_zone = np.flatnonzero(np.abs(distance - middle) < 1.4)
        if not len(middle_zone):
            continue
        apex = middle_zone[np.argmax(inside_sign * offset[middle_zone])]
        locations = [middle - 3.3, float(distance[apex]), middle + 3.3]
        side = np.interp(locations, distance[increasing], offset[increasing]) * inside_sign
        examples.append({"positions_m": locations, "inside_positive_m": side.tolist(),
                         "speed_mps": np.interp(locations, distance[increasing], speed[increasing]).tolist(),
                         "quality": float(min(-side[0], side[1], -side[2]))})
    example = max(examples, key=lambda e: e["quality"], default=None)
    highlights = example is not None and example["quality"] > .04

    fig = plt.figure(figsize=(10.5, 8), layout="constrained", facecolor="#f6f8fc")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.])
    course = fig.add_subplot(grid[0])
    pace = fig.add_subplot(grid[1])
    fig.suptitle(f"{circuit} | {report['time_s']:.2f} s | Actual line and speed", x=.09, ha="left",
                 fontsize=17, color="#18263a", fontweight="bold")
    for side in (-1, 1):
        edge = track.points + side * track.half_width * normals
        course.plot(*np.vstack([edge, edge[:1]]).T, color="#aab3bf", lw=.85)
    course.plot(*np.vstack([track.points, track.points[:1]]).T, color="#acb5c0", ls="--",
                lw=.9, label="Centerline")
    if line is not None:
        course.plot(*line.closed_points.T, color="#614d99", ls=":", lw=1.3, label="Planned line")
    positions = np.c_[x, y]
    actual = LineCollection(np.stack([positions[:-1], positions[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(2.5, 4.6), lw=2.8)
    actual.set_array(speed[:-1])
    course.add_collection(actual)
    if highlights:
        for label, sample, dy in zip(("Entry", "Apex", "Exit"), example["positions_m"], (12, -18, 12)):
            j = np.argmin(np.abs(distance - sample))
            course.scatter(x[j], y[j], s=32, color="#e57755", edgecolor="white", zorder=3)
            course.annotate(label, (x[j], y[j]), xytext=(9, dy), textcoords="offset points",
                            fontsize=9, color="#693a2c", arrowprops={"arrowstyle": "-", "color": "#aa7566"})
    course.set(aspect="equal", xlabel="X (m)", ylabel="Y (m)")
    course.legend(loc="upper left", frameon=False)
    fig.colorbar(actual, ax=course, label="Vehicle speed (m/s)", shrink=.8)

    if line is not None:
        pace.plot(distance, [r["planned_speed"] for r in rows], color="#614d99", linestyle=":",
                  lw=1.5, label="Planner speed request")
    pace.plot(distance, request, color="#a85c44", alpha=.75, lw=1., label="Final speed request")
    pace.plot(distance, speed, color="#086a78", lw=1.7, label="Actual speed")
    if highlights:
        pace.axvspan(example["positions_m"][0], example["positions_m"][2], color="#e57755", alpha=.13)
    pace.set(xlim=(0., track.length), ylim=(0., 5.2), xlabel="Centerline progress (m)", ylabel="Speed (m/s)")
    pace.legend(loc="lower right", frameon=False, ncols=2, fontsize=9)
    pace.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return {"example": example, "peak_speed_mps": float(speed.max()),
            "min_offset_m": float(offset.min()), "max_offset_m": float(offset.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate", choices=("winner", "sac", "td3", "previous"), default="winner")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads((args.bundle / "results.json").read_text(encoding="utf-8"))
    frozen = report["frozen"]
    if args.candidate == "winner":
        mode = frozen["mode"]
        checkpoint = args.bundle / "reference_policy.zip" if frozen["checkpoint_sha256"] else None
        if checkpoint:
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == frozen["checkpoint_sha256"]
        line = TimeLine.load(args.bundle / "spa_line.npz") if mode != "ppo" else None
    else:
        label = report["comparison_labels"][args.candidate]
        row = next(candidate for candidate in report["candidates"] if candidate["label"] == label)
        mode = row["mode"]
        checkpoint = Path(row["checkpoint"])
        line = None
        if mode != "ppo":
            spa = RaceEnv(layout="spa")
            try:
                line = TimeLine(spa.track, row["lateral_limit"])
            finally:
                spa.close()

    cases = report["spa_comparison"][args.candidate]["episodes"]
    expected = (next(ep for ep in cases if ep["seed"] == args.seed) if args.seed is not None else
                next(ep for ep in cases if ep["status"] == "finished"))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1., -1.25, 0.]
    camera.distance = 39.
    camera.azimuth = 112.
    camera.elevation = -30.
    result = run_cases(mode, [expected["seed"]], layout="spa", line=line,
                       checkpoint=checkpoint, replay=args.replay, trace=True, camera=camera)
    actual = result["episodes"][0]
    assert expected == actual, (expected, actual)
    if args.replay:
        with Image.open(args.replay) as gif:
            assert gif.n_frames > 50
    env = RaceEnv(layout="spa")
    try:
        if args.plot:
            data = draw(result["telemetry"], env.track, line, actual, args.plot)
            (args.bundle / f"spa_{args.candidate}_telemetry_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            print("Telemetry:", data)
    finally:
        env.close()
    (args.bundle / f"spa_{args.candidate}_telemetry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Frozen replay:", actual)


if __name__ == "__main__":
    main()
