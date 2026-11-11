"""Plot the frozen driver's actual Spa line and commanded speed."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from reference import load_policy
from racing import RaceEnv


def collect(checkpoint, seed):
    env = RaceEnv(layout="spa")
    policy = load_policy(checkpoint, "ppo")
    obs, info = env.reset(seed=seed)
    rows = []
    try:
        while info["status"] == "running":
            action = policy(obs)
            rows.append([env.arc, *env.data.qpos[:2], np.linalg.norm(env.data.qvel[:2]),
                         (action[1] + 1.) * .5 * env.max_speed,
                         env.vehicle.speed_target, obs[10] * env.track.half_width])
            obs, _, _, _, info = env.step(action)
        return env.track, np.asarray(rows), info
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    finished = [ep for ep in result["spa_transfer"]["episodes"] if ep["status"] == "finished"]
    if not finished:
        parser.error("the frozen policy has no successful Spa laps")
    sample = finished[0]
    track, rows, info = collect(args.run / "reference_policy.zip", sample["seed"])
    assert info["status"] == "finished" and abs(info["time_s"] - sample["time_s"]) < 1e-9
    distance, x, y, speed, request, reference, lateral = rows.T
    angle = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
    curvature = np.abs((np.roll(angle, -1) - np.roll(angle, 1) + np.pi) % (2 * np.pi) - np.pi)
    curvature /= 2 * np.median(track.lengths)
    smooth = sum(np.roll(curvature, i) for i in range(-4, 5)) / 9
    local_curve = np.interp(np.maximum(distance, 0), track.arc[:-1], smooth)
    straight = (local_curve < .035) & (distance > 5.)
    corner = local_curve > .32

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), layout="constrained", facecolor="#f7f8fb")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1])
    course = fig.add_subplot(grid[0])
    profile = fig.add_subplot(grid[1])
    fig.suptitle(f"Spa transfer lap | {sample['time_s']:.2f} s | seed {sample['seed']}",
                 x=.09, ha="left", fontsize=17, color="#132135", fontweight="bold")

    normal = track.tangents[:, ::-1] * [-1, 1]
    for side in (-1, 1):
        edge = track.points + side * track.half_width * normal
        edge = np.vstack([edge, edge[0]])
        course.plot(*edge.T, color="#a7b0bc", linewidth=.85)
    center = np.vstack([track.points, track.points[0]])
    course.plot(*center.T, color="#c6ccd4", linestyle="--", linewidth=.9, label="Centerline")
    xy = np.column_stack([x, y])
    path = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap="viridis",
                          norm=plt.Normalize(2.5, 4.6), linewidth=2.8)
    path.set_array(speed[:-1])
    course.add_collection(path)
    course.scatter(x[0], y[0], s=45, color="#eb6b42", edgecolor="white", zorder=3, label="Start")
    course.set_aspect("equal")
    course.set_ylabel("Y (m)")
    course.set_xlabel("X (m)")
    course.legend(loc="upper left", frameon=False)
    fig.colorbar(path, ax=course, label="Actual speed (m/s)", shrink=.8)

    profile.plot(distance, request, color="#a86653", alpha=.68, linewidth=1., label="Policy speed request")
    profile.plot(distance, speed, color="#146d82", linewidth=1.8, label="Actual vehicle speed")
    profile.fill_between(distance, 0., 5.25, where=corner, color="#a7b0bc", alpha=.23,
                         step="mid", label="Higher-curvature sections")
    profile.set(ylim=(0, 5.25), xlim=(0, track.length), xlabel="Centerline progress (m)",
                ylabel="Speed (m/s)")
    profile.grid(axis="y", alpha=.2)
    profile.legend(loc="lower right", frameon=False, ncols=2, fontsize=9)
    fig.savefig(args.plot, dpi=155)
    plt.close(fig)

    # Compare requested speed before and near each curvature peak on this completed lap.
    increasing = np.r_[True, np.diff(np.maximum.accumulate(distance)) > 0]
    examples = []
    peaks = np.flatnonzero((smooth > .32) & (smooth >= np.roll(smooth, 1))
                           & (smooth >= np.roll(smooth, -1)))
    for idx in peaks:
        apex = track.arc[idx]
        if apex < 6. or apex + 2.5 > distance.max():
            continue
        samples = [apex - 3.5, apex - .55, apex + 2.5]
        demands = np.interp(samples, distance[increasing], request[increasing])
        velocities = np.interp(samples, distance[increasing], speed[increasing])
        examples.append({"apex_m": float(apex), "locations_m": samples,
                         "requested_mps": demands.tolist(), "actual_mps": velocities.tolist()})
    illustrated = max(examples, key=lambda e: e["requested_mps"][0] - e["requested_mps"][1],
                      default=None)

    telemetry = {
        "seed": sample["seed"],
        "lap_s": info["time_s"],
        "peak_speed_mps": float(speed.max()),
        "straight_mean_mps": float(speed[straight].mean()) if straight.any() else None,
        "corner_mean_mps": float(speed[corner].mean()) if corner.any() else None,
        "straight_request_mean_mps": float(request[straight].mean()) if straight.any() else None,
        "corner_request_mean_mps": float(request[corner].mean()) if corner.any() else None,
        "braking_example": illustrated,
        "lateral_min_m": float(lateral.min()), "lateral_max_m": float(lateral.max()),
    }
    (args.run / "spa_lap_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print(telemetry)


if __name__ == "__main__":
    main()
