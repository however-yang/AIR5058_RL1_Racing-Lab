"""Calibrate the existing MuJoCo vehicle and screen pace models on Shanghai."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from planned_racer import run_cases, summary
from racing import RaceEnv
from time_line import MeasuredPace, TimeLine


def calibrate():
    env = RaceEnv()
    data, model = env.data, env.model
    try:
        env.reset(seed=0)
        data.qpos[:2] = 0.
        data.qpos[3:7] = [1., 0., 0., 0.]
        mujoco.mj_forward(model, data)
        dt = model.opt.timestep

        def trace(target, duration):
            values = [math.hypot(*data.qvel[:2])]
            for _ in range(round(duration / dt)):
                env.vehicle.apply(0., target)
                mujoco.mj_step(model, data)
                values.append(math.hypot(*data.qvel[:2]))
            return np.asarray(values)

        acceleration = trace(5., 6.)
        deceleration = trace(0., 4.)
        def fit(values, levels, increasing):
            monotonic = np.maximum.accumulate(values if increasing else -values)
            thresholds = np.asarray(levels) if increasing else -np.asarray(levels)
            times = np.interp(thresholds, monotonic, np.arange(len(values)) * dt)
            midpoints = (np.asarray(levels[1:]) + levels[:-1]) / 2.
            rates = np.abs(np.diff(levels)) / np.diff(times)
            result = sorted(zip(midpoints.tolist(), rates.tolist()))
            assert all(math.isfinite(rate) and rate > 0. for _, rate in result)
            return result

        drive = fit(acceleration, [0., .8, 1.8, 2.8, 3.6, 4.1, 4.45, 4.55], True)
        brake = fit(deceleration, [4.55, 4.15, 3.6, 2.6, 1.6, .7, .12], False)
        return {"drive": drive, "brake": brake, "peak_speed_mps": float(acceleration.max()),
                "dt_s": dt, "mu": .90, "drive_trace_mps": acceleration.tolist(),
                "brake_trace_mps": deceleration.tolist()}
    finally:
        env.close()


def trial(job):
    folder, fit, lateral, preview = job
    torch.set_num_threads(1)
    env = RaceEnv(layout="shanghai")
    try:
        line = TimeLine(env.track, lateral, pace=MeasuredPace(fit["drive"], fit["brake"]),
                        brake_preview=preview)
    finally:
        env.close()
    label = f"lat{lateral:.2f}_preview{preview:.2f}"
    path = folder / f"{label}.npz"
    line.save(path)
    dry = run_cases("planner", range(6000, 6012), line=line)
    stress = run_cases("planner", range(6100, 6112), line=line, mu=.82)
    row = {"label": label, "line": str(path), "lateral": lateral, "preview": preview,
           "estimate_s": line.estimated_time, "dry": dry, "stress": stress}
    print(label, summary(dry), summary(stress), flush=True)
    return row


def key(row):
    dry, stress = row["dry"], row["stress"]
    eligible = min(dry["finish_rate"], stress["finish_rate"]) >= .95
    speed = -(dry["median_lap_s"] or float("inf"))
    if eligible:
        return 1, speed
    return 0, min(dry["finish_rate"], stress["finish_rate"]), dry["finish_rate"], speed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    fit = calibrate()
    (args.output / "calibration.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    print("Measured curves:", fit["drive"], fit["brake"], flush=True)
    jobs = [(args.output, fit, lateral, preview)
            for lateral in (6.8, 7.1, 7.4, 7.7) for preview in (.52, .85)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(trial, jobs))
    selected = max(rows, key=key)
    import shutil
    shutil.copyfile(selected["line"], args.output / "shanghai_line.npz")
    (args.output / "screen.json").write_text(json.dumps({"selected": selected, "candidates": rows}, indent=2),
                                            encoding="utf-8")
    print("Selected calibrated planner:", selected["label"], summary(selected["dry"]),
          summary(selected["stress"]), flush=True)


if __name__ == "__main__":
    main()
