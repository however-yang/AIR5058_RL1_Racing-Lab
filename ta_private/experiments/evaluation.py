"""Common timing, selection, and frozen-asset checks for TA experiments."""


import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile


import numpy as np
import torch

from planned_racer import run_cases
from time_line import TimeLine


SECTORS = {
    "shanghai": (("spiral_back_straight", 74., 116.),),
    "spa": (("eau_rouge_kemmel", 18., 41.), ("bruxelles_to_next_straight_end", 48., 64.)),
}


WINDOWS = {
    "shanghai": (("spiral_back_straight", 74., 96., 116.),),
    "spa": (("eau_rouge_kemmel", 18., 32., 41.), ("bruxelles_to_next_straight_end", 48., 60., 64.)),
}
SPLIT_SECTORS = {
    layout: tuple(sector for name, start, exit_arc, end in windows
                  for sector in ((name, start, end), (name + "_through_corner", start, exit_arc),
                                 (name + "_after_corner", exit_arc, end)))
    for layout, windows in WINDOWS.items()
}


def artifact_path(path):
    """Read repository-relative paths from Windows and POSIX reports."""
    return Path(str(path).replace("\\", "/"))


def sha(path):
    return hashlib.sha256(artifact_path(path).read_bytes()).hexdigest()


def model_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = ("racing/car.xml", "racing/vehicle.py", "racing/env.py", "racing/track.py")
    with ZipFile(root / "mujoco-racing-student.zip") as previous:
        for name in paths:
            if (root / "student" / name).read_bytes() != previous.read(name):
                raise RuntimeError(f"Physics or layout differs from the released student model: {name}")
    return {name: sha(root / "student" / name) for name in paths}


def admitted(row):
    return min(row["dry"]["finish_rate"], row["stress"]["finish_rate"]) >= .95


def rank(row):
    pace = -(row["dry"]["median_lap_s"] or float("inf"))
    if admitted(row):
        return 1, pace, row["dry"]["finish_rate"], row["stress"]["finish_rate"]
    return 0, min(row["dry"]["finish_rate"], row["stress"]["finish_rate"]), row["dry"]["finish_rate"], pace


def timing(result, layout, baseline=None, sectors=None):
    complete = {ep["seed"]: ep for ep in result["episodes"] if ep["status"] == "finished"}
    reference = ({ep["seed"]: ep for ep in baseline["episodes"] if ep["status"] == "finished"}
                 if baseline is not None else {})
    values = np.asarray([ep["time_s"] for ep in complete.values()])
    def stats(times, prior):
        common = sorted(times.keys() & prior.keys())
        difference = [prior[seed] - times[seed] for seed in common]
        return {"median_s": float(np.median(list(times.values()))) if times else None,
                "paired_count": len(common),
                "median_paired_saving_s": float(np.median(difference)) if difference else None}

    output = {
        "finish": [len(complete), len(result["episodes"])],
        "lap": stats({k: ep["time_s"] for k, ep in complete.items()},
                     {k: ep["time_s"] for k, ep in reference.items()}),
        "p90_lap_s": float(np.percentile(values, 90)) if values.size else None,
        "sectors": {},
    }
    for name, _, _ in SECTORS[layout] if sectors is None else sectors:
        times = {k: ep["sectors_s"][name] for k, ep in complete.items() if name in ep["sectors_s"]}
        prior = {k: ep["sectors_s"][name] for k, ep in reference.items() if name in ep["sectors_s"]}
        if len(times) != len(complete):
            raise RuntimeError(f"Missing fixed-sector timing on {layout}: {name}")
        output["sectors"][name] = stats(times, prior)
    return output


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate(row, layout, seeds, mu=None, *, sectors=SPLIT_SECTORS):
    torch.set_num_threads(1)
    return run_cases(row["mode"], seeds, layout=layout, line=TimeLine.load(artifact_path(row[layout + "_line"])),
                     checkpoint=artifact_path(row["checkpoint"]), residual_scale=row["residual_scale"],
                     lane_limit=row.get("lane_limit", 0.), lane_gain=row.get("lane_gain", 1.),
                     mu=mu, sectors=sectors[layout])


def freeze_policy_assets(output, groups):
    """Record every representative, and copy only those allowed to transfer."""
    fingerprints = {name: {"line": sha(Path(row["shanghai_line"])),
                           "checkpoint": sha(Path(row["checkpoint"]))}
                    for name, row in groups.items()}
    for name, row in groups.items():
        if not admitted(row):
            continue
        directory = output / name
        directory.mkdir(exist_ok=True)
        for field, filename in (("shanghai_line", "shanghai_line.npz"), ("checkpoint", "policy.zip")):
            target = directory / filename
            shutil.copyfile(artifact_path(row[field]), target)
            row[field] = target.as_posix()
    return fingerprints


def prepare_spa_reference(output, source, expected_hash, qualified):
    """Copy and verify the reference after Shanghai selection has been written."""
    target = output / "spa_line.npz"
    shutil.copyfile(source, target)
    if sha(target) != expected_hash:
        raise RuntimeError("The shared Spa reference line changed")
    for row in qualified.values():
        row["spa_line"] = target.as_posix()
    return target, dict.fromkeys(qualified, expected_hash)


def verify_frozen_assets(spa_line, spa_hash, qualified, fingerprints):
    if sha(spa_line) != spa_hash:
        raise RuntimeError("The Spa reference line changed during evaluation")
    for name, row in qualified.items():
        if (sha(Path(row["checkpoint"])) != fingerprints[name]["checkpoint"]
                or sha(Path(row["shanghai_line"])) != fingerprints[name]["line"]):
            raise RuntimeError(f"Frozen Shanghai assets changed during evaluation: {name}")
