"""Check the shared simulator and measure the scripted driving baseline."""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "student"))

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from evaluate import PUBLIC_SEEDS, evaluate
from racing import RaceEnv


def pure_pursuit(obs, speed=4.5):
    targets = obs[:10].reshape(5, 2) * RaceEnv.lookahead[-1]
    forward = max(0., obs[13] * RaceEnv.max_speed)
    distance = .75 + .29 * forward
    x = np.interp(distance, RaceEnv.lookahead, targets[:, 0])
    y = np.interp(distance, RaceEnv.lookahead, targets[:, 1])
    steer = np.arctan2(2 * .384 * y, x*x + y*y)
    return np.clip([steer / RaceEnv.max_steer, 2 * speed / RaceEnv.max_speed - 1.], -1., 1.)


def adaptive_pursuit(obs):
    points = obs[:10].reshape(5, 2) * RaceEnv.lookahead[-1]
    incoming = points[1:-1] - points[:-2]
    outgoing = points[2:] - points[1:-1]
    area = np.abs(incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0])
    span = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    span *= np.linalg.norm(incoming + outgoing, axis=1)
    curvature = 2. * area / np.maximum(span, 1e-6)
    speed = min(RaceEnv.max_speed, math.sqrt(5.6 / max(.01, float(curvature.max()))))
    return pure_pursuit(obs, speed)


def cross_line(env, line, reverse=False):
    env.reset(seed=0)
    env.arc = line + .10 if reverse else line - .10
    position, env.segment = env.track.sample(env.arc)
    tangent = env.track.tangents[env.segment]
    yaw = math.atan2(tangent[1], tangent[0]) + (math.pi if reverse else 0.)
    env.data.qpos[:2] = position
    env.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(10):
        _, _, terminated, truncated, info = env.step([0., .6])
        assert not (terminated or truncated), info
    assert env.arc < line if reverse else env.arc > line
    return info


def main():
    env = RaceEnv()
    check_env(env)
    np.testing.assert_array_equal(env.reset(seed=7)[0], env.reset(seed=7)[0])
    assert cross_line(env, env.track.gates[0])["gates"] == 1
    assert cross_line(env, env.track.gates[0], reverse=True)["gates"] == 0
    skipped = cross_line(env, env.track.length)
    assert skipped["gates"] == 0 and skipped["progress"] <= .125
    for action, expected in [([0., -1.], "timeout"), ([0., 1.], "off_track")]:
        report = evaluate(lambda _: np.array(action), [0])
        assert report["episodes"][0]["status"] == expected, report
    for speed in [3.5, 4.5, 5.0]:
        results = evaluate(lambda obs: pure_pursuit(obs, speed), PUBLIC_SEEDS)
        summary = {k: v for k, v in results.items() if k != "episodes"}
        print(speed, json.dumps(summary), flush=True)
        if speed == 3.5:
            assert results["finish_rate"] == 1., results
            assert all(ep["gates"] == 8 for ep in results["episodes"])
    print("Gymnasium/SB3, deterministic reset, timeout, departure, checkpoint order, and lap checks passed.")


if __name__ == "__main__":
    main()
