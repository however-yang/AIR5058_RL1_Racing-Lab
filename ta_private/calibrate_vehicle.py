"""Verify the physical model before starting any policy training."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from check_env import adaptive_pursuit, cross_line, pure_pursuit
from evaluate import evaluate
from racing import RaceEnv
from racing.vehicle import MAX_STEER


def bench(dt):
    env = RaceEnv(physics_dt=dt, diagnostics=True)
    data, model, vehicle = env.data, env.model, env.vehicle

    def reset(mu=.90):
        env.reset(seed=0)
        data.qpos[:2] = 0.
        data.qpos[3:7] = [1., 0., 0., 0.]
        model.pair_friction[:, :2] = mu
        mujoco.mj_forward(model, data)

    def phase(seconds, steer, speed):
        rows = []
        for _ in range(round(seconds / dt)):
            vehicle.apply(steer, speed)
            mujoco.mj_step(model, data)
            _, _, grip, slip = vehicle.contacts()
            ground_speed = math.hypot(*data.qvel[:2])
            rows.append([ground_speed, data.qvel[5], grip, slip, vehicle.speed_target,
                         *vehicle.steer_target, *data.qpos[vehicle.steer_qpos]])
        return np.asarray(rows)

    def circle(speed, mu):
        reset(mu)
        entry = phase(5., 0., speed)[-1, 0]
        trace = phase(3., math.atan(.384 / 1.5), speed)
        steady = trace[-round(1. / dt):]
        radius = np.median(steady[:, 0] / np.maximum(np.abs(steady[:, 1]), .01))
        return {"entry_mps": float(entry), "radius_m": float(radius),
                "slip_mps": float(np.quantile(steady[:, 3], .9)),
                "grip": float(np.quantile(steady[:, 2], .9))}

    def corner_brake(speed):
        reset()
        steer = math.atan(.384 / 1.5)
        phase(5., 0., 4.)
        phase(1.6, steer, 4.)
        return phase(.50, steer, speed)

    try:
        reset()
        straight = phase(6., 0., 5.)
        braking = phase(4., 0., 0.)
        stop = int(np.flatnonzero(braking[:, 0] <= .10)[0])
        assert np.max(straight[:, 0]) <= 5.10
        assert straight[-1, 0] >= 4.
        assert np.max(np.diff(np.r_[0., straight[:, 4]])) <= 3. * dt + 1e-8
        assert np.max(-np.diff(np.r_[straight[-1, 4], braking[:, 4]])) <= 5. * dt + 1e-8
        assert braking[-1, 0] < .05
        assert np.min(data.qvel[vehicle.rear_dof]) > -.1

        phase(2., MAX_STEER, 0.)
        initial = data.qpos[vehicle.steer_qpos].copy()
        initial_target = vehicle.steer_target.copy()
        turn = phase(2., -MAX_STEER, 0.)
        target_rate = np.max(np.abs(np.diff(np.vstack([initial_target, turn[:, 5:7]]), axis=0))) / dt
        actual_rate = np.max(np.abs(np.diff(np.vstack([initial, turn[:, 7:9]]), axis=0))) / dt
        assert target_rate <= 2. + 1e-7
        assert actual_rate <= 2.00, (dt, target_rate, actual_rate)
        assert initial[0] > initial[1] > 0. and turn[-1, 8] < turn[-1, 7] < 0.

        reset()
        phase(2., MAX_STEER, 3.)
        rolling_start = data.qpos[vehicle.steer_qpos].copy()
        rolling_turn = phase(2., -MAX_STEER, 3.)
        rolling_rate = np.max(np.abs(np.diff(np.vstack([rolling_start, rolling_turn[:, 7:9]]), axis=0))) / dt
        held = corner_brake(4.)
        braked = corner_brake(0.)
        assert braked[-1, 0] < held[-1, 0] - .3
        assert max(held[:, 2].max(), braked[:, 2].max()) <= 1.001

        low = circle(3., .90)
        high = circle(5., .90)
        mid = circle(4., .90)
        slick = circle(4., .65)
        assert abs(low["radius_m"] - 1.5) < .40
        assert high["entry_mps"] > low["entry_mps"] + 1.
        assert high["radius_m"] > low["radius_m"] + .30
        assert high["slip_mps"] > low["slip_mps"] + .10
        assert slick["slip_mps"] > mid["slip_mps"] + .05, (mid, slick)
        assert np.allclose(model.pair_friction[:, :2], .65)
        return {"dt_s": dt, "peak_straight_mps": float(straight[:, 0].max()),
                "brake_to_point1_s": (stop + 1) * dt,
                "brake_distance_m": float(np.sum(braking[:stop + 1, 0]) * dt),
                "steer_target_rate_rad_s": float(target_rate),
                "steer_actual_rate_rad_s": float(actual_rate),
                "rolling_steer_rate_rad_s": float(rolling_rate),
                "corner_hold_exit_mps": float(held[-1, 0]), "corner_brake_exit_mps": float(braked[-1, 0]),
                "circle_slow_dry": low,
                "circle_fast_dry": high, "circle_mid_dry": mid, "circle_slick": slick}
    finally:
        env.close()


def main():
    report = {"bench": []}
    for dt in (.01, .005):
        result = bench(dt)
        report["bench"].append(result)
        print(json.dumps(result), flush=True)
    ten, five = report["bench"]
    assert abs(ten["brake_distance_m"] - five["brake_distance_m"]) < .20
    env = RaceEnv()
    check_env(env)
    np.testing.assert_array_equal(env.reset(seed=7)[0], env.reset(seed=7)[0])
    assert cross_line(env, env.track.gates[0])["gates"] == 1
    assert cross_line(env, env.track.gates[0], reverse=True)["gates"] == 0
    assert cross_line(env, env.track.length)["gates"] == 0
    env.reset(seed=0)
    for invalid in ([0., np.nan], [[1, 2], [3]], [2., 0.]):
        env.reset(seed=0)
        assert env.step(invalid)[4]["status"] == "invalid_action"
    env.close()
    report["courses"] = {}
    for layout in ("shanghai", "spa"):
        # Engineering-only starts. No learned policy is selected using Spa.
        seeds = range(3)
        slow = evaluate(adaptive_pursuit, seeds, layout=layout)
        fast = evaluate(lambda obs: pure_pursuit(obs, 5.), seeds, layout=layout)
        assert slow["finish_rate"] == 1., slow
        report["courses"][layout] = {"conservative": slow, "full_throttle": fast}
        finer_slow = evaluate(adaptive_pursuit, [0], layout=layout, physics_dt=.005)
        finer_fast = evaluate(lambda obs: pure_pursuit(obs, 5.), [0], layout=layout, physics_dt=.005)
        assert finer_slow["finish_rate"] == 1.
        assert finer_fast["episodes"][0]["status"] == fast["episodes"][0]["status"]
        report["courses"][layout].update({"fine_conservative": finer_slow, "fine_full_throttle": finer_fast})
        print(layout, slow["finish_rate"], slow["median_lap_s"], fast["finish_rate"], flush=True)
    assert any(c["full_throttle"]["off_track_rate"] > 0. for c in report["courses"].values())
    destination = Path(__file__).with_name("physics_calibration.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Physical and competition checks passed.")


if __name__ == "__main__":
    main()
