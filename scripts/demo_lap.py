"""Run the saved TA policy on one fixed MuJoCo lap."""

import argparse
from pathlib import Path
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))

from evaluate import evaluate, load_policy
from racing import RaceEnv

SEED = 1500


def set_sideline(camera, mujoco):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1., -1.25, 0.]
    camera.distance = 39.
    camera.azimuth = 112.
    camera.elevation = -30.


def live(policy, camera, layout, seed):
    import mujoco
    from mujoco import viewer

    env = RaceEnv(render_mode="rgb_array", layout=layout)
    obs, _ = env.reset(seed=seed)
    try:
        with viewer.launch_passive(
            env.model, env.data, show_left_ui=False, show_right_ui=False
        ) as window:
            with window.lock():
                if camera == "sideline":
                    set_sideline(window.cam, mujoco)
                else:
                    window.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    window.cam.fixedcamid = env.model.camera(camera).id
            while window.is_running():
                with window.lock():
                    obs, _ = env.reset(seed=seed)
                window.sync()
                sleep(1.5)
                start = perf_counter()
                done = False
                while window.is_running() and not done:
                    action = policy(obs)
                    with window.lock():
                        obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    window.sync()
                    sleep(max(0., start + info["time_s"] - perf_counter()))
                if done:
                    print(info, flush=True)
                    sleep(2.5)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--replay", type=Path)
    parser.add_argument("--camera", choices=("overview", "chase", "sideline"), default="chase")
    parser.add_argument("--model", type=Path, default=ROOT / "ta_private/best/reference_policy.zip")
    parser.add_argument("--layout", choices=("shanghai", "spa"), default="shanghai")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    policy = load_policy(args.model, "ppo")
    if args.live:
        live(policy, args.camera, args.layout, args.seed)
    else:
        camera = args.camera
        if camera == "sideline":
            import mujoco

            camera = mujoco.MjvCamera()
            set_sideline(camera, mujoco)
        episode, = evaluate(policy, [args.seed], replay=args.replay, camera=camera, layout=args.layout)["episodes"]
        print(episode, flush=True)


if __name__ == "__main__":
    main()
