"""Private reward, reference training and held-out scoring."""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "student"))

from evaluate import evaluate, load_policy
from train import parser, train

VALIDATION_SEEDS = tuple(range(100, 120))
HOLDOUT_SEEDS = tuple(range(1500, 1540))


def reference_reward(s):
    shaping = 2. * s.progress_delta_m - .35 * s.dt_s
    shaping -= s.dt_s * (.9 * s.lateral_error_m ** 2 + .7 * s.heading_error_rad ** 2)
    shaping -= .025 * s.steering_change ** 2
    return shaping + 12. * s.finished - 8. * s.off_track - 2. * s.timed_out


def safe_reward(s):
    return reference_reward(s) - 6. * s.off_track - .08 * s.steering_change ** 2


def cornering_reward(s):
    curvature = abs(math.tan(.54 * s.steering)) / .384
    lateral_acceleration = s.speed_mps ** 2 * curvature
    excess = max(0., lateral_acceleration - 8.)
    line_cost = 3.4 * s.lateral_error_m ** 2 + 1.0 * s.heading_error_rad ** 2
    return safe_reward(s) - s.dt_s * (line_cost + .16 * excess ** 2)


def progress_reward(s):
    progress = s.new_progress_m + min(0., s.progress_delta_m)
    return 2. * progress - .35 * s.dt_s + 12. * s.finished - 8. * s.failed - 2. * s.timed_out


def racing_reward(s):
    edge_risk = min(1., max(0., (.15 - s.clearance_m) / .15))
    return progress_reward(s) - s.dt_s * (edge_risk ** 2 + .5 * s.grip_risk_sq)


def main():
    p = parser()
    p.add_argument("--profile", choices=("progress", "racing", "fast", "safe", "cornering"), default="racing")
    p.add_argument("--holdout", type=Path, help="Score a selected checkpoint on private test starts")
    p.add_argument("--replay", type=Path)
    p.add_argument("--validation-layout", choices=("shanghai", "spa"), default="spa",
                   help="Evaluation course used only with --holdout")
    args = p.parse_args()
    if args.holdout:
        result = evaluate(load_policy(args.holdout, args.algo), HOLDOUT_SEEDS, replay=args.replay, layout=args.validation_layout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print({k: v for k, v in result.items() if k != "episodes"})
    else:
        reward_fn = {"progress": progress_reward, "racing": racing_reward,
                     "fast": reference_reward, "safe": safe_reward, "cornering": cornering_reward}[args.profile]
        train(args, reward_fn, seeds=VALIDATION_SEEDS)


if __name__ == "__main__":
    main()
