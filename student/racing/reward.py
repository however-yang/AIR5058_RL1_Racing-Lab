"""Implement your training reward here. See the assignment brief."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSignals:
    progress_delta_m: float       # Signed change in centerline arc length
    new_progress_m: float         # Newly reached legal lap distance
    lateral_error_m: float       # Positive to the left of the centerline
    heading_error_rad: float     # Car yaw minus track yaw, wrapped to [-pi, pi]
    speed_mps: float             # Signed forward speed
    steering: float              # Normalized command in [-1, 1]
    steering_change: float       # Signed change from the previous command
    clearance_m: float            # Minimum distance from the footprint to the road edge
    grip_risk_sq: float           # Load-weighted squared risk above 85% grip utilization
    dt_s: float                  # Elapsed simulation time for this step
    finished: bool
    off_track: bool
    timed_out: bool
    failed: bool                  # Departure, rollover, or sustained overspeed


def reward(signals: RewardSignals) -> float:
    """Smoke-test reward. Replace it with your own shaping before training."""
    return float(signals.finished)
