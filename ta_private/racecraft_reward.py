"""Reward terms for whole-lap time, corner exits, and measured instability."""

import numpy as np

from racing.vehicle import MAX_STEER


STRAIGHT_BOOST = .30
EDGE_COST = .12
STEERING_COST = .08


def straight_time_weights(track, arcs):
    """Give long post-corner straights extra weight in the remaining-time estimate."""
    midpoints = (arcs[1:] + arcs[:-1]) / 2.
    spacing = arcs[1] - arcs[0]
    _, before = track.sample(midpoints - .75)
    _, after = track.sample(midpoints + .75)
    incoming, outgoing = track.tangents[before], track.tangents[after]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    curvature = np.abs(np.arctan2(cross, dot)) / 1.5
    straight = curvature <= .05
    weights = np.ones_like(midpoints)
    count = len(weights)

    for start in np.flatnonzero(straight & ~np.roll(straight, 1)):
        indices = (start + np.arange(count)) % count
        length = int(np.argmin(straight[indices]))
        distance = length * spacing
        approach = (start - np.arange(1, max(2, round(6. / spacing) + 1))) % count
        if distance < 7. or curvature[approach].max() < .10:
            continue
        position = (np.arange(length) + .5) * spacing
        taper = np.minimum(1., np.minimum(position, distance - position) / 1.5)
        strength = STRAIGHT_BOOST * min(1., (distance - 6.) / 7.)
        weights[indices[:length]] += strength * taper
    return weights


def driving_risk_seconds(signals):
    """Charge only for grip saturation coupled with tight clearance or fast steering."""
    load = float(np.clip(signals.grip_risk_sq, 0., 1.))
    edge = float(np.clip((.20 - signals.clearance_m) / .20, 0., 1.))
    rate = abs(signals.steering_change) * MAX_STEER / max(1e-6, signals.dt_s)
    fast_steering = float(np.clip((rate - .9) / .8, 0., 1.))
    moving = float(np.clip((signals.speed_mps - 1.) / 2., 0., 1.))
    return signals.dt_s * load * (EDGE_COST * edge**2 + STEERING_COST * moving * fast_steering**2)
