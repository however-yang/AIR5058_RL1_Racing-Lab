"""Estimate time to finish from several reachable racing lines."""

from copy import copy
import math

import numpy as np
import torch

from racing.vehicle import MAX_STEER, STEER_RATE, WHEELBASE
from time_line import MAX_GROUND_SPEED, geometry, speed_envelope


def racing_lines(track, reference):
    """Keep the existing line and vary the amount and timing of lateral placement."""
    arcs = reference.arcs[:-1]
    center, _ = track.sample(arcs)
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = tangent[:, ::-1] * [-1., 1.]
    offset = ((reference.points - center) * normal).sum(axis=1)
    width = track.half_width - .40
    offsets = (
        .68 * offset,
        offset + .22 * offset * (1. - (offset / width)**2),
        np.interp(arcs + .85, arcs, offset, period=track.length),
        np.interp(arcs - .85, arcs, offset, period=track.length),
    )
    yield reference
    for placement in offsets:
        yield with_speed_envelope(reference, center + placement[:, None] * normal, reference.lateral_limit)


def with_speed_envelope(reference, points, lateral_limit):
    line = copy(reference)
    line.points = points
    line.lateral_limit = lateral_limit
    length, curvature = geometry(torch.as_tensor(points, dtype=torch.float32))
    line.speeds = speed_envelope(length, curvature, lateral_limit, line.pace).numpy()
    line.closed_points = np.vstack((points, points[:1]))
    line.closed_speeds = np.r_[line.speeds, line.speeds[0]]
    return line


class MultiLineTimeToGo:
    def __init__(self, track, reference, single_line_estimate, step_weights=None):
        self.lines = list(racing_lines(track, reference))
        self.finish = reference.length
        self.spacing = reference.step
        self.step_weights = step_weights
        self.midpoints = (reference.arcs[1:] + reference.arcs[:-1]) / 2.
        self.lateral_limit = reference.lateral_limit
        self.pace = reference.pace
        self.fallback = single_line_estimate(reference)
        self.points = np.stack([line.points for line in self.lines])
        self.derivative = (np.roll(self.points, -1, axis=1) - np.roll(self.points, 1, axis=1)) / (2. * self.spacing)
        self.second = (np.roll(self.points, -1, axis=1) - 2. * self.points
                       + np.roll(self.points, 1, axis=1)) / self.spacing**2
        self.speeds = np.stack([line.closed_speeds for line in self.lines])
        self.suffix = np.stack([single_line_estimate(line, curvature=self.curvature(first, second)).times
                                for line, first, second in zip(self.lines, self.derivative, self.second)])
        self.speed_grid = self.fallback.speed_grid
        self.rows = np.arange(len(self.lines))
        self.fractions = np.linspace(0., 1., 9)
        u = self.fractions
        # Quintic correction with position, tangent and curvature matched at both ends.
        self.blend = np.array([1. - 10.*u**3 + 15.*u**4 - 6.*u**5,
                               u - 6.*u**3 + 8.*u**4 - 3.*u**5,
                               .5 * (u**2 - 3.*u**3 + 3.*u**4 - u**5)]).T
        self.blend_first = np.array([-30.*u**2 + 60.*u**3 - 30.*u**4,
                                     1. - 18.*u**2 + 32.*u**3 - 15.*u**4,
                                     u - 4.5*u**2 + 6.*u**3 - 2.5*u**4]).T
        self.blend_second = np.array([-60.*u + 180.*u**2 - 120.*u**3,
                                      -36.*u + 96.*u**2 - 60.*u**3,
                                      1. - 9.*u + 18.*u**2 - 10.*u**3]).T

    @staticmethod
    def curvature(first, second):
        return ((first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
                / np.maximum(np.linalg.norm(first, axis=-1), .2)**3)

    def remaining(self, env, arc):
        """Return a time for each feasible join and infinity for infeasible joins."""
        velocity = env.data.qvel[:2]
        speed = np.linalg.norm(velocity)
        heading = env._yaw()
        body_direction = np.array([math.cos(heading), math.sin(heading)])
        # At low speed use the actual heading. At speed the velocity also reflects sideslip.
        direction = velocity + .65 * body_direction
        direction /= max(np.linalg.norm(direction), 1e-5)
        actual_steer = env.data.qpos[env.vehicle.steer_qpos].mean()
        actual_curvature = math.tan(actual_steer) / WHEELBASE
        reach = min(3.4 + .30 * speed, self.finish - arc)
        arcs = arc + reach * self.fractions
        steps = np.remainder(arcs, self.finish) / self.spacing
        left = np.floor(steps).astype(int) % self.points.shape[1]
        right = (left + 1) % self.points.shape[1]
        ratio = (steps % 1.)[None, :, None]
        points, first, second = (values[:, left] + ratio * (values[:, right] - values[:, left])
                                 for values in (self.points, self.derivative, self.second))
        slope = np.linalg.norm(first[:, 0], axis=1)[:, None]
        inward = direction[::-1] * [-1., 1.]
        delta = np.stack((env.data.qpos[:2] - points[:, 0],
                          reach * (slope * direction - first[:, 0]),
                          reach**2 * (slope**2 * actual_curvature * inward - second[:, 0])), axis=1)
        points += self.blend @ delta
        first += (self.blend_first @ delta) / reach
        second += (self.blend_second @ delta) / reach**2
        curvature = self.curvature(first, second)
        wheel = np.arctan(WHEELBASE * curvature)

        center, indices = env.track.sample(arcs)
        tangent = env.track.tangents[indices]
        normal = tangent[:, ::-1] * [-1., 1.]
        lateral = np.abs(((points - center) * normal).sum(axis=2))
        norm = np.maximum(np.linalg.norm(first, axis=2), .1)
        sin_heading = np.abs((first * normal).sum(axis=2)) / norm
        footprint = .225 * np.sqrt(np.maximum(0., 1. - sin_heading**2)) + .31 * sin_heading
        legal = np.all(lateral + footprint <= env.track.half_width - .025, axis=1)
        legal &= np.all((first * tangent).sum(axis=2) > .25, axis=1)
        legal &= np.all(np.abs(wheel) <= MAX_STEER + .035, axis=1)

        distance = np.linalg.norm(np.diff(points, axis=1), axis=2)
        corner = np.maximum(np.abs(curvature), 1e-4)
        cap = np.minimum(MAX_GROUND_SPEED**2, self.lateral_limit / corner)
        # The speed envelope at the join includes braking for the next bends.
        finish_node = min(int(arcs[-1] / self.spacing), self.points.shape[1] - 1)
        blend = (arcs[-1] - finish_node * self.spacing) / self.spacing
        target = self.speeds[:, finish_node] + blend * (self.speeds[:, finish_node + 1] - self.speeds[:, finish_node])
        cap[:, -1] = np.minimum(cap[:, -1], target**2)
        remaining_grip = np.sqrt(1. - np.minimum(.96, .87 * cap * corner / self.lateral_limit)**2)
        braking = np.interp(np.sqrt(cap), self.pace.brake[:, 0], self.pace.brake[:, 1]) * remaining_grip
        # Also leave time for the steering servo to change direction.
        steer_cap = (STEER_RATE * distance / np.maximum(np.abs(np.diff(wheel, axis=1)), .01))**2
        cap[:, :-1] = np.minimum(cap[:, :-1], steer_cap)
        for i in range(distance.shape[1] - 1, -1, -1):
            cap[:, i] = np.minimum(cap[:, i], cap[:, i + 1] + 2. * distance[:, i] * braking[:, i + 1])
        legal &= speed**2 <= cap[:, 0] + 1.25

        weights = (np.ones(distance.shape[1]) if self.step_weights is None else
                   np.interp((arcs[:-1] + arcs[1:]) / 2., self.midpoints,
                             self.step_weights, period=self.finish))
        elapsed = np.zeros(len(self.lines))
        current = np.full(len(self.lines), speed)
        for i, (ds, weight) in enumerate(zip(distance.T, weights)):
            drive = np.interp(current, self.pace.drive[:, 0], self.pace.drive[:, 1])
            reserve = np.sqrt(1. - np.minimum(.96, .87 * current**2 * corner[:, i] / self.lateral_limit)**2)
            brake = np.interp(current, self.pace.brake[:, 0], self.pace.brake[:, 1])
            squared = np.clip(cap[:, i + 1], current**2 - 2. * ds * brake * reserve,
                              current**2 + 2. * ds * drive * reserve)
            following = np.sqrt(np.maximum(0., squared))
            legal &= following**2 <= cap[:, i + 1] + 1.25
            elapsed += weight * (2. * ds / np.maximum(.15, current + following))
            current = following

        index = np.searchsorted(self.speed_grid, current).clip(1, len(self.speed_grid) - 1)
        left_speed, right_speed = self.speed_grid[index - 1], self.speed_grid[index]
        fraction = np.clip((current - left_speed) / (right_speed - left_speed), 0., 1.)
        at_speed = (self.suffix[self.rows, :, index - 1] * (1. - fraction[:, None])
                    + self.suffix[self.rows, :, index] * fraction[:, None])
        suffix = (1. - blend) * at_speed[:, finish_node] + blend * at_speed[:, finish_node + 1]
        return np.where(legal, elapsed + suffix, np.inf)

    def potential(self, env, obs):
        gate = env.track.gates[min(env.next_gate, len(env.track.gates) - 1)]
        arc = float(min(env.arc, gate))
        old_time = -self.fallback.potential(env, obs)
        if self.finish - arc <= .8:
            return -old_time
        minimum = self.remaining(env, arc).min()
        # Bound the approximation where no short join can be certified.
        estimate = min(minimum, old_time + .75)
        weight = min(1., (self.finish - arc - .8) / .8)
        return -(weight * estimate + (1. - weight) * old_time)
