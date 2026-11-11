"""A bounded, periodic minimum-time line and a local tracking reference."""

import math

import numpy as np
import torch

from racing.vehicle import MAX_SPEED, MAX_STEER, WHEELBASE


MAX_GROUND_SPEED = 4.58
DRIVE_ACCEL = 2.15
BRAKE_ACCEL = 2.6
LOOKAHEAD = np.array([.6, 1.3, 2.2, 3.5, 5.])


def geometry(points):
    edges = points.roll(-1, 0) - points
    prior = edges.roll(1, 0)
    length = edges.norm(dim=1).clamp_min(.02)
    cross = prior[:, 0] * edges[:, 1] - prior[:, 1] * edges[:, 0]
    curvature = 2. * cross / (length * length.roll(1) * (prior + edges).norm(dim=1)).clamp_min(.002)
    curvature = (curvature.roll(1) + 2. * curvature + curvature.roll(-1)) / 4.
    return length, curvature


class MeasuredPace:
    """Piecewise linear longitudinal capability measured in MuJoCo."""

    def __init__(self, drive, brake):
        self.drive = np.asarray(drive, dtype=np.float32)
        self.brake = np.asarray(brake, dtype=np.float32)
        self.curves = tuple(torch.as_tensor(row) for row in (self.drive, self.brake))

    @staticmethod
    def interp(speed, knots):
        interval = torch.searchsorted(knots[:, 0].contiguous(), speed).clamp(1, len(knots) - 1)
        left, right = knots[interval - 1], knots[interval]
        weight = ((speed - left[:, 0]) / (right[:, 0] - left[:, 0])).clamp(0., 1.)
        return left[:, 1] + weight * (right[:, 1] - left[:, 1])

    def limits(self, speed):
        return tuple(self.interp(speed, curve) for curve in self.curves)


def speed_envelope(length, curvature, lateral_limit, pace=None):
    """Propagate corner limits in both directions with finite braking and drive."""
    absolute = (curvature.square() + 1e-5).sqrt()
    squared = (lateral_limit / absolute).clamp_max(MAX_GROUND_SPEED ** 2)
    for _ in range(12):
        # Reserve longitudinal grip in bends, using the speed at the constrained point.
        remaining = (1. - (.87 * squared * absolute / lateral_limit).clamp_max(.98).square()).sqrt()
        drive, brake = (DRIVE_ACCEL, BRAKE_ACCEL) if pace is None else pace.limits(squared.sqrt())
        braking = BRAKE_ACCEL * remaining.roll(-1) if pace is None else (brake * remaining).roll(-1)
        ahead = squared.roll(-1) + 2. * length * braking
        behind = (squared + 2. * drive * length * remaining).roll(1)
        squared = torch.minimum(squared, torch.minimum(ahead, behind))
    return squared.sqrt()


class TimeLine:
    def __init__(self, track, lateral_limit=7.1, iterations=400, pace=None, brake_preview=.52):
        self.length = track.length
        self.lateral_limit = lateral_limit
        self.pace = pace
        self.brake_preview = brake_preview
        self.count = round(track.length / .42)
        self.step = track.length / self.count
        samples = np.arange(self.count) * self.step
        center, _ = track.sample(samples)
        tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = tangent[:, ::-1] * [-1., 1.]
        center = torch.tensor(center, dtype=torch.float32)
        normal = torch.tensor(normal, dtype=torch.float32)

        # A periodic cubic spline shares offsets across adjacent corners.
        controls = round(track.length / 1.75)
        index = np.arange(self.count) * controls / self.count
        segment = np.floor(index).astype(int)
        u = index - segment
        basis = torch.zeros((self.count, controls))
        weights = ((1. - u) ** 3, 3. * u ** 3 - 6. * u ** 2 + 4.,
                   -3. * u ** 3 + 3. * u ** 2 + 3. * u + 1., u ** 3)
        for shift, weight in zip((-1, 0, 1, 2), weights):
            basis[torch.arange(self.count), (segment + shift) % controls] = torch.tensor(weight / 6., dtype=torch.float32)
        free = torch.zeros(controls, requires_grad=True)
        optimizer = torch.optim.Adam([free], lr=.035)
        width = track.half_width - .40

        def line():
            return center + (basis @ (width * free.tanh()))[:, None] * normal

        for _ in range(iterations):
            points = line()
            length, curvature = geometry(points)
            speed = speed_envelope(length, curvature, lateral_limit, pace)
            seconds = (2. * length / (speed + speed.roll(-1))).sum()
            smoothness = (free - 2. * free.roll(1) + free.roll(2)).square().mean()
            loss = seconds + .35 * curvature.square().mean() + .17 * smoothness
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            self.points = line().numpy()
            length, curvature = geometry(line())
            self.speeds = speed_envelope(length, curvature, lateral_limit, pace).numpy()
            self.estimated_time = float((2. * length / (torch.tensor(self.speeds) + torch.tensor(self.speeds).roll(-1))).sum())
        self.arcs = np.r_[samples, self.length]
        self.closed_points = np.vstack((self.points, self.points[:1]))
        self.closed_speeds = np.r_[self.speeds, self.speeds[0]]

    def save(self, path):
        curves = {} if self.pace is None else {"drive": self.pace.drive, "brake": self.pace.brake}
        np.savez_compressed(path, points=self.points, speeds=self.speeds, length=self.length,
                            lateral_limit=self.lateral_limit, estimated_time=self.estimated_time,
                            brake_preview=self.brake_preview, **curves)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            line = cls.__new__(cls)
            line.length = float(data["length"])
            line.lateral_limit = float(data["lateral_limit"])
            line.estimated_time = float(data["estimated_time"])
            line.points = data["points"]
            line.speeds = data["speeds"]
            line.brake_preview = float(data["brake_preview"]) if "brake_preview" in data else .52
            line.pace = MeasuredPace(data["drive"], data["brake"]) if "drive" in data else None
        line.count = len(line.points)
        line.step = line.length / line.count
        line.arcs = np.linspace(0., line.length, line.count + 1)
        line.closed_points = np.vstack((line.points, line.points[:1]))
        line.closed_speeds = np.r_[line.speeds, line.speeds[0]]
        return line

    def reference(self, env, lane_fraction=0., lane_limit=.65):
        velocity = math.hypot(*env.data.qvel[:2])
        yaw = env._yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        position = env.data.qpos[:2]
        look = .68 + .28 * velocity
        arc = (env.arc + np.r_[look, LOOKAHEAD]) % self.length
        targets = np.column_stack([np.interp(arc, self.arcs, xy) for xy in self.closed_points.T])
        if lane_fraction:
            center, _ = env.track.sample(arc)
            before, _ = env.track.sample(arc - .35)
            after, _ = env.track.sample(arc + .35)
            tangent = after - before
            normal = tangent[:, ::-1] * [-1., 1.] / np.linalg.norm(tangent, axis=1)[:, None]
            edge = center + math.copysign(lane_limit, lane_fraction) * normal
            targets += abs(lane_fraction) * (edge - targets)
        local = (targets - position) @ np.array([[c, -s], [s, c]])
        aim = local[0]
        steer = math.atan2(2. * WHEELBASE * aim[1], aim @ aim)
        speeds = np.interp((env.arc + np.array([self.brake_preview, 1.3, 2.4, 3.8])) % self.length,
                           self.arcs, self.closed_speeds)
        target_speed = min(MAX_SPEED, float(speeds[0]) + .13)
        action = np.clip([steer / MAX_STEER, 2. * target_speed / MAX_SPEED - 1.], -1., 1.)
        context = np.r_[local[1:].ravel() / 5., speeds / MAX_SPEED, action].astype(np.float32)
        return action.astype(np.float32), context
