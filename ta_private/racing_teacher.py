"""Shanghai-only offline teacher for racing-line and speed demonstrations."""

import math

import numpy as np
import torch

from racing.vehicle import MAX_SPEED, MAX_STEER, WHEELBASE


class RacingTeacher:
    def __init__(self, track, lateral_budget=5.9):
        self.length = track.length
        count = round(self.length / .35)
        arc = np.linspace(0., self.length, count, endpoint=False)
        center, _ = track.sample(arc)
        tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = tangent[:, ::-1] * [-1., 1.]
        center_t = torch.as_tensor(center, dtype=torch.float32)
        normal_t = torch.as_tensor(normal, dtype=torch.float32)
        offset = torch.zeros(count, requires_grad=True)
        optimizer = torch.optim.Adam([offset], lr=.035)
        width = track.half_width - .40
        for _ in range(500):
            points = center_t + (width * offset.tanh())[:, None] * normal_t
            edge = points.roll(-1, 0) - points
            previous = edge.roll(1, 0)
            length = edge.norm(dim=1)
            cross = previous[:, 0] * edge[:, 1] - previous[:, 1] * edge[:, 0]
            curvature = 2. * cross / (length * length.roll(1) * (previous + edge).norm(dim=1)).clamp_min(.005)
            loss = curvature.square().mean() + .018 * length.mean() + .003 * offset.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.offsets = width * offset.detach().tanh().numpy()
        self.points = center + self.offsets[:, None] * normal
        edges = np.roll(self.points, -1, axis=0) - self.points
        prev = np.roll(edges, 1, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        cross = prev[:, 0] * edges[:, 1] - prev[:, 1] * edges[:, 0]
        curvature = 2. * np.abs(cross) / (lengths * np.roll(lengths, 1) * np.linalg.norm(prev + edges, axis=1))
        self.curvature = (np.roll(curvature, 1) + 2. * curvature + np.roll(curvature, -1)) / 4.
        self.speeds = np.minimum(4.55, np.sqrt(lateral_budget / np.maximum(self.curvature, .01)))
        for _ in range(3):
            for i in range(count - 1, -1, -1):
                j = (i + 1) % count
                self.speeds[i] = min(self.speeds[i], math.sqrt(self.speeds[j] ** 2 + 2. * 2.45 * lengths[i]))
            for i in range(count):
                j = (i + 1) % count
                self.speeds[j] = min(self.speeds[j], math.sqrt(self.speeds[i] ** 2 + 2. * 2.15 * lengths[i]))

        self.arc = np.r_[arc, self.length]
        self.closed_points = np.vstack([self.points, self.points[:1]])
        self.closed_speeds = np.r_[self.speeds, self.speeds[0]]

    def action(self, env):
        speed = max(0., math.hypot(*env.data.qvel[:2]))
        yaw = env._yaw()
        lookahead = .64 + .30 * speed
        arc = (env.arc + lookahead) % self.length
        target = np.array([np.interp(arc, self.arc, self.closed_points[:, i]) for i in range(2)])
        displacement = target - env.data.qpos[:2]
        left = displacement @ [-math.sin(yaw), math.cos(yaw)]
        steer = math.atan2(2. * WHEELBASE * left, displacement @ displacement)
        command = np.interp((env.arc + .45) % self.length, self.arc, self.closed_speeds) + .14
        return np.clip([steer / MAX_STEER, 2. * command / MAX_SPEED - 1.], -1., 1.).astype(np.float32)
