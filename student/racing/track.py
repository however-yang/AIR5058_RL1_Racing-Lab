"""Sampled closed centerline, arc length and track visuals."""

import math
from xml.etree import ElementTree as ET

import numpy as np

from .visuals import dress_car, dress_world, prop


class Track:
    half_width = 1.05

    def __init__(self, layout="shanghai"):
        self.layout = layout
        # The periodic cubic spline rounds the corners of the control polygon.
        control = np.array([
            # Pit straight and the tightening first spiral.
            [-4.75, -.15], [-3.05, 4.70], [-1.35, 9.30], [.25, 11.65],
            [2.85, 12.40], [5.35, 11.65], [6.20, 9.95], [5.55, 8.15],
            [4.02, 7.66], [2.35, 8.10], [1.10, 7.38], [.96, 5.82], [2.23, 4.42],
            # The top short straight, its hairpin, and the infield esses.
            [4.25, 4.35], [7.25, 5.20], [10.20, 6.05], [12.70, 5.77],
            [14.25, 4.53], [14.02, 2.74], [12.15, 1.73], [9.70, 1.80], [7.20, 2.05],
            [5.40, 1.65], [4.75, .20], [5.38, -1.12], [6.35, -2.55],
            [5.80, -3.94], [4.25, -4.22], [2.55, -3.65], [1.35, -3.50],
            [.05, -3.85], [-.62, -5.15], [.05, -6.45], [1.35, -6.94], [2.75, -6.96], [5.10, -6.96], [7.30, -6.96],
            # Opening spiral onto the long back straight.
            [9.40, -6.92], [10.48, -5.94], [10.40, -4.20], [10.80, -2.72],
            [12.35, -2.10], [14.10, -2.85], [15.45, -4.60], [15.72, -6.85],
            [14.60, -9.12], [12.20, -10.55], [8.75, -10.95], [2.00, -10.95], [-5.50, -10.95],
            # Back-straight hairpin and final corner to the pits.
            [-11.75, -10.95], [-14.60, -10.75], [-15.63, -9.32], [-14.94, -7.85], [-13.20, -6.80],
            [-11.70, -6.65], [-8.85, -6.65], [-7.05, -6.35], [-5.95, -4.65],
        ])
        if layout == "spa":
            control = np.array([
                # Pit straight, La Source and the Eau Rouge–Raidillon sweep.
                [-10.4, -7.8], [-13.7, -10.15], [-16.0, -10.95],
                [-17.2, -9.75], [-16.45, -7.5], [-15.1, -5.35],
                [-13.5, -3.4], [-11.4, -1.55], [-9.75, .85], [-7.6, 2.0],
                [-5.4, 3.9], [-2.4, 6.05], [2.5, 8.4], [7.35, 10.7],
                # Kemmel, Les Combes and the wide descent through Bruxelles.
                [9.8, 11.5], [11.5, 10.55], [13.55, 10.75], [15.25, 9.2],
                [16.9, 7.3], [17.1, 5.65], [15.6, 4.85], [13.85, 6.10],
                [12.0, 6.65], [8.9, 4.9], [5.7, 3.65], [4.1, 2.2],
                # Pouhon, Fagnes and the long arc back from Stavelot.
                [4.05, .15], [5.35, -1.50], [8.0, -2.35], [10.2, -3.15],
                [11.0, -4.5], [11.0, -6.25], [12.85, -7.6], [15.0, -8.65],
                [15.65, -10.2], [14.55, -12.15], [12.75, -13.05], [10.2, -12.45],
                [7.5, -11.05], [5.25, -9.25], [3.4, -7.05], [1.4, -5.6],
                # Blanchimont and a softened Bus Stop.
                [-.5, -5.0], [-2.6, -5.55], [-4.6, -6.45], [-6.2, -6.5],
                [-7.65, -5.95], [-8.95, -6.75],
            ])
        elif layout != "shanghai":
            raise ValueError(f"Unknown circuit layout: {layout}")

        u = np.linspace(0., 1., 32, endpoint=False)[:, None, None]
        weights = ((1. - u)**3, 3. * u**3 - 6. * u**2 + 4.,
                   -3. * u**3 + 3. * u**2 + 3. * u + 1., u**3)
        smooth = sum(w * np.roll(control, shift, axis=0)[None, :, :]
                     for w, shift in zip(weights, (1, 0, -1, -2))) / 6.
        dense = smooth.transpose(1, 0, 2).reshape(-1, 2)
        closed = np.vstack((dense, dense[:1]))
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        uniform = np.linspace(0., distance[-1], 640, endpoint=False)
        self.points = np.column_stack([np.interp(uniform, distance, axis)
                                       for axis in closed.T])
        self.edges = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.edges, axis=1)
        self.tangents = self.edges / self.lengths[:, None]
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.length = float(self.arc[-1])
        self.gates = np.linspace(self.length / 8, self.length, 8)
        self.neighbors = (np.arange(len(self.points))[:, None] + np.arange(-13, 14)) % len(self.points)
        self.local_points = self.points[self.neighbors]
        self.local_tangents = self.tangents[self.neighbors]
        self.local_lengths = self.lengths[self.neighbors]

    def sample(self, arc):
        distance = np.asarray(arc) % self.length
        # A tiny negative arc can round to length after wrapping.
        distance = np.where(distance < self.length, distance, 0.)
        idx = np.searchsorted(self.arc, distance, side="right") - 1
        return self.points[idx] + (distance - self.arc[idx])[..., None] * self.tangents[idx], idx

    def project(self, positions, near):
        """Project a batch of points near the last known road segment."""
        tangents = self.local_tangents[near]
        offsets = positions[:, None, :] - self.local_points[near]
        along = np.clip(np.einsum("mni,ni->mn", offsets, tangents),
                        0., self.local_lengths[near])
        residual = offsets - along[..., None] * tangents
        squared = np.einsum("mni,mni->mn", residual, residual)
        rows = np.arange(len(positions))
        best = squared.argmin(axis=1)
        segments = self.neighbors[near, best]
        return segments, self.arc[segments] + along[rows, best], squared[rows, best]

    def add_visuals(self, model):
        assets = ET.SubElement(model, "asset")
        world = model.find("worldbody")
        dress_car(assets, world)
        dress_world(assets, world, self)
        ET.SubElement(model, "statistic", center="0 0 0", extent="24")
        ET.SubElement(model.find("visual"), "map", znear=".004", zfar="8")

        def strip(a, b, half_width, z, color):
            x, y = (a + b) / 2
            dx, dy = b - a
            ET.SubElement(world, "geom", type="box", pos=f"{x} {y} {z}",
                          euler=f"0 0 {math.atan2(dy, dx)}",
                          size=f"{math.hypot(dx, dy) / 2 + .02} {half_width} .001",
                          rgba=color, contype="0", conaffinity="0")

        normals = self.tangents[:, ::-1] * [-1, 1]
        def band(name, left, right, height, color):
            vertices, faces = [], []
            for a, b in zip(self.points + left * normals, self.points + right * normals):
                vertices.extend([(*a, height), (*b, height), (*a, height - .001), (*b, height - .001)])
            for i in range(len(self.points)):
                a, b, c, d = range(4 * i, 4 * i + 4)
                j = 4 * ((i + 1) % len(self.points))
                e, f, g, h = range(j, j + 4)
                faces.extend([(a, b, f), (a, f, e), (c, g, h), (c, h, d),
                              (a, e, g), (a, g, c), (b, d, h), (b, h, f)])
            ET.SubElement(assets, "mesh", name=name,
                          vertex=" ".join(str(x) for vertex in vertices for x in vertex),
                          face=" ".join(str(x) for face in faces for x in face))
            ET.SubElement(world, "geom", type="mesh", mesh=name, rgba=color,
                          contype="0", conaffinity="0")

        w = self.half_width
        band("shoulder", w + .20, -w - .20, .002, ".31 .34 .34 1")
        band("asphalt", w, -w, .004, ".18 .21 .24 1")
        band("inner_line", w + .022, w - .022, .006, ".88 .90 .92 1")
        band("outer_line", -w + .022, -w - .022, .006, ".88 .90 .92 1")

        for gate in self.gates[:-1]:
            center, idx = self.sample(gate)
            normal = normals[idx]
            for side in (-1, 1):
                strip(center + side * .77 * normal, center + side * w * normal,
                      .025, .009, ".59 .64 .65 1")

        # Paint alternating curb tiles on the inside of bends.
        tile_length = .28
        for tile, s in enumerate(np.arange(.5, self.length - .5, tile_length)):
            center, idx = self.sample(s)
            _, before = self.sample(s - .45)
            _, after = self.sample(s + .45)
            incoming, outgoing = self.tangents[[before, after]]
            turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if abs(turn) < .13:
                continue
            edge = center + np.sign(turn) * (w + .10) * normals[idx]
            yaw = math.atan2(self.tangents[idx, 1], self.tangents[idx, 0])
            color = ".86 .14 .12 1" if tile % 2 else ".96 .95 .88 1"
            inner_scale = max(.38, 1. - abs(turn) * (w + .10) / .90)
            half_tile = .42 * tile_length * inner_scale
            prop(world, "box", (*edge, .010), (half_tile, .068, .009), color,
                 euler=f"0 0 {yaw}")

        center, idx = self.sample(0.)
        forward, normal = self.tangents[idx], normals[idx]
        yaw = math.atan2(forward[1], forward[0])
        rotation = f"0 0 {yaw}"
        cell = 2 * w / 14
        for row in range(2):
            for column in range(14):
                tile = center + (row - .5) * cell * forward + (column - 6.5) * cell * normal
                color = ".95 .95 .91 1" if (row + column) % 2 == 0 else ".038 .045 .05 1"
                prop(world, "box", (*tile, .010), (cell / 2, cell / 2, .003),
                     color, euler=rotation)
        for side in (-1, 1):
            pole = center + side * (w + .25) * normal
            prop(world, "box", (*pole, .61), (.045, .045, .61), ".15 .21 .24 1",
                 euler=rotation)
        prop(world, "box", (*center, 1.23), (.075, w + .30, .080), ".15 .21 .24 1",
             euler=rotation)
        prop(world, "box", (*center, 1.313), (.074, w + .30, .008), ".98 .30 .07 1",
             euler=rotation)
