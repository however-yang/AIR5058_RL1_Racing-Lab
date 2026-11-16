"""Timing lines used for matched race segments."""

import numpy as np


SPLIT_SECTORS = (
    ("spiral_back_straight", 74., 116.),
    ("spiral_back_straight_through_corner", 74., 96.),
    ("spiral_back_straight_after_corner", 96., 116.),
)


class SectorClock:
    def __init__(self, track, sectors):
        self.sectors = tuple(sectors)
        self.arcs = np.unique([arc for _, entry, exit in self.sectors for arc in (entry, exit)])
        self.points, indices = track.sample(self.arcs)
        self.tangents = track.tangents[indices]
        self.width = track.half_width
        self.times = []

    def record(self, before, after, time_before, time_after, arc):
        i = len(self.times)
        if i >= len(self.arcs) or abs(arc - self.arcs[i]) > 1.5:
            return
        center, tangent = self.points[i], self.tangents[i]
        first = float((before - center) @ tangent)
        last = float((after - center) @ tangent)
        if first < 0. <= last:
            fraction = -first / (last - first)
            crossing = before + fraction * (after - before)
            if np.linalg.norm(crossing - center) <= self.width + .02:
                self.times.append(time_before + fraction * (time_after - time_before))
                return float(self.arcs[i]), fraction

    def results(self):
        crossings = dict(zip(self.arcs, self.times))
        return {name: crossings[exit] - crossings[entry] for name, entry, exit in self.sectors
                if entry in crossings and exit in crossings}
