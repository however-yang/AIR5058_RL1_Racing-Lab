"""Visualize the measured speed and the exact scored bend samples."""

import numpy as np


def smooth_bends(values):
    """Centered distance average, keeping straight and unvisited gaps blank."""
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    cuts = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
    for start, end in cuts:
        segment = values[start:end]
        weights = np.ones(min(5, len(segment)))
        result[start:end] = (np.convolve(segment, weights, mode="same")
                             / np.convolve(np.ones_like(segment), weights, mode="same"))
    return result


def plot_comparison(record, seed, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    laps = [(label, color, next(lap for lap in record[key]["episodes"] if lap["seed"] == seed))
            for key, label, color in (("baseline", "Centerline baseline", "#285273"),
                                      ("student", "Submitted policy", "#b34026")) if key in record]
    profile = laps[0][2]["profile"]
    arc = np.asarray(profile["arc_m"])
    bends = np.asarray(profile["bend"])
    straight = np.asarray(profile["straight"])
    limit = profile["max_speed_mps"]
    fig, (speed_axis, grip_axis) = plt.subplots(
        2, 1, figsize=(13, 6.6), sharex=True, layout="constrained")

    speed_axis.fill_between(arc, 0, 1, where=straight, transform=speed_axis.get_xaxis_transform(),
                            step="mid", alpha=.38, color="#d6ece4", label="Straight samples")
    grip_axis.fill_between(arc, 0, 1, where=bends, transform=grip_axis.get_xaxis_transform(),
                            step="mid", alpha=.4, color="#dce7f3", label="Scored bends")
    for label, color, ep in laps:
        values = ep["profile"]
        note = f'{ep["time_s"]:.2f} s' if ep["status"] == "finished" else ep["status"].replace("_", " ")
        speed_axis.plot(arc, np.asarray(values["speed_mps"], dtype=float), color=color,
                        lw=1.65, label=f"{label} · {note}")
        grip = np.full(arc.shape, np.nan)
        grip[bends] = np.asarray(values["effective_grip"], dtype=float)
        grip_axis.plot(arc, grip, color=color, lw=.65, marker=".", ms=2., alpha=.24)
        grip_axis.plot(arc, smooth_bends(grip), color=color, lw=1.55, label=f"{label} · 0.5 m average")
        if label == "Submitted policy" or len(laps) == 1:
            raw = np.full(arc.shape, np.nan)
            raw[bends] = np.asarray(values["inward_grip_before_slip_discount"], dtype=float)
            grip_axis.plot(arc, smooth_bends(raw), color=color, lw=1., ls=":",
                           alpha=.8, label="Before slip discount · 0.5 m average")
        if label == "Submitted policy":
            speed_axis.plot(arc, np.asarray(values["target_mps"], dtype=float), color=color,
                            lw=.9, ls=":", alpha=.72, label="Actuator speed target")

    speed_axis.axhline(limit, color="#42515d", ls="--", lw=1, label=f"Configured limit · {limit:.2f} m/s")
    speed_axis.axhline(.95 * limit, color="#657b6c", ls=":", lw=1, label="95% of configured limit")
    grip_axis.axhline(.90, color="#42515d", ls="--", lw=.9, label="90% effective grip")
    speed_axis.set(ylabel="Speed (m/s)", ylim=(0, limit * 1.07))
    grip_axis.set(xlabel="Centerline distance (m)", ylabel="Effective inward grip", ylim=(0, 1.06),
                  xlim=(0, arc[-1]))
    grip_axis.yaxis.set_major_formatter(PercentFormatter(1))
    for axis in (speed_axis, grip_axis):
        axis.grid(alpha=.22)
        axis.legend(loc="lower left", bbox_to_anchor=(0., 1.005), borderaxespad=0,
                    ncol=3, frameon=False, fontsize=8)

    fig.suptitle(f'{record["layout"].title()} · {record["surface"]} · paired seed {seed}',
                 fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
