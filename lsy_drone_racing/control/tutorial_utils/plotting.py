"""Plotting for tutorial flight logs.

Called automatically at the end of every episode by the tutorial controller, which writes a PNG
next to the log in <repo>/logs/. Can also be run by hand:

    python -m lsy_drone_racing.control.tutorial_utils.plotting tutorial_log.npz          # window
    python -m lsy_drone_racing.control.tutorial_utils.plotting tutorial_log.npz --save   # PNG

Panels: x/y/z vs. target, top-down path, distance to target, commanded vs. actual roll and pitch,
thrust command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.tutorial_utils.history import LOG_DIR

ACTUAL = "#2a78d6"  # blue: what the drone did
COMMAND = "#eb6834"  # orange: what was asked for (target / command)
INK = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _unique_waypoints(target: np.ndarray) -> np.ndarray:
    """Waypoints in the order they were targeted (consecutive duplicates removed)."""
    keep = np.ones(len(target), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(target, axis=0)) > 1e-9, axis=1)
    return target[keep]


def resolve_log(path: str | Path) -> Path:
    """Accept an absolute path, a path from the current folder, or a bare name in <repo>/logs/."""
    path = Path(path)
    return path if path.exists() else LOG_DIR / path


def make_figure(log_path: str | Path, fig: Figure | None = None) -> Figure:
    """Build the analysis figure for one log.

    With ``fig=None`` a standalone Figure is created without any GUI backend, so it is safe to call
    from inside the sim process.
    """
    log_path = resolve_log(log_path)
    d = np.load(log_path)
    t, pos, target, action = d["t"], d["pos"], d["target"], d["action"]
    actual_rpy = R.from_quat(d["quat"]).as_euler("xyz")
    cmd_roll, cmd_pitch, thrust = action[:, 0], action[:, 1], action[:, 3]

    if fig is None:
        fig = Figure(figsize=(12, 11))
    fig.set_layout_engine("constrained")
    fig.set_facecolor(SURFACE)
    axes = fig.subplots(4, 2)
    fig.suptitle(f"Flight log: {log_path.name}  ({t[-1]:.1f} s, {len(t)} steps)",
                 fontsize=12, x=0.01, ha="left")

    for i, (ax, name) in enumerate(zip(axes[:3, 0], "xyz")):
        ax.plot(t, target[:, i], color=COMMAND, linewidth=2, drawstyle="steps-post", label="target")
        ax.plot(t, pos[:, i], color=ACTUAL, linewidth=2, label="actual")
        _style(ax, f"{name} position", "m")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    axes[2, 0].set_xlabel("time (s)", fontsize=9, color=INK)

    ax = axes[0, 1]
    wps = _unique_waypoints(target)
    ax.plot(pos[:, 0], pos[:, 1], color=ACTUAL, linewidth=2, label="path")
    ax.plot(wps[:, 0], wps[:, 1], color=COMMAND, linewidth=1, linestyle="--")
    ax.scatter(wps[:, 0], wps[:, 1], s=60, color=COMMAND, edgecolor=SURFACE, linewidth=2,
               zorder=3, label="waypoints")
    labels: dict[tuple[float, float], list[str]] = {}
    for k, wp in enumerate(wps):
        labels.setdefault((round(wp[0], 3), round(wp[1], 3)), []).append(str(k))
    for (x, y), ks in labels.items():
        ax.annotate(",".join(ks), (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=INK)
    ax.scatter(pos[0, 0], pos[0, 1], s=40, color=INK, marker="s", zorder=3, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, "Top-down path", "y (m)")
    ax.set_xlabel("x (m)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="best")

    ax = axes[1, 1]
    ax.plot(t, np.linalg.norm(target - pos, axis=1), color=ACTUAL, linewidth=2)
    _style(ax, "Distance to current target", "m")

    for ax, cmd, act, name in ((axes[2, 1], cmd_roll, actual_rpy[:, 0], "Roll"),
                               (axes[3, 0], cmd_pitch, actual_rpy[:, 1], "Pitch")):
        ax.plot(t, np.degrees(act), color=ACTUAL, linewidth=2, label="actual")
        ax.plot(t, np.degrees(cmd), color=COMMAND, linewidth=1.5, linestyle="--",
                label="commanded")
        _style(ax, f"{name}: commanded vs actual", "deg")
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.set_xlabel("time (s)", fontsize=9, color=INK)

    ax = axes[3, 1]
    ax.plot(t, thrust, color=COMMAND, linewidth=2)
    _style(ax, "Thrust command", "N")
    ax.set_xlabel("time (s)", fontsize=9, color=INK)
    return fig


def save_plot(log_path: str | Path) -> Path:
    """Write the figure as a PNG next to the log and return its path."""
    log_path = resolve_log(log_path)
    png = log_path.with_suffix(".png")
    make_figure(log_path).savefig(png, dpi=150)
    return png


def main():
    parser = argparse.ArgumentParser(description="Plot a tutorial flight log.")
    parser.add_argument("log", nargs="?", default="tutorial_log.npz",
                        help="Log file (bare names are looked up in <repo>/logs/)")
    parser.add_argument("--save", action="store_true", help="Write a PNG instead of a window")
    args = parser.parse_args()
    if args.save:
        print(f"Saved {save_plot(args.log)}")
        return
    import matplotlib.pyplot as plt  # GUI backend only when a window is wanted

    make_figure(args.log, fig=plt.figure(figsize=(12, 11)))
    plt.show()


if __name__ == "__main__":
    main()