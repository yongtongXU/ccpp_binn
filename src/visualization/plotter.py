from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _draw_map(ax, grid_map):
    display = np.ones((*grid_map.grid.shape, 3))
    display[grid_map.grid == 1] = [0.15, 0.15, 0.15]
    display[grid_map.grid == 2] = [0.72, 0.72, 0.72]
    ax.imshow(display, origin="lower", extent=[0, grid_map.width, 0, grid_map.height])
    ax.set_aspect("equal")


def plot_trajectory(result, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_map(ax, result.grid_map)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(result.paths_by_usv))))
    for idx, (uid, path) in enumerate(result.paths_by_usv.items()):
        if not path:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, lw=1.5, color=colors[idx], label=uid)
        ax.scatter(xs[0], ys[0], s=45, marker="o", color=colors[idx])
        ax.scatter(xs[-1], ys[-1], s=45, marker="x", color=colors[idx])
    theta = np.deg2rad(result.best_scan_angle)
    ax.arrow(4, result.grid_map.height - 5, 6 * np.cos(theta), 6 * np.sin(theta), width=0.25, color="crimson")
    ax.set_title(f"Trajectory, scan angle {result.best_scan_angle} deg")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_segments(result, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_map(ax, result.grid_map)
    cmap = plt.cm.viridis
    max_line = max([s.line_index for s in result.segments], default=1)
    for seg in result.segments:
        pts = seg.sampled_points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=1.3, color=cmap(seg.line_index / max(1, max_line)))
        ax.scatter([seg.start_point[0], seg.end_point[0]], [seg.start_point[1], seg.end_point[1]], s=8, color="crimson")
        mid = pts[len(pts) // 2]
        if seg.id % 5 == 0:
            ax.text(mid[0], mid[1], str(seg.id), fontsize=6)
    ax.set_title("Coverage segments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_coverage(result, out_path: str | Path) -> None:
    grid = np.zeros((*result.grid_map.grid.shape, 3))
    grid[result.grid_map.grid == 0] = [0.93, 0.93, 0.93]
    grid[result.coverage_state.covered_map] = [0.25, 0.65, 0.35]
    grid[result.coverage_state.uncovered_mask()] = [0.9, 0.3, 0.25]
    grid[result.grid_map.grid == 1] = [0.05, 0.05, 0.05]
    grid[result.grid_map.grid == 2] = [0.55, 0.55, 0.55]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, origin="lower", extent=[0, result.grid_map.width, 0, result.grid_map.height])
    ax.set_title("Coverage map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gbnn(result, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if result.gbnn_activity is None:
        data = np.zeros(result.grid_map.grid.shape)
    else:
        data = result.gbnn_activity
    im = ax.imshow(data, origin="lower", cmap="coolwarm", extent=[0, result.grid_map.width, 0, result.grid_map.height])
    ax.set_title("GBNN coverage demand activity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metrics_curve(result, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["coverage", "repeat", "residual before", "residual after"]
    values = [
        result.metrics["coverage_rate"],
        result.metrics["repeated_coverage_rate"],
        result.metrics["residual_before"],
        result.metrics["residual_after"],
    ]
    ax.bar(labels, values, color=["#2d7f5e", "#c58f2b", "#666666", "#8d3f3f"])
    ax.set_title("Planning metrics")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
