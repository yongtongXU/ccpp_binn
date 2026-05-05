from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


MODE_COLORS = {
    "main_sweep": "#1f77b4",
    "connection": "#f28e2b",
    "recovery": "#d62728",
    "idle": "#777777",
}


def save_coverage_gif(result, out_path: str | Path, step: int = 1, duration: float = 0.08) -> None:
    """Save an execution animation with progressive paths and covered cells.

    `step=1` renders every stored path sample. Larger values skip frames but still
    update coverage through the skipped points, so the animation remains faithful.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _frame_indices(result, max(1, step))
    if not frame_indices:
        return

    usv_by_id = {u.id: u for u in result.usvs}
    visit_count = np.zeros(result.grid_map.grid.shape, dtype=np.uint16)
    covered = np.zeros(result.grid_map.grid.shape, dtype=bool)
    progress = {uid: 0 for uid in result.paths_by_usv}
    frames = []

    for frame_no, limit in enumerate(frame_indices, start=1):
        for uid, path in result.paths_by_usv.items():
            radius = usv_by_id[uid].coverage_radius
            while progress[uid] < min(limit, len(path)):
                _mark_covered(result.grid_map, covered, visit_count, path[progress[uid]], radius)
                progress[uid] += 1
        frames.append(_render_frame(result, covered, visit_count, limit, frame_no, len(frame_indices)))

    imageio.mimsave(out_path, frames, duration=duration)


def _frame_indices(result, step: int) -> list[int]:
    longest = max((len(p) for p in result.paths_by_usv.values()), default=0)
    if longest == 0:
        return []
    indices = list(range(1, longest + 1, step))
    if indices[-1] != longest:
        indices.append(longest)
    return indices


def _mark_covered(grid_map, covered: np.ndarray, visit_count: np.ndarray, point, coverage_radius: float) -> None:
    center = grid_map.world_to_grid(point)
    radius_cells = max(0, int(round(coverage_radius / grid_map.resolution)))
    r0, c0 = center
    for r in range(max(0, r0 - radius_cells), min(grid_map.height, r0 + radius_cells + 1)):
        for c in range(max(0, c0 - radius_cells), min(grid_map.width, c0 + radius_cells + 1)):
            if grid_map.grid[r, c] != 0:
                continue
            if (r - r0) ** 2 + (c - c0) ** 2 <= radius_cells**2:
                covered[r, c] = True
                visit_count[r, c] += 1


def _render_frame(result, covered: np.ndarray, visit_count: np.ndarray, limit: int, frame_no: int, total_frames: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_coverage_rgb(result.grid_map, covered, visit_count), origin="lower", extent=[0, result.grid_map.width, 0, result.grid_map.height])

    for uid, path in result.paths_by_usv.items():
        modes = result.modes_by_usv.get(uid, [])
        shown = path[: min(limit, len(path))]
        if not shown:
            continue
        _plot_path_by_mode(ax, shown, modes[: len(shown)])
        cur = shown[-1]
        ax.scatter(cur[0], cur[1], s=42, marker="o", edgecolors="white", linewidths=0.8, label=f"{uid} current")
        ax.text(cur[0] + 0.8, cur[1] + 0.8, uid, fontsize=7, color="black")

    free = result.grid_map.grid == 0
    coverage_rate = float((covered & free).sum() / max(1, free.sum()))
    ax.set_title(
        f"{result.metrics['scenario']} | step {limit} | frame {frame_no}/{total_frames} | coverage {coverage_rate:.3f}"
    )
    ax.set_xlim(0, result.grid_map.width)
    ax.set_ylim(0, result.grid_map.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _legend(ax)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return image


def _coverage_rgb(grid_map, covered: np.ndarray, visit_count: np.ndarray) -> np.ndarray:
    rgb = np.ones((*grid_map.grid.shape, 3), dtype=float)
    rgb[grid_map.grid == 0] = [0.94, 0.94, 0.92]
    rgb[covered] = [0.68, 0.86, 0.70]
    rgb[(visit_count > 6) & (grid_map.grid == 0)] = [0.50, 0.75, 0.56]
    rgb[grid_map.grid == 1] = [0.12, 0.12, 0.12]
    rgb[grid_map.grid == 2] = [0.65, 0.65, 0.65]
    return rgb


def _plot_path_by_mode(ax, points, modes) -> None:
    start = 0
    while start < len(points) - 1:
        mode = modes[start] if start < len(modes) else "idle"
        end = start + 1
        while end < len(points) and (modes[end] if end < len(modes) else mode) == mode:
            end += 1
        chunk = points[start:end]
        if len(chunk) >= 2:
            ax.plot([p[0] for p in chunk], [p[1] for p in chunk], color=MODE_COLORS.get(mode, "#777777"), lw=1.8)
        start = max(end - 1, start + 1)


def _legend(ax) -> None:
    handles = [
        plt.Line2D([0], [0], color=MODE_COLORS["main_sweep"], lw=2, label="main sweep"),
        plt.Line2D([0], [0], color=MODE_COLORS["connection"], lw=2, label="connection"),
        plt.Line2D([0], [0], color=MODE_COLORS["recovery"], lw=2, label="recovery"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
