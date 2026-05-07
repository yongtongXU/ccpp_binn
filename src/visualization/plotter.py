from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sbinn_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.core.cell_map import COVERED, OBSTACLE, UNCOVERED, CellMap
from src.core.gbnn_field import GBNNField
from src.core.usv import USV


def _setup_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_trajectory(cell_map: CellMap, usv: USV, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    base = np.zeros_like(cell_map.grid, dtype=int)
    base[cell_map.grid == UNCOVERED] = 0
    base[cell_map.grid == COVERED] = 1
    base[cell_map.grid == OBSTACLE] = 2
    cmap = ListedColormap(["#f7f7f7", "#d9ead3", "#303030"])
    ax.imshow(base, origin="lower", cmap=cmap, interpolation="nearest")

    def draw_segments(mode: str, escape_type: str | None, color: str, linestyle: str, label: str) -> None:
        first = True
        for i in range(1, len(usv.path)):
            if usv.mode_history[i] != mode:
                continue
            if escape_type is not None and usv.escape_type_history[i] != escape_type:
                continue
            a, b = usv.path[i - 1], usv.path[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linestyle=linestyle, linewidth=1.2, label=label if first else None)
            first = False

    draw_segments("normal", None, "#1f77b4", "-", "normal")
    draw_segments("escape", "backtracking", "#2ca02c", "--", "backtracking escape")
    draw_segments("escape", "dijkstra", "#ff7f0e", "--", "dijkstra escape")
    if usv.path:
        ax.scatter([usv.path[0][0]], [usv.path[0][1]], c="#00aa00", s=45, marker="o", label="start", zorder=5)
        ax.scatter([usv.path[-1][0]], [usv.path[-1][1]], c="#cc0000", s=45, marker="x", label="end", zorder=5)
    ax.set_title("Coverage trajectory")
    ax.set_xlim(-0.5, cell_map.width - 0.5)
    ax.set_ylim(-0.5, cell_map.height - 0.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(color="#bbbbbb", linewidth=0.15)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)


def plot_activity_map(gbnn_field: GBNNField, path: str | Path) -> None:
    activity = gbnn_field.normalized_activity()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(activity, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title("Final GBNN activity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)


def plot_coverage_map(cell_map: CellMap, path: str | Path) -> None:
    base = np.zeros_like(cell_map.grid, dtype=int)
    base[cell_map.grid == UNCOVERED] = 0
    base[cell_map.grid == COVERED] = 1
    base[cell_map.grid == OBSTACLE] = 2
    cmap = ListedColormap(["#ffffff", "#74c476", "#222222"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(base, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title("Final coverage map")
    ax.set_aspect("equal")
    ax.grid(color="#bbbbbb", linewidth=0.15)
    fig.tight_layout()
    fig.savefig(_setup_path(path), dpi=300)
    plt.close(fig)
