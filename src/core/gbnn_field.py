from __future__ import annotations

import numpy as np

from src.core.cell_map import COVERED, OBSTACLE, UNCOVERED, Cell, CellMap


class GBNNField:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.activity: np.ndarray | None = None

    def initialize(self, cell_map: CellMap) -> None:
        self.activity = np.zeros((cell_map.height, cell_map.width), dtype=float)
        self.update(cell_map)

    def _external_input(self, cell_map: CellMap) -> np.ndarray:
        cfg = self.config
        external = np.zeros_like(cell_map.grid, dtype=float)
        external[cell_map.grid == UNCOVERED] = float(cfg.get("external_excitation", 1.0))
        external[cell_map.grid == COVERED] = float(cfg.get("covered_input", 0.0))
        external[cell_map.grid == OBSTACLE] = float(cfg.get("obstacle_inhibition", -2.0))
        return external

    def update(self, cell_map: CellMap) -> None:
        if self.activity is None:
            self.initialize(cell_map)
            return
        cfg = self.config
        if cfg.get("enabled", True) is False:
            self.activity = self._external_input(cell_map)
            return
        iterations = int(cfg.get("iterations_per_step", 1))
        decay = float(cfg.get("activity_decay", 0.8))
        neighbor_weight = float(cfg.get("neighbor_weight", 0.2))
        external = self._external_input(cell_map)
        for _ in range(max(1, iterations)):
            spread = np.zeros_like(self.activity)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = np.zeros_like(self.activity)
                    ys = slice(max(0, dy), cell_map.height + min(0, dy))
                    xs = slice(max(0, dx), cell_map.width + min(0, dx))
                    tys = slice(max(0, -dy), cell_map.height + min(0, -dy))
                    txs = slice(max(0, -dx), cell_map.width + min(0, -dx))
                    shifted[ys, xs] = self.activity[tys, txs]
                    spread += shifted
            self.activity = decay * self.activity + neighbor_weight * spread / 8.0 + external
            self.activity[cell_map.grid == OBSTACLE] = float(cfg.get("obstacle_inhibition", -2.0))

    def get_activity(self, cell: Cell) -> float:
        if self.activity is None:
            return 0.0
        x, y = cell
        if y < 0 or y >= self.activity.shape[0] or x < 0 or x >= self.activity.shape[1]:
            return float(self.config.get("obstacle_inhibition", -2.0))
        return float(self.activity[y, x])

    def normalized_activity(self) -> np.ndarray:
        if self.activity is None:
            return np.array([])
        arr = self.activity.copy()
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
