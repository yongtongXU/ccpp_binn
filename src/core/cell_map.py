from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np


UNCOVERED = 0
COVERED = 1
OBSTACLE = 2
BLOCKED = 3
Cell = tuple[int, int]


@dataclass
class CellMap:
    width: int
    height: int
    cell_size: float = 1.0
    grid: np.ndarray | None = None
    visit_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        if self.visit_count is None:
            self.visit_count = np.zeros((self.height, self.width), dtype=np.int32)

    @classmethod
    def from_config(cls, config: dict) -> "CellMap":
        map_cfg = config.get("map", config)
        cell_map = cls(
            width=int(map_cfg["width"]),
            height=int(map_cfg["height"]),
            cell_size=float(map_cfg.get("cell_size", 1.0)),
        )
        for obstacle in map_cfg.get("obstacles", []) or []:
            cell_map.add_obstacle(obstacle)
        return cell_map

    def copy(self) -> "CellMap":
        return CellMap(
            width=self.width,
            height=self.height,
            cell_size=self.cell_size,
            grid=deepcopy(self.grid),
            visit_count=deepcopy(self.visit_count),
        )

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def state(self, cell: Cell) -> int:
        if not self.in_bounds(cell):
            return BLOCKED
        x, y = cell
        return int(self.grid[y, x])

    def is_traversable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.state(cell) in (UNCOVERED, COVERED)

    def is_obstacle(self, cell: Cell) -> bool:
        return self.state(cell) == OBSTACLE

    def is_uncovered(self, cell: Cell) -> bool:
        return self.state(cell) == UNCOVERED

    def mark_covered(self, cell: Cell) -> None:
        if not self.is_traversable(cell):
            return
        x, y = cell
        self.visit_count[y, x] += 1
        self.grid[y, x] = COVERED

    def neighbors8(self, cell: Cell) -> list[Cell]:
        x, y = cell
        result: list[Cell] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if self.is_traversable(n):
                    result.append(n)
        return result

    def has_uncovered_neighbor8(self, cell: Cell) -> bool:
        return any(self.is_uncovered(n) for n in self.neighbors8(cell))

    def is_dead_zone(self, cell: Cell) -> bool:
        return self.is_traversable(cell) and not self.has_uncovered_neighbor8(cell)

    def all_traversable_cells(self) -> list[Cell]:
        ys, xs = np.where(self.grid != OBSTACLE)
        return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

    def get_uncovered_cells(self) -> list[Cell]:
        ys, xs = np.where(self.grid == UNCOVERED)
        return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

    def get_uncovered_components(self) -> list[list[Cell]]:
        remaining = set(self.get_uncovered_cells())
        components: list[list[Cell]] = []
        while remaining:
            root = remaining.pop()
            q = deque([root])
            comp = [root]
            while q:
                cell = q.popleft()
                for n in self.neighbors8(cell):
                    if n in remaining and self.is_uncovered(n):
                        remaining.remove(n)
                        q.append(n)
                        comp.append(n)
            components.append(comp)
        return components

    def coverage_rate(self) -> float:
        free = int(np.count_nonzero(self.grid != OBSTACLE))
        if free == 0:
            return 1.0
        covered = int(np.count_nonzero((self.grid != OBSTACLE) & (self.visit_count > 0)))
        return covered / free

    def repeated_coverage_rate(self) -> float:
        visits = int(self.visit_count[self.grid != OBSTACLE].sum())
        if visits == 0:
            return 0.0
        repeats = int(np.maximum(self.visit_count[self.grid != OBSTACLE] - 1, 0).sum())
        return repeats / visits

    def add_obstacle(self, obstacle: dict) -> None:
        kind = obstacle.get("type", "rectangle")
        if kind == "rectangle":
            self._add_rectangle(obstacle)
        elif kind == "circle":
            self._add_circle(obstacle)
        elif kind in ("u_shape", "concave"):
            for rect in obstacle.get("rectangles", []):
                self._add_rectangle({"type": "rectangle", **rect})
        else:
            raise ValueError(f"Unsupported obstacle type: {kind}")

    def _add_rectangle(self, obstacle: dict) -> None:
        x = int(obstacle.get("x", 0))
        y = int(obstacle.get("y", 0))
        width = int(obstacle.get("width", 1))
        height = int(obstacle.get("height", 1))
        x0, x1 = max(0, x), min(self.width, x + width)
        y0, y1 = max(0, y), min(self.height, y + height)
        self.grid[y0:y1, x0:x1] = OBSTACLE

    def _add_circle(self, obstacle: dict) -> None:
        cx = float(obstacle["x"])
        cy = float(obstacle["y"])
        radius = float(obstacle["radius"])
        yy, xx = np.ogrid[: self.height, : self.width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        self.grid[mask] = OBSTACLE


def traversable_neighbors_from(cell_map: CellMap, cells: Iterable[Cell]) -> set[Cell]:
    result: set[Cell] = set()
    for cell in cells:
        result.update(cell_map.neighbors8(cell))
    return result
