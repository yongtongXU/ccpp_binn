from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from src.utils.geometry import Cell, Point

FREE = 0
OBSTACLE = 1
UNAVAILABLE = 2


@dataclass
class GridMap:
    width: int
    height: int
    resolution: float = 1.0
    grid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    @classmethod
    def from_config(cls, cfg: dict) -> "GridMap":
        map_cfg = cfg.get("map", cfg)
        gm = cls(
            width=int(map_cfg.get("width", 100)),
            height=int(map_cfg.get("height", 100)),
            resolution=float(map_cfg.get("resolution", 1.0)),
        )
        boundary = map_cfg.get("boundary")
        if boundary:
            gm.apply_polygon_boundary([tuple(p) for p in boundary])
        for obs in map_cfg.get("obstacles", []):
            kind = obs.get("type", "rectangle")
            if kind == "rectangle":
                gm.add_rectangle_obstacle(obs["x"], obs["y"], obs["width"], obs["height"])
            elif kind == "circle":
                gm.add_circle_obstacle(obs["cx"], obs["cy"], obs["radius"])
            elif kind == "polygon":
                gm.add_polygon_obstacle([tuple(p) for p in obs["points"]])
            else:
                raise ValueError(f"Unsupported obstacle type: {kind}")
        return gm

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free_cell(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.grid[cell] == FREE

    def world_to_grid(self, point: Point) -> Cell:
        x, y = point
        c = int(math.floor(x / self.resolution))
        r = int(math.floor(y / self.resolution))
        return (max(0, min(self.height - 1, r)), max(0, min(self.width - 1, c)))

    def grid_to_world(self, cell: Cell) -> Point:
        r, c = cell
        return ((c + 0.5) * self.resolution, (r + 0.5) * self.resolution)

    def add_rectangle_obstacle(self, x: float, y: float, width: float, height: float) -> None:
        r0, c0 = self.world_to_grid((x, y))
        r1, c1 = self.world_to_grid((x + width, y + height))
        self.grid[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = OBSTACLE

    def add_circle_obstacle(self, cx: float, cy: float, radius: float) -> None:
        for r in range(self.height):
            for c in range(self.width):
                x, y = self.grid_to_world((r, c))
                if math.hypot(x - cx, y - cy) <= radius:
                    self.grid[r, c] = OBSTACLE

    def add_polygon_obstacle(self, points: list[Point]) -> None:
        for r in range(self.height):
            for c in range(self.width):
                if _point_in_polygon(self.grid_to_world((r, c)), points):
                    self.grid[r, c] = OBSTACLE

    def apply_polygon_boundary(self, points: list[Point]) -> None:
        for r in range(self.height):
            for c in range(self.width):
                if not _point_in_polygon(self.grid_to_world((r, c)), points):
                    self.grid[r, c] = UNAVAILABLE

    def line_is_free(self, a: Cell | Point, b: Cell | Point, world: bool = False) -> bool:
        start = self.world_to_grid(a) if world else a
        goal = self.world_to_grid(b) if world else b
        for cell in _bresenham(start, goal):
            if not self.is_free_cell(cell):
                return False
        return True

    def neighbors(self, cell: Cell, diagonal: bool = False) -> list[Cell]:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        result = []
        for dr, dc in dirs:
            nb = (cell[0] + dr, cell[1] + dc)
            if self.is_free_cell(nb):
                result.append(nb)
        return result

    def get_free_cells(self) -> list[Cell]:
        rows, cols = np.where(self.grid == FREE)
        return list(zip(rows.astype(int), cols.astype(int)))

    def connected_components(self, mask: np.ndarray) -> list[list[Cell]]:
        seen = np.zeros(mask.shape, dtype=bool)
        components: list[list[Cell]] = []
        valid = mask.astype(bool)
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if not valid[r, c] or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                comp: list[Cell] = []
                while q:
                    cur = q.popleft()
                    comp.append(cur)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nb = (cur[0] + dr, cur[1] + dc)
                        if 0 <= nb[0] < mask.shape[0] and 0 <= nb[1] < mask.shape[1]:
                            if valid[nb] and not seen[nb]:
                                seen[nb] = True
                                q.append(nb)
                components.append(comp)
        return components


def _bresenham(a: Cell, b: Cell) -> Iterable[Cell]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        yield (r, c)
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside
