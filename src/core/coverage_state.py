from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoverageState:
    grid_map: object

    def __post_init__(self) -> None:
        shape = self.grid_map.grid.shape
        self.covered_map = np.zeros(shape, dtype=bool)
        self.visit_count = np.zeros(shape, dtype=np.uint16)
        self.nominal_overlap = 1

    def update_by_path(self, path, coverage_radius: float) -> None:
        radius_cells = max(1, int(round(coverage_radius / self.grid_map.resolution)))
        self.nominal_overlap = max(self.nominal_overlap, 2 * radius_cells + 2)
        for point in path:
            self.update_by_point(point, coverage_radius)

    def update_by_point(self, point, coverage_radius: float) -> None:
        center = self.grid_map.world_to_grid(point) if isinstance(point[0], float) else point
        radius_cells = max(0, int(round(coverage_radius / self.grid_map.resolution)))
        r0, c0 = center
        for r in range(max(0, r0 - radius_cells), min(self.grid_map.height, r0 + radius_cells + 1)):
            for c in range(max(0, c0 - radius_cells), min(self.grid_map.width, c0 + radius_cells + 1)):
                if self.grid_map.grid[r, c] != 0:
                    continue
                if (r - r0) ** 2 + (c - c0) ** 2 <= radius_cells**2:
                    self.covered_map[r, c] = True
                    self.visit_count[r, c] += 1

    @property
    def coverage_rate(self) -> float:
        free = self.grid_map.grid == 0
        total = int(free.sum())
        return float((self.covered_map & free).sum() / total) if total else 1.0

    @property
    def repeated_coverage_rate(self) -> float:
        free = self.grid_map.grid == 0
        covered = int((self.visit_count[free] > 0).sum())
        repeated = int((self.visit_count[free] > self.nominal_overlap).sum())
        return float(repeated / covered) if covered else 0.0

    def uncovered_mask(self) -> np.ndarray:
        return (self.grid_map.grid == 0) & (~self.covered_map)

    def residual_components(self, min_size: int = 1) -> list[list[tuple[int, int]]]:
        comps = self.grid_map.connected_components(self.uncovered_mask())
        return [comp for comp in comps if len(comp) >= min_size]
