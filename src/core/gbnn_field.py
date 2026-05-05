from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GBNNField:
    grid_map: object
    coverage_state: object
    iterations: int = 5
    excitation_uncovered: float = 1.0
    inhibition_covered: float = 0.2
    inhibition_obstacle: float = 2.0
    residual_boost: float = 1.5
    neighbor_weight: float = 0.2

    def compute(self, residual_components: list[list[tuple[int, int]]] | None = None) -> np.ndarray:
        free = self.grid_map.grid == 0
        activity = np.zeros(self.grid_map.grid.shape, dtype=float)
        activity[self.coverage_state.uncovered_mask()] = self.excitation_uncovered
        activity[self.coverage_state.covered_map & free] = -self.inhibition_covered
        activity[self.grid_map.grid != 0] = -self.inhibition_obstacle
        for comp in residual_components or []:
            for r, c in comp:
                activity[r, c] += self.residual_boost

        for _ in range(self.iterations):
            propagated = activity.copy()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                shifted = np.roll(activity, shift=(dr, dc), axis=(0, 1))
                if dr == -1:
                    shifted[-1, :] = 0
                if dr == 1:
                    shifted[0, :] = 0
                if dc == -1:
                    shifted[:, -1] = 0
                if dc == 1:
                    shifted[:, 0] = 0
                propagated += self.neighbor_weight * shifted
            activity = np.where(free, propagated / (1 + 4 * self.neighbor_weight), -self.inhibition_obstacle)
        return activity
