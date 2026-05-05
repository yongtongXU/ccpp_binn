from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.sweep_generator import SweepGenerator
from src.core.segment_planner import SegmentPlanner
from src.utils.astar import astar
from src.utils.geometry import Cell, dedupe_path, euclidean, path_length


@dataclass
class RecoveryResult:
    cells: list[Cell]
    modes: list[str]
    components_before: int
    components_after: int = 0
    recovery_path_length: float = 0.0


class ResidualRecovery:
    def __init__(self, grid_map, coverage_state, min_residual_size: int = 3, coverage_width: float = 4.0):
        self.grid_map = grid_map
        self.coverage_state = coverage_state
        self.min_residual_size = min_residual_size
        self.coverage_width = coverage_width

    def recover(self, current: Cell, activity_map: np.ndarray | None, coverage_radius: float) -> RecoveryResult:
        comps = self.coverage_state.residual_components(self.min_residual_size)
        before = len(comps)
        ordered = self._rank_components(comps, current, activity_map)
        path: list[Cell] = []
        modes: list[str] = []
        cur = current
        for comp in ordered:
            targets = self._component_path(comp)
            if not targets:
                continue
            entry_index = min(range(len(targets)), key=lambda i: euclidean(cur, targets[i]))
            targets = targets[entry_index:] + targets[:entry_index]
            local = self._connect_sequence(targets)
            connector = astar(self.grid_map, cur, local[0]) or (_line_cells(cur, local[0]) if self.grid_map.line_is_free(cur, local[0]) else [])
            if connector:
                path.extend(connector[1:] if path else connector)
                modes.extend(["connection"] * (len(connector[1:] if path else connector)))
            path.extend(local[1:] if path and path[-1] == local[0] else local)
            modes.extend(["recovery"] * (len(path) - len(modes)))
            for cell in local:
                self.coverage_state.update_by_point(cell, coverage_radius)
            cur = path[-1] if path else cur
        path = dedupe_path(path)
        world = [self.grid_map.grid_to_world(c) for c in path]
        after = len(self.coverage_state.residual_components(self.min_residual_size))
        return RecoveryResult(path, modes[: len(path)], before, after, path_length(world))

    def _rank_components(self, comps: list[list[Cell]], current: Cell, activity_map: np.ndarray | None) -> list[list[Cell]]:
        def score(comp: list[Cell]) -> float:
            centroid = (sum(r for r, _ in comp) / len(comp), sum(c for _, c in comp) / len(comp))
            act = float(np.mean([activity_map[r, c] for r, c in comp])) if activity_map is not None else 1.0
            return euclidean(current, centroid) - 2.0 * act - 0.05 * len(comp)

        return sorted(comps, key=score)

    def _component_path(self, comp: list[Cell]) -> list[Cell]:
        if len(comp) <= max(6, self.min_residual_size * 2):
            return self._nearest_neighbor(comp)
        return self._local_sweep(comp)

    def _nearest_neighbor(self, comp: list[Cell]) -> list[Cell]:
        remaining = set(comp)
        cur = min(remaining)
        path = [cur]
        remaining.remove(cur)
        while remaining:
            nxt = min(remaining, key=lambda cell: euclidean(cur, cell))
            path.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        return path

    def _local_sweep(self, comp: list[Cell]) -> list[Cell]:
        mask = np.zeros_like(self.grid_map.grid, dtype=bool)
        for cell in comp:
            mask[cell] = True
        rows = sorted(set(r for r, _ in comp))
        result: list[Cell] = []
        flip = False
        step = max(1, int(round(self.coverage_width / self.grid_map.resolution)))
        for r in rows[::step]:
            cols = sorted(c for rr, c in comp if rr == r)
            if not cols:
                continue
            cells = [(r, c) for c in (reversed(cols) if flip else cols)]
            result.extend(cells)
            flip = not flip
        return result

    def _connect_sequence(self, targets: list[Cell]) -> list[Cell]:
        if not targets:
            return []
        expanded = [targets[0]]
        for target in targets[1:]:
            cur = expanded[-1]
            connector = astar(self.grid_map, cur, target) or (_line_cells(cur, target) if self.grid_map.line_is_free(cur, target) else [target])
            expanded.extend(connector[1:] if connector and connector[0] == cur else connector)
        return dedupe_path(expanded)


def _line_cells(a: Cell, b: Cell) -> list[Cell]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells
