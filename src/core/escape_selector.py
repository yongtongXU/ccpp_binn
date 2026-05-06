from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.cell_map import Cell, CellMap
from src.core.graph_search import astar, dijkstra, move_cost
from src.core.rolling_optimizer import RollingOptimizer
from src.core.usv import USV


@dataclass
class EscapeCandidate:
    target: Cell
    escape_type: str
    path: list[Cell]
    score: float
    reason: str
    details: dict


class EscapeSelector:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def detect_deadlock(
        self,
        usv: USV,
        cell_map: CellMap,
        coverage_history: list[float],
        rolling_success: bool = True,
        last_new_coverage_step: int | None = None,
    ) -> tuple[bool, str]:
        if not rolling_success:
            return True, "rolling_optimizer_failed"
        stagnation_steps = int(self.config.get("stagnation_steps", self.config.get("planner_stagnation_steps", 20)))
        if len(coverage_history) > stagnation_steps:
            recent = coverage_history[-stagnation_steps:]
            if max(recent) - min(recent) <= float(self.config.get("min_coverage_increment", 0.0001)):
                return True, "coverage_stagnation"
        return False, "none"

    def find_backtracking_candidate(
        self,
        usv: USV,
        cell_map: CellMap,
        rolling_optimizer: RollingOptimizer | None = None,
        gbnn_field=None,
    ) -> EscapeCandidate | None:
        if self.config.get("backtracking_enabled", True) is False:
            return None
        max_steps = int(self.config.get("backtracking_max_steps", 500))
        history = usv.path[:-1]
        for idx_from_end, cell in enumerate(reversed(history[-max_steps:]), start=1):
            if not cell_map.is_traversable(cell):
                continue
            if not self._has_future_potential(cell_map, cell):
                continue
            path = [usv.current_cell] + list(reversed(history[len(history) - idx_from_end :]))
            if path[-1] != cell:
                path.append(cell)
            score = self.evaluate_escape_candidate(cell_map, path, cell)
            return EscapeCandidate(cell, "backtracking", path, score, "history_cell_with_uncovered_neighbor", {})
        return None

    def find_dijkstra_candidate(self, usv: USV, cell_map: CellMap) -> EscapeCandidate | None:
        if self.config.get("dijkstra_enabled", True) is False:
            return None

        def cost_fn(a: Cell, b: Cell) -> float:
            repeat = 1.0 if cell_map.visit_count[b[1], b[0]] > 0 else 0.0
            obstacle = self._obstacle_proximity(cell_map, b)
            narrow = max(0.0, 3.0 - len(cell_map.neighbors8(b))) / 3.0
            return move_cost(a, b) + 0.8 * repeat + 0.3 * obstacle + 0.4 * narrow

        def target_condition(c: Cell) -> bool:
            return cell_map.is_uncovered(c) or self._has_future_potential(cell_map, c)

        path, _, info = dijkstra(
            cell_map,
            usv.current_cell,
            target_condition=target_condition,
            cost_fn=cost_fn,
            max_expansion=int(self.config.get("dijkstra_max_expansion", 5000)),
        )
        if not path:
            return None
        target = path[-1]
        score = self.evaluate_escape_candidate(cell_map, path, target)
        return EscapeCandidate(target, "dijkstra", path, score, "reachable_uncovered_or_component_entry", info)

    def evaluate_escape_candidate(self, cell_map: CellMap, path: list[Cell], target: Cell) -> float:
        cfg = self.config
        distance = max(0, len(path) - 1)
        repeats = sum(1 for c in path[1:] if cell_map.visit_count[c[1], c[0]] > 0)
        turns = _count_turns(path)
        target_uncovered_neighbors = sum(1 for n in cell_map.neighbors8(target) if cell_map.is_uncovered(n))
        component_size = self._component_size_for(cell_map, target)
        future = target_uncovered_neighbors + min(component_size, 50) / 10.0
        return (
            cfg.get("w_escape_distance", 1.0) * distance
            + cfg.get("w_escape_repeat", 2.0) * repeats
            + cfg.get("w_escape_turn", 1.0) * turns
            + cfg.get("w_escape_component_size", -1.5) * math.log1p(component_size)
            + cfg.get("w_escape_future_potential", -1.0) * future
        )

    def select_escape_target(
        self,
        usv: USV,
        cell_map: CellMap,
        rolling_optimizer: RollingOptimizer | None = None,
        gbnn_field=None,
        reason: str = "deadlock",
    ) -> tuple[Cell | None, str, list[Cell], dict]:
        if self.config.get("enabled", True) is False:
            return None, "none", [], {"reason": "escape_disabled"}
        candidates = []
        backtracking = self.find_backtracking_candidate(usv, cell_map, rolling_optimizer, gbnn_field)
        if backtracking:
            candidates.append(backtracking)
        dij = self.find_dijkstra_candidate(usv, cell_map)
        if dij:
            candidates.append(dij)
        if not candidates:
            return None, "none", [], {"reason": "no_escape_candidate"}
        selected = min(candidates, key=lambda c: c.score)
        if selected.escape_type == "backtracking":
            # Validate with graph search if the historical segment is no longer directly usable.
            if not all(cell_map.is_traversable(c) for c in selected.path):
                path, _, _ = astar(cell_map, usv.current_cell, selected.target)
                selected.path = path or []
        debug = {
            "reason": reason,
            "candidate_score": selected.score,
            "candidates": [
                {"type": c.escape_type, "target": c.target, "score": c.score, "path_length": len(c.path) - 1}
                for c in candidates
            ],
        }
        return selected.target, selected.escape_type, selected.path, debug

    def _has_future_potential(self, cell_map: CellMap, cell: Cell) -> bool:
        return sum(1 for n in cell_map.neighbors8(cell) if cell_map.is_uncovered(n)) >= int(
            self.config.get("min_uncovered_neighbors", 1)
        )

    def _component_size_for(self, cell_map: CellMap, target: Cell) -> int:
        if cell_map.is_uncovered(target):
            start = target
        else:
            uncovered_neighbors = [n for n in cell_map.neighbors8(target) if cell_map.is_uncovered(n)]
            if not uncovered_neighbors:
                return 0
            start = uncovered_neighbors[0]
        seen = {start}
        stack = [start]
        while stack:
            c = stack.pop()
            for n in cell_map.neighbors8(c):
                if n not in seen and cell_map.is_uncovered(n):
                    seen.add(n)
                    stack.append(n)
        return len(seen)

    def _obstacle_proximity(self, cell_map: CellMap, cell: Cell) -> float:
        x, y = cell
        risk = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if not cell_map.in_bounds(n) or cell_map.is_obstacle(n):
                    risk += 1.0
        return risk / 8.0


def _count_turns(path: list[Cell]) -> int:
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        heading = (0 if b[0] == a[0] else (1 if b[0] > a[0] else -1), 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1))
        if prev is not None and heading != prev:
            turns += 1
        prev = heading
    return turns
