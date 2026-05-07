from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.cell_map import Cell, CellMap
from src.core.escape_selector import EscapeSelector
from src.core.gbnn_field import GBNNField
from src.core.rolling_optimizer import RollingOptimizer
from src.core.usv import USV


@dataclass
class StepDecision:
    next_cell: Cell | None
    branch: list[Cell]
    details: dict
    mode: str = "normal"
    escape_type: str = "none"
    advance_strip: bool | None = None
    escape_record: dict | None = None
    deadlock: bool = False
    failure_reason: str = ""


class CoverageStrategy(Protocol):
    name: str

    def choose_next(self, step: int, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> StepDecision:
        ...

    def after_step(self, coverage_rate: float) -> None:
        ...


class RollingGBNNStrategy:
    name = "rolling_gbnn"

    def __init__(self, config: dict):
        self.config = config
        self.rolling = RollingOptimizer(config.get("rolling_optimizer", {}))
        escape_cfg = {**config.get("escape", {}), **config.get("planner", {})}
        self.escape = EscapeSelector(escape_cfg)
        self.coverage_history: list[float] = []
        self._escape_path: list[Cell] = []
        self._escape_type = "none"
        self._escape_id = 0

    def choose_next(self, step: int, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> StepDecision:
        if self._escape_path:
            return self._continue_escape(usv)

        next_cell, branch, details = self.rolling.select_next_cell(usv, cell_map, gbnn_field)
        if next_cell is None and self._escape_allowed(usv, cell_map, gbnn_field):
            return self._start_escape(step, usv, cell_map, gbnn_field)

        if next_cell is None:
            fallback = self.rolling._local_strip_fallback(usv, cell_map, gbnn_field)
            if not fallback:
                return StepDecision(None, [], details, failure_reason="rolling_failed_escape_not_allowed")
            next_cell = fallback[0]
            branch = fallback
            details = self.rolling.score_branch(usv, cell_map, gbnn_field, branch)

        return StepDecision(
            next_cell=next_cell,
            branch=branch,
            details=details,
            advance_strip=self._should_advance_strip(usv, cell_map, next_cell),
        )

    def after_step(self, coverage_rate: float) -> None:
        self.coverage_history.append(coverage_rate)

    def _continue_escape(self, usv: USV) -> StepDecision:
        next_cell = self._escape_path.pop(0)
        if next_cell == usv.current_cell and self._escape_path:
            next_cell = self._escape_path.pop(0)
        if next_cell == usv.current_cell:
            self._escape_path = []
            self._escape_type = "none"
            return StepDecision(None, [], {"reason": "escape_path_stalled"})
        if not self._escape_path:
            escape_type = self._escape_type
            self._escape_type = "none"
        else:
            escape_type = self._escape_type
        return StepDecision(next_cell, [next_cell, *self._escape_path], {}, mode="escape", escape_type=escape_type)

    def _start_escape(self, step: int, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> StepDecision:
        reason = "rolling_none_strip_stagnation"
        target_cell, escape_type, escape_path, debug = self.escape.select_escape_target(
            usv, cell_map, self.rolling, gbnn_field, reason=reason
        )
        if not target_cell or len(escape_path) <= 1:
            return StepDecision(None, [], debug, deadlock=True, failure_reason=debug.get("reason", "escape_failed"))

        self._escape_id += 1
        self._escape_type = escape_type
        self._escape_path = escape_path[1:]
        next_cell = self._escape_path.pop(0)
        if not self._escape_path:
            self._escape_type = "none"
        record = {
            "escape_id": self._escape_id,
            "step": step,
            "escape_type": escape_type,
            "start_x": usv.current_cell[0],
            "start_y": usv.current_cell[1],
            "target_x": target_cell[0],
            "target_y": target_cell[1],
            "path_length": len(escape_path) - 1,
            "candidate_score": debug.get("candidate_score"),
            "reason": reason,
        }
        return StepDecision(next_cell, [next_cell, *self._escape_path], debug, mode="escape", escape_type=escape_type, escape_record=record, deadlock=True)

    def _escape_allowed(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> bool:
        if cell_map.is_dead_zone(usv.current_cell) and cell_map.coverage_rate() < 1.0:
            return True
        planner_cfg = self.config.get("planner", {})
        stagnation_steps = int(planner_cfg.get("stagnation_steps", 20))
        min_increment = float(planner_cfg.get("min_coverage_increment", 0.0001))
        if len(self.coverage_history) < stagnation_steps:
            return False
        recent = self.coverage_history[-stagnation_steps:]
        if max(recent) - min(recent) > min_increment:
            return False
        if self.rolling.current_or_adjacent_strip_has_continuation(usv, cell_map, gbnn_field):
            return False
        return True

    def _should_advance_strip(self, usv: USV, cell_map: CellMap, next_cell: Cell) -> bool:
        current_strip = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if next_cell[1] == current_strip:
            return False
        return not self.rolling.current_strip_has_forward_uncovered(usv, cell_map)


class GBNNGreedyStrategy:
    name = "gbnn_greedy"
    candidate_type = "gbnn_greedy"

    def __init__(self, config: dict):
        self.config = config
        self.method_cfg = config.get("method", {})
        escape_cfg = {**config.get("escape", {}), **config.get("planner", {})}
        self.escape = EscapeSelector(escape_cfg)
        self._escape_path: list[Cell] = []
        self._escape_type = "none"
        self._escape_id = 0

    def choose_next(self, step: int, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> StepDecision:
        if self._escape_path:
            return self._continue_escape(usv)
        if cell_map.is_dead_zone(usv.current_cell) and cell_map.coverage_rate() < 1.0:
            return self._start_escape(step, usv, cell_map, gbnn_field, reason="dead_zone")

        candidates = cell_map.neighbors8(usv.current_cell)
        if not candidates:
            return self._start_escape(step, usv, cell_map, gbnn_field, reason="no_traversable_neighbor")
        if not self.method_cfg.get("allow_immediate_backtrack", False) and len(usv.path) >= 2:
            filtered = [c for c in candidates if c != usv.path[-2]]
            if filtered:
                candidates = filtered
        next_cell = max(candidates, key=lambda cell: self._score_cell(cell, usv, cell_map, gbnn_field))
        candidate_branches = [
            {
                "type": self.candidate_type,
                "score": float(self._score_cell(cell, usv, cell_map, gbnn_field)),
                "path": [[int(cell[0]), int(cell[1])]],
            }
            for cell in sorted(candidates, key=lambda cell: self._score_cell(cell, usv, cell_map, gbnn_field), reverse=True)
        ]
        details = {
            "branch_score": float(self._score_cell(next_cell, usv, cell_map, gbnn_field)),
            "activity_score": float(gbnn_field.get_activity(next_cell)),
            "direction_score": float(1.0 - usv.heading_change_to(next_cell) / 4.0),
            "new_coverage_score": float(1.0 if cell_map.is_uncovered(next_cell) else 0.0),
            "structure_score": 0.0,
            "turn_penalty": 0.0,
            "repeat_penalty": float(1.0 if cell_map.visit_count[next_cell[1], next_cell[0]] > 0 else 0.0),
            "dead_zone_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "candidate_branches": candidate_branches,
        }
        return StepDecision(next_cell, [next_cell], details)

    def _score_cell(self, cell: Cell, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> float:
        heading_weight = float(self.method_cfg.get("heading_weight", 0.5))
        uncovered_bonus = float(self.method_cfg.get("uncovered_bonus", 0.25))
        activity = gbnn_field.get_activity(cell)
        heading = 1.0 - usv.heading_change_to(cell) / 4.0
        coverage = uncovered_bonus if cell_map.is_uncovered(cell) else 0.0
        return activity + heading_weight * heading + coverage

    def _continue_escape(self, usv: USV) -> StepDecision:
        next_cell = self._escape_path.pop(0)
        if next_cell == usv.current_cell and self._escape_path:
            next_cell = self._escape_path.pop(0)
        if next_cell == usv.current_cell:
            self._escape_path = []
            self._escape_type = "none"
            return StepDecision(None, [], {"reason": "escape_path_stalled"})
        escape_type = self._escape_type
        if not self._escape_path:
            self._escape_type = "none"
        return StepDecision(next_cell, [next_cell, *self._escape_path], {}, mode="escape", escape_type=escape_type)

    def _start_escape(self, step: int, usv: USV, cell_map: CellMap, gbnn_field: GBNNField, reason: str) -> StepDecision:
        target_cell, escape_type, escape_path, debug = self.escape.select_escape_target(
            usv, cell_map, None, gbnn_field, reason=reason
        )
        if not target_cell or len(escape_path) <= 1:
            return StepDecision(None, [], debug, deadlock=True, failure_reason=debug.get("reason", "escape_failed"))
        self._escape_id += 1
        self._escape_type = escape_type
        self._escape_path = escape_path[1:]
        next_cell = self._escape_path.pop(0)
        if not self._escape_path:
            self._escape_type = "none"
        record = {
            "escape_id": self._escape_id,
            "step": step,
            "escape_type": escape_type,
            "start_x": usv.current_cell[0],
            "start_y": usv.current_cell[1],
            "target_x": target_cell[0],
            "target_y": target_cell[1],
            "path_length": len(escape_path) - 1,
            "candidate_score": debug.get("candidate_score"),
            "reason": reason,
        }
        return StepDecision(next_cell, [next_cell, *self._escape_path], debug, mode="escape", escape_type=escape_type, escape_record=record, deadlock=True)

    def after_step(self, coverage_rate: float) -> None:
        return None


class OriginalBINNStrategy(GBNNGreedyStrategy):
    name = "original_binn"
    candidate_type = "original_binn"

    def _score_cell(self, cell: Cell, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> float:
        return gbnn_field.get_activity(cell)


class ImprovedBINNStrategy(GBNNGreedyStrategy):
    name = "improved_binn"
    candidate_type = "improved_binn"

    def _score_cell(self, cell: Cell, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> float:
        uncovered = 2.0 if cell_map.is_uncovered(cell) else 0.0
        heading = 1.0 - usv.heading_change_to(cell) / 4.0
        future = sum(1 for n in cell_map.neighbors8(cell) if cell_map.is_uncovered(n)) / 8.0
        repeat = 1.0 if cell_map.visit_count[cell[1], cell[0]] > 0 else 0.0
        return gbnn_field.get_activity(cell) + uncovered + 0.6 * heading + 0.8 * future - repeat


def create_strategy(config: dict) -> CoverageStrategy:
    method_cfg = config.get("method", {})
    name = str(method_cfg.get("name") or config.get("planner", {}).get("method") or "rolling_gbnn")
    registry = {
        RollingGBNNStrategy.name: RollingGBNNStrategy,
        GBNNGreedyStrategy.name: GBNNGreedyStrategy,
        OriginalBINNStrategy.name: OriginalBINNStrategy,
        ImprovedBINNStrategy.name: ImprovedBINNStrategy,
    }
    try:
        return registry[name](config)
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown coverage method '{name}'. Available methods: {available}") from exc
