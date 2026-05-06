from __future__ import annotations

from pathlib import Path

from src.core.cell_map import CellMap
from src.core.escape_selector import EscapeSelector
from src.core.gbnn_field import GBNNField
from src.core.metrics import SUMMARY_COLUMNS, compute_metrics
from src.core.rolling_optimizer import RollingOptimizer
from src.core.usv import USV
from src.utils.io import ensure_scenario_dirs, write_rows_csv


class CoveragePlanner:
    def __init__(self, config: dict):
        self.config = config
        self.scenario = config.get("scenario", {}).get("name") or Path(config.get("scenario_path", "scenario")).stem
        self.cell_map = CellMap.from_config(config)
        start_cfg = config.get("start", {"x": 0, "y": 0})
        start = (int(start_cfg.get("x", 0)), int(start_cfg.get("y", 0)))
        if not self.cell_map.is_traversable(start):
            candidates = self.cell_map.all_traversable_cells()
            if not candidates:
                raise ValueError("Map has no traversable cells")
            start = candidates[0]
        self.usv = USV(start)
        self.gbnn = GBNNField(config.get("gbnn", {}))
        self.rolling = RollingOptimizer(config.get("rolling_optimizer", {}))
        escape_cfg = {**config.get("escape", {}), **config.get("planner", {})}
        self.escape = EscapeSelector(escape_cfg)
        self.path_rows: list[dict] = []
        self.decision_rows: list[dict] = []
        self.escape_rows: list[dict] = []
        self.coverage_history: list[float] = []
        self.deadlock_count = 0
        self.failure_reason = ""
        self.mode = "normal"
        self._escape_path: list[tuple[int, int]] = []
        self._escape_type = "none"
        self._escape_id = 0

    def run(self, output_root: str | Path | None = None, save_outputs: bool = True) -> dict:
        planner_cfg = self.config.get("planner", {})
        max_steps = int(planner_cfg.get("max_steps", 30000))
        target = float(planner_cfg.get("target_coverage_rate", 1.0))
        output_root = output_root or self.config.get("output", {}).get("root", "outputs")
        self.cell_map.mark_covered(self.usv.current_cell)
        self.gbnn.initialize(self.cell_map)
        self._record_path_row(0, "normal", "none")
        self.coverage_history.append(self.cell_map.coverage_rate())

        step = 0
        while self.cell_map.coverage_rate() < target and step < max_steps:
            step += 1
            self.gbnn.update(self.cell_map)
            if self._escape_path:
                next_cell = self._escape_path.pop(0)
                if next_cell == self.usv.current_cell and self._escape_path:
                    next_cell = self._escape_path.pop(0)
                if next_cell == self.usv.current_cell:
                    self._escape_path = []
                    self._escape_type = "none"
                    continue
                self.usv.move_to(next_cell, "escape", self._escape_type)
                self.cell_map.mark_covered(next_cell)
                if not self._escape_path:
                    self._escape_type = "none"
            else:
                next_cell, branch, details = self.rolling.select_next_cell(self.usv, self.cell_map, self.gbnn)
                if next_cell is None and self._escape_allowed():
                    self.deadlock_count += 1
                    reason = "rolling_none_strip_stagnation"
                    target_cell, escape_type, escape_path, debug = self.escape.select_escape_target(
                        self.usv, self.cell_map, self.rolling, self.gbnn, reason=reason
                    )
                    if not target_cell or len(escape_path) <= 1:
                        self.failure_reason = debug.get("reason", "escape_failed")
                        break
                    self._escape_id += 1
                    self._escape_type = escape_type
                    self._escape_path = escape_path[1:]
                    self.escape_rows.append(
                        {
                            "escape_id": self._escape_id,
                            "step": step,
                            "escape_type": escape_type,
                            "start_x": self.usv.current_cell[0],
                            "start_y": self.usv.current_cell[1],
                            "target_x": target_cell[0],
                            "target_y": target_cell[1],
                            "path_length": len(escape_path) - 1,
                            "candidate_score": debug.get("candidate_score"),
                            "reason": reason,
                        }
                    )
                    next_cell = self._escape_path.pop(0)
                    self.usv.move_to(next_cell, "escape", self._escape_type)
                    self.cell_map.mark_covered(next_cell)
                    if not self._escape_path:
                        self._escape_type = "none"
                else:
                    if next_cell is None:
                        fallback = self.rolling._local_strip_fallback(self.usv, self.cell_map, self.gbnn)
                        if not fallback:
                            self.failure_reason = "rolling_failed_escape_not_allowed"
                            break
                        next_cell = fallback[0]
                        branch = fallback
                        details = self.rolling.score_branch(self.usv, self.cell_map, self.gbnn, branch)
                    advance_strip = self._should_advance_strip(next_cell)
                    self.usv.move_to(next_cell, "normal", "none", advance_strip=advance_strip)
                    self.cell_map.mark_covered(next_cell)
                    self._record_decision(step, branch, details, next_cell)
            self.coverage_history.append(self.cell_map.coverage_rate())
            self._record_path_row(step, self.usv.mode_history[-1], self.usv.escape_type_history[-1])

        if self.cell_map.coverage_rate() >= target:
            self.mode = "finished"
            success = True
            self.failure_reason = ""
        else:
            success = self.cell_map.coverage_rate() >= min(target, 0.98)
            if not self.failure_reason:
                self.failure_reason = "max_steps_reached" if step >= max_steps else "planner_stopped"

        metrics = compute_metrics(self.scenario, success, self.cell_map, self.usv, self.deadlock_count, self.escape_rows, self.failure_reason)
        if save_outputs:
            self.write_outputs(output_root, metrics)
        return metrics

    def write_outputs(self, output_root: str | Path, metrics: dict) -> None:
        from src.visualization.plotter import plot_activity_map, plot_coverage_map, plot_trajectory

        dirs = ensure_scenario_dirs(output_root, self.scenario)
        write_rows_csv(dirs["data"] / "path.csv", self.path_rows, ["step", "x", "y", "mode", "escape_type", "coverage_rate", "repeated_coverage_rate"])
        write_rows_csv(
            dirs["data"] / "decisions.csv",
            self.decision_rows,
            [
                "step",
                "current_x",
                "current_y",
                "selected_x",
                "selected_y",
                "branch_score",
                "new_coverage_score",
                "activity_score",
                "direction_score",
                "structure_score",
                "turn_penalty",
                "repeat_penalty",
                "dead_zone_penalty",
                "obstacle_penalty",
                "selected_branch",
            ],
        )
        write_rows_csv(dirs["data"] / "escapes.csv", self.escape_rows, ["escape_id", "step", "escape_type", "start_x", "start_y", "target_x", "target_y", "path_length", "candidate_score", "reason"])
        write_rows_csv(dirs["data"] / "metrics.csv", [metrics], SUMMARY_COLUMNS)
        plot_trajectory(self.cell_map, self.usv, dirs["figures"] / "trajectory.png")
        plot_activity_map(self.gbnn, dirs["figures"] / "activity_map.png")
        plot_coverage_map(self.cell_map, dirs["figures"] / "coverage_map.png")

    def _record_path_row(self, step: int, mode: str, escape_type: str) -> None:
        x, y = self.usv.current_cell
        self.path_rows.append(
            {
                "step": step,
                "x": x,
                "y": y,
                "mode": mode,
                "escape_type": escape_type,
                "coverage_rate": self.cell_map.coverage_rate(),
                "repeated_coverage_rate": self.cell_map.repeated_coverage_rate(),
            }
        )

    def _escape_allowed(self) -> bool:
        planner_cfg = self.config.get("planner", {})
        stagnation_steps = int(planner_cfg.get("stagnation_steps", 20))
        min_increment = float(planner_cfg.get("min_coverage_increment", 0.0001))
        if len(self.coverage_history) < stagnation_steps:
            return False
        recent = self.coverage_history[-stagnation_steps:]
        if max(recent) - min(recent) > min_increment:
            return False
        if self.rolling.current_or_adjacent_strip_has_continuation(self.usv, self.cell_map, self.gbnn):
            return False
        return True

    def _should_advance_strip(self, next_cell: tuple[int, int]) -> bool:
        current_strip = self.usv.current_strip_id if self.usv.current_strip_id is not None else self.usv.current_cell[1]
        if next_cell[1] == current_strip:
            return False
        return not self.rolling.current_strip_has_forward_uncovered(self.usv, self.cell_map)

    def _record_decision(self, step: int, branch: list[tuple[int, int]], details: dict, selected: tuple[int, int]) -> None:
        cur = self.usv.path[-2] if len(self.usv.path) >= 2 else self.usv.current_cell
        self.decision_rows.append(
            {
                "step": step,
                "current_x": cur[0],
                "current_y": cur[1],
                "selected_x": selected[0],
                "selected_y": selected[1],
                "selected_branch": ";".join(f"{x}:{y}" for x, y in branch),
                **{k: details.get(k) for k in ["branch_score", "new_coverage_score", "activity_score", "direction_score", "structure_score", "turn_penalty", "repeat_penalty", "dead_zone_penalty", "obstacle_penalty"]},
            }
        )
