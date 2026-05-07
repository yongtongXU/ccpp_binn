from __future__ import annotations

import json
from pathlib import Path

from src.core.cell_map import CellMap
from src.core.gbnn_field import GBNNField
from src.core.metrics import SUMMARY_COLUMNS, compute_metrics
from src.core.strategy import StepDecision, create_strategy
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
        self.strategy = create_strategy(config)
        self.path_rows: list[dict] = []
        self.decision_rows: list[dict] = []
        self.escape_rows: list[dict] = []
        self.coverage_history: list[float] = []
        self.deadlock_count = 0
        self.failure_reason = ""
        self.mode = "normal"

    def run(self, output_root: str | Path | None = None, save_outputs: bool = True) -> dict:
        planner_cfg = self.config.get("planner", {})
        max_steps = int(planner_cfg.get("max_steps", 30000))
        target = 1.0
        output_root = output_root or self.config.get("output", {}).get("root", "outputs")
        self.cell_map.mark_covered(self.usv.current_cell)
        self.gbnn.initialize(self.cell_map)
        self._record_path_row(0, "normal", "none")
        self.coverage_history.append(self.cell_map.coverage_rate())
        self.strategy.after_step(self.cell_map.coverage_rate())

        step = 0
        while self.cell_map.coverage_rate() < target and step < max_steps:
            step += 1
            self.gbnn.update(self.cell_map)
            decision = self.strategy.choose_next(step, self.usv, self.cell_map, self.gbnn)
            if decision.deadlock:
                self.deadlock_count += 1
            if decision.escape_record:
                self.escape_rows.append(decision.escape_record)
            if decision.next_cell is None:
                self.failure_reason = decision.failure_reason or decision.details.get("reason", "strategy_stopped")
                break

            self.usv.move_to(
                decision.next_cell,
                decision.mode,
                decision.escape_type,
                advance_strip=decision.advance_strip,
            )
            self.cell_map.mark_covered(decision.next_cell)
            self._record_decision(step, decision, decision.next_cell)

            self.coverage_history.append(self.cell_map.coverage_rate())
            self.strategy.after_step(self.cell_map.coverage_rate())
            self._record_path_row(step, self.usv.mode_history[-1], self.usv.escape_type_history[-1])

        if self.cell_map.coverage_rate() >= target:
            self.mode = "finished"
            success = True
            self.failure_reason = ""
        else:
            success = self.cell_map.coverage_rate() >= min(target, 0.98)
            if not self.failure_reason:
                self.failure_reason = "max_steps_reached" if step >= max_steps else "planner_stopped"

        metrics = compute_metrics(
            self.scenario,
            success,
            self.cell_map,
            self.usv,
            self.deadlock_count,
            self.escape_rows,
            self.failure_reason,
            method=self.strategy.name,
        )
        if save_outputs:
            self.write_outputs(output_root, metrics)
        return metrics

    def run_events(self, output_root: str | Path | None = None, save_outputs: bool = True):
        planner_cfg = self.config.get("planner", {})
        max_steps = int(planner_cfg.get("max_steps", 30000))
        target = 1.0
        output_root = output_root or self.config.get("output", {}).get("root", "outputs")
        self.cell_map.mark_covered(self.usv.current_cell)
        self.gbnn.initialize(self.cell_map)
        self._record_path_row(0, "normal", "none")
        self.coverage_history.append(self.cell_map.coverage_rate())
        self.strategy.after_step(self.cell_map.coverage_rate())
        yield {"type": "init", "pathRow": self.path_rows[-1]}

        step = 0
        while self.cell_map.coverage_rate() < target and step < max_steps:
            step += 1
            self.gbnn.update(self.cell_map)
            decision = self.strategy.choose_next(step, self.usv, self.cell_map, self.gbnn)
            if decision.deadlock:
                self.deadlock_count += 1
            if decision.escape_record:
                self.escape_rows.append(decision.escape_record)
            if decision.next_cell is None:
                self.failure_reason = decision.failure_reason or decision.details.get("reason", "strategy_stopped")
                yield {"type": "stopped", "step": step, "reason": self.failure_reason}
                break
            decision_row = self._build_decision_row(step, decision, decision.next_cell, self.usv.current_cell)
            yield {"type": "decision", "decisionRow": decision_row}

            self.usv.move_to(
                decision.next_cell,
                decision.mode,
                decision.escape_type,
                advance_strip=decision.advance_strip,
            )
            self.cell_map.mark_covered(decision.next_cell)
            self._record_decision(step, decision, decision.next_cell)

            self.coverage_history.append(self.cell_map.coverage_rate())
            self.strategy.after_step(self.cell_map.coverage_rate())
            self._record_path_row(step, self.usv.mode_history[-1], self.usv.escape_type_history[-1])
            yield {
                "type": "step",
                "pathRow": self.path_rows[-1],
                "decisionRow": self.decision_rows[-1],
            }

        metrics = self._finalize_metrics(step, target, max_steps)
        if save_outputs:
            self.write_outputs(output_root, metrics)
        yield {"type": "done", "metrics": metrics}

    def _finalize_metrics(self, step: int, target: float, max_steps: int) -> dict:
        if self.cell_map.coverage_rate() >= target:
            self.mode = "finished"
            success = True
            self.failure_reason = ""
        else:
            success = self.cell_map.coverage_rate() >= min(target, 0.98)
            if not self.failure_reason:
                self.failure_reason = "max_steps_reached" if step >= max_steps else "planner_stopped"

        return compute_metrics(
            self.scenario,
            success,
            self.cell_map,
            self.usv,
            self.deadlock_count,
            self.escape_rows,
            self.failure_reason,
            method=self.strategy.name,
        )

    def write_outputs(self, output_root: str | Path, metrics: dict) -> None:
        from src.visualization.plotter import (
            plot_activity_map,
            plot_coverage_map,
            plot_planning_animation,
            plot_planning_viewer,
            plot_trajectory,
        )

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
                "mode",
                "escape_type",
                "method",
                "branch_score",
                "new_coverage_score",
                "activity_score",
                "direction_score",
                "structure_score",
                "branch_urgency_score",
                "turn_penalty",
                "repeat_penalty",
                "dead_zone_penalty",
                "obstacle_penalty",
                "selected_branch",
                "candidate_branches",
                "candidate_tree",
            ],
        )
        write_rows_csv(dirs["data"] / "escapes.csv", self.escape_rows, ["escape_id", "step", "escape_type", "start_x", "start_y", "target_x", "target_y", "path_length", "candidate_score", "reason"])
        write_rows_csv(dirs["data"] / "metrics.csv", [metrics], SUMMARY_COLUMNS)
        plot_trajectory(self.cell_map, self.usv, dirs["figures"] / "trajectory.png")
        plot_activity_map(self.gbnn, dirs["figures"] / "activity_map.png")
        plot_coverage_map(self.cell_map, dirs["figures"] / "coverage_map.png")
        animation_cfg = self.config.get("output", {}).get("animation", {}) or {}
        if animation_cfg.get("enabled", True):
            plot_planning_animation(
                self.cell_map,
                self.path_rows,
                dirs["animations"] / "planning_process.gif",
                fps=int(animation_cfg.get("fps", 12)),
                max_frames=int(animation_cfg.get("max_frames", 300)),
            )
            plot_planning_viewer(
                self.cell_map,
                self.path_rows,
                self.decision_rows,
                dirs["animations"] / "planning_viewer.html",
                playback_speed=float(animation_cfg.get("playback_speed", 1.0)),
            )

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

    def _record_decision(self, step: int, decision: StepDecision, selected: tuple[int, int]) -> None:
        cur = self.usv.path[-2] if len(self.usv.path) >= 2 else self.usv.current_cell
        self.decision_rows.append(self._build_decision_row(step, decision, selected, cur))

    def _build_decision_row(self, step: int, decision: StepDecision, selected: tuple[int, int], current: tuple[int, int]) -> dict:
        branch = decision.branch
        details = decision.details
        return {
            "step": step,
            "current_x": current[0],
            "current_y": current[1],
            "selected_x": selected[0],
            "selected_y": selected[1],
            "mode": decision.mode,
            "escape_type": decision.escape_type,
            "method": self.strategy.name,
            "selected_branch": ";".join(f"{x}:{y}" for x, y in branch),
            "candidate_branches": json.dumps(self._candidate_branches(decision), ensure_ascii=True),
            "candidate_tree": json.dumps(self._candidate_tree(decision), ensure_ascii=True),
            **{k: details.get(k) for k in ["branch_score", "new_coverage_score", "activity_score", "direction_score", "structure_score", "branch_urgency_score", "turn_penalty", "repeat_penalty", "dead_zone_penalty", "obstacle_penalty"]},
        }

    def _candidate_branches(self, decision: StepDecision) -> list[dict]:
        details = decision.details or {}
        if "candidate_branches" in details:
            return details["candidate_branches"]
        if "candidates" in details:
            return details["candidates"]
        if decision.branch:
            return [
                {
                    "type": decision.escape_type if decision.mode == "escape" else decision.mode,
                    "score": details.get("branch_score"),
                    "path": [[int(x), int(y)] for x, y in decision.branch],
                }
            ]
        return []

    def _candidate_tree(self, decision: StepDecision) -> dict:
        details = decision.details or {}
        if "candidate_tree" in details:
            return details["candidate_tree"]
        branches = self._candidate_branches(decision)
        return {"levels": [{"depth": 1, "branches": branches}]} if branches else {"levels": []}
