from __future__ import annotations

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
        target = float(planner_cfg.get("target_coverage_rate", 1.0))
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
            if decision.mode == "normal":
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
                "method",
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

    def _record_decision(self, step: int, decision: StepDecision, selected: tuple[int, int]) -> None:
        cur = self.usv.path[-2] if len(self.usv.path) >= 2 else self.usv.current_cell
        branch = decision.branch
        details = decision.details
        self.decision_rows.append(
            {
                "step": step,
                "current_x": cur[0],
                "current_y": cur[1],
                "selected_x": selected[0],
                "selected_y": selected[1],
                "method": self.strategy.name,
                "selected_branch": ";".join(f"{x}:{y}" for x, y in branch),
                **{k: details.get(k) for k in ["branch_score", "new_coverage_score", "activity_score", "direction_score", "structure_score", "turn_penalty", "repeat_penalty", "dead_zone_penalty", "obstacle_penalty"]},
            }
        )
