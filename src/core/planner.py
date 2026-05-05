from __future__ import annotations

from dataclasses import dataclass

from src.core.coverage_state import CoverageState
from src.core.gbnn_field import GBNNField
from src.core.grid_map import GridMap
from src.core.heterogeneous_allocator import HeterogeneousAllocator
from src.core.metrics import compute_metrics
from src.core.residual_recovery import ResidualRecovery
from src.core.segment_planner import SegmentPlanner
from src.core.sweep_generator import SweepGenerator
from src.core.usv import USV


@dataclass
class PlanResult:
    grid_map: GridMap
    coverage_state: CoverageState
    usvs: list[USV]
    segments: list
    paths_by_usv: dict[str, list[tuple[float, float]]]
    modes_by_usv: dict[str, list[str]]
    best_scan_angle: float
    gbnn_activity: object | None
    metrics: dict
    recovery_cells: list[tuple[int, int]]


class CoveragePlanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.grid_map = GridMap.from_config(cfg)
        self.usvs = [USV.from_config(u) for u in cfg.get("usvs", [{"id": "USV1", "start": [1.5, 1.5]}])]
        self.mode = "cluster" if cfg.get("cluster", {}).get("enabled", False) or len(self.usvs) > 1 else "single"

    def run(self, gbnn_enabled: bool = True) -> PlanResult:
        if self.mode == "cluster":
            return self._run_cluster(gbnn_enabled)
        return self._run_single(gbnn_enabled)

    def _select_best_angle(self, coverage_width: float, start_cell) -> tuple[float, list, object]:
        best = None
        planning = self.cfg.get("planning", {})
        sweep_cfg = self.cfg.get("sweep", {})
        for angle in planning.get("scan_angles", [0, 30, 45, 60, 90, 120, 135, 150]):
            generator = SweepGenerator(
                self.grid_map,
                coverage_width=coverage_width,
                scan_angle=angle,
                min_segment_length=int(sweep_cfg.get("min_segment_length", 3)),
            )
            segments = generator.generate()
            planner = SegmentPlanner(
                self.grid_map,
                allow_astar_connection=planning.get("allow_astar_connection", True),
                cost_weights=self.cfg.get("cost", {}),
            )
            planned = planner.plan(segments, start=self.grid_map.grid_to_world(start_cell))
            if best is None or planned.cost < best[2].cost:
                best = (angle, segments, planned)
        return best

    def _run_single(self, gbnn_enabled: bool) -> PlanResult:
        usv = self.usvs[0]
        coverage_width = usv.coverage_width or self.cfg.get("sweep", {}).get("coverage_width", 4)
        start_cell = self.grid_map.world_to_grid(usv.start)
        angle, segments, planned = self._select_best_angle(coverage_width, start_cell)

        state = CoverageState(self.grid_map)
        main_world = planned.points
        state.update_by_path(main_world, usv.coverage_radius)
        residual_before = len(state.residual_components(self.cfg.get("planning", {}).get("min_residual_size", 3)))
        activity = self._compute_gbnn(state, gbnn_enabled)

        end_cell = self.grid_map.world_to_grid(planned.points[-1]) if planned.points else start_cell
        recovery = ResidualRecovery(
            self.grid_map,
            state,
            min_residual_size=self.cfg.get("planning", {}).get("min_residual_size", 3),
            coverage_width=coverage_width,
        ).recover(end_cell, activity, usv.coverage_radius)
        recovery_world = [self.grid_map.grid_to_world(c) for c in recovery.cells]
        state.update_by_path(recovery_world, usv.coverage_radius)

        full_path = main_world + recovery_world
        modes = planned.modes + recovery.modes
        metrics = compute_metrics(
            self.grid_map,
            state,
            {usv.id: full_path},
            len(segments),
            planned.astar_connection_count,
            residual_before,
            len(state.residual_components(self.cfg.get("planning", {}).get("min_residual_size", 3))),
            recovery.recovery_path_length,
            self.cfg.get("scenario_name", "scenario"),
            "single",
            angle,
        )
        return PlanResult(self.grid_map, state, self.usvs, segments, {usv.id: full_path}, {usv.id: modes}, angle, activity, metrics, recovery.cells)

    def _run_cluster(self, gbnn_enabled: bool) -> PlanResult:
        widest = max(u.coverage_width for u in self.usvs)
        start_cell = self.grid_map.world_to_grid(self.usvs[0].start)
        angle, segments, _ = self._select_best_angle(widest, start_cell)
        allocator = HeterogeneousAllocator(self.grid_map, self.cfg.get("cluster", {}).get("balance_weight", 1.0))
        assignments = allocator.allocate(segments, self.usvs)
        planned_by_usv = allocator.build_paths(
            assignments,
            self.usvs,
            allow_astar=self.cfg.get("planning", {}).get("allow_astar_connection", True),
            cost_weights=self.cfg.get("cost", {}),
        )

        state = CoverageState(self.grid_map)
        paths_by_usv = {}
        modes_by_usv = {}
        astar_count = 0
        for usv in self.usvs:
            planned = planned_by_usv[usv.id]
            world = planned.points
            state.update_by_path(world, usv.coverage_radius)
            paths_by_usv[usv.id] = world
            modes_by_usv[usv.id] = planned.modes
            astar_count += planned.astar_connection_count

        min_size = self.cfg.get("planning", {}).get("min_residual_size", 3)
        residual_before = len(state.residual_components(min_size))
        activity = self._compute_gbnn(state, gbnn_enabled)
        recovery_cells = []
        recovery_length = 0.0
        for usv in self.usvs:
            if not state.residual_components(min_size):
                break
            current_world = paths_by_usv[usv.id][-1] if paths_by_usv[usv.id] else usv.start
            current = self.grid_map.world_to_grid(current_world)
            recovery = ResidualRecovery(self.grid_map, state, min_size, usv.coverage_width).recover(current, activity, usv.coverage_radius)
            world = [self.grid_map.grid_to_world(c) for c in recovery.cells]
            state.update_by_path(world, usv.coverage_radius)
            paths_by_usv[usv.id].extend(world)
            modes_by_usv[usv.id].extend(recovery.modes)
            recovery_cells.extend(recovery.cells)
            recovery_length += recovery.recovery_path_length

        metrics = compute_metrics(
            self.grid_map,
            state,
            paths_by_usv,
            len(segments),
            astar_count,
            residual_before,
            len(state.residual_components(min_size)),
            recovery_length,
            self.cfg.get("scenario_name", "scenario"),
            "cluster",
            angle,
        )
        return PlanResult(self.grid_map, state, self.usvs, segments, paths_by_usv, modes_by_usv, angle, activity, metrics, recovery_cells)

    def _compute_gbnn(self, state: CoverageState, enabled: bool):
        if not enabled or not self.cfg.get("gbnn", {}).get("enabled", True):
            return None
        g = self.cfg.get("gbnn", {})
        residual = state.residual_components(self.cfg.get("planning", {}).get("min_residual_size", 3))
        return GBNNField(
            self.grid_map,
            state,
            iterations=int(g.get("iterations", 5)),
            excitation_uncovered=float(g.get("excitation_uncovered", 1.0)),
            inhibition_covered=float(g.get("inhibition_covered", 0.2)),
            inhibition_obstacle=float(g.get("inhibition_obstacle", 2.0)),
            residual_boost=float(g.get("residual_boost", 1.5)),
            neighbor_weight=float(g.get("neighbor_weight", 0.2)),
        ).compute(residual)
