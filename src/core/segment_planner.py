from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.core.sweep_generator import CoverageSegment
from src.utils.astar import astar
from src.utils.geometry import Cell, Point, count_turns, cumulative_heading_change, euclidean, path_length


@dataclass
class PlannedPath:
    points: list[Point]
    modes: list[str]
    ordered_segments: list[CoverageSegment]
    astar_connection_count: int = 0
    connection_length: float = 0.0
    cost: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def cells(self) -> list[Cell]:
        return []


class SegmentPlanner:
    def __init__(self, grid_map, allow_astar_connection: bool = True, cost_weights: dict | None = None):
        self.grid_map = grid_map
        self.allow_astar_connection = allow_astar_connection
        self.cost_weights = cost_weights or {}

    def plan(self, segments: Sequence[CoverageSegment], start: Point | Cell | None = None, estimate_residual: int = 0) -> PlannedPath:
        if not segments:
            return PlannedPath([], [], [], cost=float("inf"))
        start_point = self._as_point(start) if start is not None else segments[0].start_point
        current = start_point
        path: list[Point] = [current]
        modes: list[str] = ["connection"]
        ordered: list[CoverageSegment] = []
        astar_count = 0
        connection_len = 0.0

        for line_order, line_index in enumerate(sorted({s.line_index for s in segments})):
            line_segments = sorted(
                [s for s in segments if s.line_index == line_index],
                key=lambda s: min(s.along_start, s.along_end),
            )
            line_forward = line_order % 2 == 0
            while line_segments:
                seg, forward = self._choose_interval(current, line_segments, line_forward)
                seg_points = seg.oriented_points(forward)
                connector, used_astar = self._connect(current, seg_points[0])
                if connector:
                    connection_len += path_length(connector)
                    if used_astar:
                        astar_count += 1
                    path.extend(connector[1:] if _same_point(path[-1], connector[0]) else connector)
                    modes.extend(["connection"] * (len(path) - len(modes)))
                path.extend(seg_points[1:] if _same_point(path[-1], seg_points[0]) else seg_points)
                modes.extend(["main_sweep"] * (len(path) - len(modes)))
                current = path[-1]
                ordered.append(seg)
                line_segments.remove(seg)

        path, modes = _dedupe_points_with_modes(path, modes)
        metrics = {
            "path_length": path_length(path),
            "turns": count_turns(path),
            "cumulative_heading_change": cumulative_heading_change(path),
            "connection_length": connection_len,
            "astar_connection_count": astar_count,
        }
        cost = self.evaluate_cost(metrics, estimate_residual, segments)
        return PlannedPath(path, modes, ordered, astar_count, connection_len, cost, metrics)

    def evaluate_cost(self, metrics: dict, residual_count: int, segments: Sequence[CoverageSegment]) -> float:
        w = self.cost_weights
        obstacle_penalty = sum(s.nearby_obstacle_score for s in segments)
        return (
            w.get("w_path_length", 1.0) * metrics.get("path_length", 0.0)
            + w.get("w_turns", 2.0) * metrics.get("turns", 0)
            + w.get("w_connection", 1.5) * metrics.get("connection_length", 0.0)
            + w.get("w_astar", 3.0) * metrics.get("astar_connection_count", 0)
            + w.get("w_residual", 4.0) * residual_count
            + w.get("w_obstacle", 1.0) * obstacle_penalty
        )

    def _choose_interval(self, current: Point, candidates: list[CoverageSegment], line_forward: bool) -> tuple[CoverageSegment, bool]:
        best: tuple[CoverageSegment, bool] | None = None
        best_cost = float("inf")
        for seg in candidates:
            options = [(seg.start_point, True), (seg.end_point, False)]
            for entry, forward in options:
                orientation_penalty = 0.25 if forward != line_forward else 0.0
                cost = euclidean(current, entry) + orientation_penalty
                if cost < best_cost:
                    best_cost = cost
                    best = (seg, forward)
        return best

    def _connect(self, start: Point, goal: Point) -> tuple[list[Point], bool]:
        if _same_point(start, goal):
            return [start], False
        start_cell = self.grid_map.world_to_grid(start)
        goal_cell = self.grid_map.world_to_grid(goal)
        if self.grid_map.line_is_free(start, goal, world=True):
            return _straight_samples(start, goal, self.grid_map.resolution), False
        if self.allow_astar_connection:
            cell_path = astar(self.grid_map, start_cell, goal_cell)
            if cell_path:
                return [self.grid_map.grid_to_world(c) for c in cell_path], True
        return _straight_samples(start, goal, self.grid_map.resolution), False

    def _as_point(self, value: Point | Cell) -> Point:
        if isinstance(value[0], int):
            return self.grid_map.grid_to_world(value)
        return (float(value[0]), float(value[1]))


def _same_point(a: Point, b: Point, eps: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _straight_samples(a: Point, b: Point, step: float) -> list[Point]:
    dist = euclidean(a, b)
    if dist <= 1e-12:
        return [a]
    n = max(1, int(dist / max(step, 1e-6)))
    return [
        (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        for i in range(n + 1)
    ]


def _dedupe_points_with_modes(points: list[Point], modes: list[str]) -> tuple[list[Point], list[str]]:
    if not points:
        return [], []
    new_points = [points[0]]
    new_modes = [modes[0] if modes else "connection"]
    for i, point in enumerate(points[1:], start=1):
        if _same_point(point, new_points[-1]):
            continue
        new_points.append(point)
        new_modes.append(modes[i] if i < len(modes) else new_modes[-1])
    return new_points, new_modes
