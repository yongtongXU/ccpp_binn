from __future__ import annotations

from dataclasses import dataclass
import math

from src.utils.geometry import Cell, Point, euclidean, path_length


@dataclass
class CoverageSegment:
    id: int
    line_index: int
    interval_index: int
    start_point: Point
    end_point: Point
    sampled_points: list[Point]
    sampled_cells: list[Cell]
    length: float
    direction: tuple[float, float]
    nearby_obstacle_score: float = 0.0
    boundary_score: float = 0.0
    assigned_usv: str | None = None
    along_start: float = 0.0
    along_end: float = 0.0

    @property
    def start_cell(self) -> Cell:
        return self.sampled_cells[0]

    @property
    def end_cell(self) -> Cell:
        return self.sampled_cells[-1]

    @property
    def cells(self) -> list[Cell]:
        return self.sampled_cells

    def oriented_points(self, forward: bool = True) -> list[Point]:
        return self.sampled_points if forward else list(reversed(self.sampled_points))

    def oriented_cells(self, forward: bool = True) -> list[Cell]:
        return self.sampled_cells if forward else list(reversed(self.sampled_cells))


class SweepGenerator:
    def __init__(
        self,
        grid_map,
        coverage_width: float,
        scan_angle: float,
        min_segment_length: int = 3,
        sample_step: float | None = None,
    ):
        self.grid_map = grid_map
        self.coverage_width = float(coverage_width)
        self.scan_angle = scan_angle
        self.min_segment_length = min_segment_length
        self.sample_step = sample_step or max(grid_map.resolution * 0.5, 0.5)
        theta = math.radians(scan_angle)
        self.u = (math.cos(theta), math.sin(theta))
        self.v = (-math.sin(theta), math.cos(theta))

    def generate(self) -> list[CoverageSegment]:
        min_along, max_along, min_cross, max_cross = self._map_projection_bounds()
        margin = max(self.grid_map.width, self.grid_map.height) * self.grid_map.resolution
        min_along -= margin
        max_along += margin

        segments: list[CoverageSegment] = []
        seg_id = 0
        line_index = 0
        crosses = []
        start_cross = min_cross + self.coverage_width / 2.0
        end_cross = max_cross - self.coverage_width / 2.0
        cross = start_cross
        while cross <= end_cross + 1e-9:
            crosses.append(cross)
            cross += self.coverage_width
        if end_cross > min_cross and (not crosses or end_cross - crosses[-1] > self.grid_map.resolution):
            crosses.append(end_cross)

        for cross in crosses:
            samples = self._sample_line(cross, min_along, max_along)
            intervals = self._free_intervals(samples)
            for interval_index, interval in enumerate(intervals):
                segment = self._make_segment(seg_id, line_index, interval_index, interval)
                if segment is None:
                    continue
                segments.append(segment)
                seg_id += 1
            line_index += 1
        return segments

    def _map_projection_bounds(self) -> tuple[float, float, float, float]:
        w = self.grid_map.width * self.grid_map.resolution
        h = self.grid_map.height * self.grid_map.resolution
        corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        along = [self._along(p) for p in corners]
        cross = [self._cross(p) for p in corners]
        return min(along), max(along), min(cross), max(cross)

    def _sample_line(self, cross: float, min_along: float, max_along: float) -> list[tuple[float, Point, Cell, bool]]:
        samples = []
        t = min_along
        while t <= max_along + 1e-9:
            point = (t * self.u[0] + cross * self.v[0], t * self.u[1] + cross * self.v[1])
            cell = self.grid_map.world_to_grid(point)
            in_world = 0 <= point[0] < self.grid_map.width * self.grid_map.resolution and 0 <= point[1] < self.grid_map.height * self.grid_map.resolution
            is_free = in_world and self.grid_map.is_free_cell(cell)
            samples.append((t, point, cell, is_free))
            t += self.sample_step
        return samples

    def _free_intervals(self, samples: list[tuple[float, Point, Cell, bool]]) -> list[list[tuple[float, Point, Cell]]]:
        intervals: list[list[tuple[float, Point, Cell]]] = []
        current: list[tuple[float, Point, Cell]] = []
        last_cell: Cell | None = None
        for along, point, cell, is_free in samples:
            same_cell_repeat = last_cell == cell
            last_cell = cell
            if is_free:
                if not same_cell_repeat:
                    current.append((along, point, cell))
            elif current:
                intervals.append(current)
                current = []
        if current:
            intervals.append(current)
        return intervals

    def _make_segment(
        self,
        seg_id: int,
        line_index: int,
        interval_index: int,
        interval: list[tuple[float, Point, Cell]],
    ) -> CoverageSegment | None:
        if len(interval) < self.min_segment_length:
            return None
        along_values = [item[0] for item in interval]
        points = [item[1] for item in interval]
        cells = [item[2] for item in interval]
        length = path_length(points)
        obstacle_score, boundary_score = self._scores(cells)
        return CoverageSegment(
            id=seg_id,
            line_index=line_index,
            interval_index=interval_index,
            start_point=points[0],
            end_point=points[-1],
            sampled_points=points,
            sampled_cells=cells,
            length=length,
            direction=self.u,
            nearby_obstacle_score=obstacle_score,
            boundary_score=boundary_score,
            along_start=along_values[0],
            along_end=along_values[-1],
        )

    def _scores(self, cells: list[Cell]) -> tuple[float, float]:
        obstacle_score = 0.0
        boundary_score = 0.0
        for cell in cells:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nb = (cell[0] + dr, cell[1] + dc)
                    if not self.grid_map.in_bounds(nb) or self.grid_map.grid[nb] == 2:
                        boundary_score += 1.0
                    elif self.grid_map.grid[nb] == 1:
                        obstacle_score += 1.0
        norm = max(1, len(cells))
        return obstacle_score / norm, boundary_score / norm

    def _along(self, point: Point) -> float:
        return point[0] * self.u[0] + point[1] * self.u[1]

    def _cross(self, point: Point) -> float:
        return point[0] * self.v[0] + point[1] * self.v[1]
