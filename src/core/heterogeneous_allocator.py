from __future__ import annotations

from src.core.segment_planner import SegmentPlanner


class HeterogeneousAllocator:
    def __init__(self, grid_map, balance_weight: float = 1.0):
        self.grid_map = grid_map
        self.balance_weight = balance_weight

    def allocate(self, segments, usvs) -> dict[str, list]:
        assignments = {u.id: [] for u in usvs}
        if not segments or not usvs:
            return assignments

        lines = []
        for line_index in sorted({s.line_index for s in segments}):
            line_segments = [s for s in segments if s.line_index == line_index]
            line_length = sum(s.length for s in line_segments)
            lines.append((line_index, line_segments, line_length))

        total_capacity = sum(max(u.max_speed * u.coverage_width, 1e-6) for u in usvs)
        total_length = sum(item[2] for item in lines)
        targets = {
            u.id: total_length * (max(u.max_speed * u.coverage_width, 1e-6) / total_capacity)
            for u in usvs
        }

        usv_order = sorted(usvs, key=lambda u: (-u.max_speed * u.coverage_width, u.id))
        usv_idx = 0
        current_load = 0.0
        for line_index, line_segments, line_length in lines:
            usv = usv_order[min(usv_idx, len(usv_order) - 1)]
            if (
                usv_idx < len(usv_order) - 1
                and current_load >= targets[usv.id] * 0.85
                and assignments[usv.id]
            ):
                usv_idx += 1
                usv = usv_order[usv_idx]
                current_load = 0.0
            for seg in line_segments:
                seg.assigned_usv = usv.id
                assignments[usv.id].append(seg)
            current_load += line_length
        return assignments

    def build_paths(self, assignments, usvs, allow_astar=True, cost_weights=None):
        paths = {}
        for usv in usvs:
            planner = SegmentPlanner(self.grid_map, allow_astar, cost_weights)
            planned = planner.plan(assignments.get(usv.id, []), start=usv.start)
            paths[usv.id] = planned
        return paths
