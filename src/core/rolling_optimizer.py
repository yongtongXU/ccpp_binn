from __future__ import annotations

from dataclasses import dataclass

from src.core.cell_map import Cell, CellMap
from src.core.gbnn_field import GBNNField
from src.core.usv import USV


@dataclass
class BranchState:
    branch: list[Cell]
    score: float
    details: dict[str, float]


class RollingOptimizer:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._feature_cache: dict[Cell, dict] = {}
        self._last_candidate_tree: dict = {"levels": []}

    def select_next_cell(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> tuple[Cell | None, list[Cell], dict]:
        self._feature_cache = {}
        candidate_limit = int(self.config.get("record_candidate_count", self.config.get("beam_width", 30)))
        if self.config.get("enabled", True) is False:
            for n in cell_map.neighbors8(usv.current_cell):
                if cell_map.is_uncovered(n):
                    branch = [n]
                    details = self.score_branch(usv, cell_map, gbnn_field, branch)
                    details["candidate_branches"] = _serialize_candidates(self._local_candidate_states(usv, cell_map, gbnn_field), candidate_limit)
                    details["candidate_tree"] = _serialize_tree([self._local_candidate_states(usv, cell_map, gbnn_field)], candidate_limit)
                    return n, branch, details
            return None, [], {"reason": "rolling_disabled_no_uncovered_neighbor"}
        candidates = self.build_candidate_tree(usv, cell_map, gbnn_field)
        if not candidates:
            strip_step = self._direct_strip_step(usv, cell_map)
            if strip_step is not None:
                branch = [strip_step]
                details = self.score_branch(usv, cell_map, gbnn_field, branch)
                details["candidate_branches"] = _serialize_candidates(self._local_candidate_states(usv, cell_map, gbnn_field), candidate_limit)
                details["candidate_tree"] = _serialize_tree([self._local_candidate_states(usv, cell_map, gbnn_field)], candidate_limit)
                return strip_step, branch, details
            if self.current_strip_has_forward_uncovered(usv, cell_map):
                fallback = self._local_strip_fallback(usv, cell_map, gbnn_field)
                if fallback:
                    details = self.score_branch(usv, cell_map, gbnn_field, fallback)
                    details["candidate_branches"] = _serialize_candidates(
                        [BranchState(fallback, float(details["branch_score"]), details)], candidate_limit
                    )
                    details["candidate_tree"] = _serialize_tree([[BranchState(fallback, float(details["branch_score"]), details)]], candidate_limit)
                    return fallback[0], fallback, details
            return None, [], {"reason": "no_candidate_branch"}
        best = max(candidates, key=lambda b: b.score)
        best.details["candidate_branches"] = _serialize_candidates(candidates, candidate_limit)
        best.details["candidate_tree"] = self._last_candidate_tree
        if best.details.get("new_coverage_score", 0.0) <= 0.0 and not self.current_strip_has_forward_uncovered(usv, cell_map):
            return None, [], {"reason": "no_strip_new_coverage_candidate"}
        if best.score <= -1e8:
            if self.current_strip_has_forward_uncovered(usv, cell_map):
                fallback = self._local_strip_fallback(usv, cell_map, gbnn_field)
                if fallback:
                    details = self.score_branch(usv, cell_map, gbnn_field, fallback)
                    details["candidate_branches"] = _serialize_candidates(
                        [BranchState(fallback, float(details["branch_score"]), details)], candidate_limit
                    )
                    details["candidate_tree"] = _serialize_tree([[BranchState(fallback, float(details["branch_score"]), details)]], candidate_limit)
                    return fallback[0], fallback, details
            return None, [], {"reason": "invalid_candidate_score"}
        return best.branch[0], best.branch, best.details

    def _direct_strip_step(self, usv: USV, cell_map: CellMap) -> Cell | None:
        current_strip = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if usv.current_cell[1] != current_strip:
            return None
        direction = 1 if usv.strip_direction >= 0 else -1
        if not self.current_strip_has_forward_uncovered(usv, cell_map):
            for dy in (1, -1):
                adjacent_strip = current_strip + dy
                candidate = (usv.current_cell[0], adjacent_strip)
                if 0 <= adjacent_strip < cell_map.height and cell_map.is_traversable(candidate):
                    if cell_map.is_uncovered(candidate):
                        return candidate
            return None
        front = (usv.current_cell[0] + direction, current_strip)
        if cell_map.is_traversable(front):
            return front
        return None

    def _local_candidate_states(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> list[BranchState]:
        candidates: list[BranchState] = []
        for n in cell_map.neighbors8(usv.current_cell):
            branch = [n]
            details = self.score_branch(usv, cell_map, gbnn_field, branch)
            candidates.append(BranchState(branch, float(details["branch_score"]), details))
        return candidates

    def build_candidate_tree(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> list[BranchState]:
        horizon = int(self.config.get("horizon", 5))
        tree_limit = int(self.config.get("record_tree_count", self.config.get("record_candidate_count", 512)))
        levels: list[list[BranchState]] = [[] for _ in range(horizon)]
        finished: list[BranchState] = []

        def dfs(branch: list[Cell]) -> None:
            if len(branch) == horizon:
                if branch:
                    details = self.score_branch(usv, cell_map, gbnn_field, branch)
                    finished.append(BranchState(branch.copy(), float(details["branch_score"]), details))
                return
            root = branch[-1] if branch else usv.current_cell
            for n in cell_map.neighbors8(root):
                new_branch = branch + [n]
                details = self.score_branch(usv, cell_map, gbnn_field, new_branch)
                levels[len(new_branch) - 1].append(BranchState(new_branch.copy(), float(details["branch_score"]), details))
                dfs(new_branch)

        dfs([])
        finished.sort(key=lambda b: b.score, reverse=True)
        for level in levels:
            level.sort(key=lambda b: b.score, reverse=True)
        self._last_candidate_tree = _serialize_tree(levels, tree_limit)
        return finished

    def _allowed_next(self, usv: USV, cell_map: CellMap, branch: list[Cell], cell: Cell) -> bool:
        return True

    def score_branch(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField, branch: list[Cell]) -> dict:
        cfg = self.config
        virtual_seen: set[Cell] = set()
        new_count = 0
        activity = 0.0
        repeat = 0
        turns = 0
        dead_zone = 0.0
        obstacle = 0.0
        meaningless = 0
        strip_forward = 0.0
        strip_transition = 0.0
        strip_reverse_penalty = 0.0
        strip_cross_penalty = 0.0
        strip_loop_penalty = 0.0
        immediate_backtrack_penalty = 0.0
        prev = usv.current_cell
        prev_heading = usv.heading
        current_strip = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        strip_direction = 1 if usv.strip_direction >= 0 else -1
        preferred_direction = strip_direction
        current_has_forward = self.current_strip_has_forward_uncovered(usv, cell_map)
        current_has_any = self.current_strip_has_any_uncovered(usv, cell_map)
        direction_score = 0.0
        structure_score = 0.0
        for i, cell in enumerate(branch):
            if i == 0 and len(usv.path) >= 2 and cell == usv.path[-2]:
                immediate_backtrack_penalty += 1.0
            elif i >= 2 and cell == branch[i - 2]:
                immediate_backtrack_penalty += 1.0
            features = self._cell_features(cell_map, gbnn_field, cell)
            strip_delta = cell[1] - current_strip
            abs_strip_delta = abs(strip_delta)
            dx = cell[0] - prev[0]
            if abs_strip_delta == 0:
                if dx * preferred_direction > 0:
                    strip_forward += 1.0
                    if features["uncovered"]:
                        strip_forward += 0.8
                elif dx * preferred_direction < 0:
                    strip_reverse_penalty += 1.0
                elif features["uncovered"]:
                    strip_forward += 0.2
            elif abs_strip_delta == 1:
                if current_has_forward:
                    strip_cross_penalty += 3.0
                else:
                    strip_transition += 1.0
                    if features["uncovered"]:
                        strip_transition += 0.6
                if dx * strip_direction < 0:
                    strip_reverse_penalty += 0.5
            else:
                strip_cross_penalty += 12.0 * abs_strip_delta
            if features["uncovered"] and cell not in virtual_seen:
                new_count += 1
                virtual_seen.add(cell)
            elif cell in virtual_seen or features["visited"]:
                repeat += 1
            activity += features["activity"]
            heading = _heading_between(prev, cell)
            if prev_heading is not None and heading is not None:
                delta = _heading_delta(prev_heading, heading)
                if delta == 0:
                    direction_score += 1.0
                elif delta <= 1:
                    direction_score += 0.4
                else:
                    turns += 1
            if i >= 2 and branch[i] == branch[i - 2]:
                meaningless += 1
            if branch.count(cell) > 1:
                strip_loop_penalty += 2.0
            structure_score += features["structure"]
            dead_zone += features["dead_zone"]
            obstacle += features["obstacle"]
            prev = cell
            prev_heading = heading
        new_coverage_score = cfg.get("w_new_coverage", 8.0) * new_count
        activity_score = cfg.get("w_activity", 1.0) * activity / max(1, len(branch))
        direction_term = cfg.get("w_direction", 4.0) * direction_score / max(1, len(branch))
        structure_term = cfg.get("w_structure", 3.0) * structure_score / max(1, len(branch))
        turn_penalty = cfg.get("w_turn", 3.0) * turns
        repeat_penalty = cfg.get("w_repeat", 2.0) * repeat
        dead_zone_penalty = cfg.get("w_dead_zone", 5.0) * dead_zone / max(1, len(branch))
        obstacle_penalty = cfg.get("w_obstacle", 1.5) * obstacle / max(1, len(branch))
        loop_penalty = 5.0 * meaningless
        strip_forward_score = cfg.get("w_strip_forward", 18.0) * strip_forward
        strip_transition_score = cfg.get("w_strip_transition", 14.0) * strip_transition
        strip_reverse_term = cfg.get("w_strip_reverse", 25.0) * strip_reverse_penalty
        strip_cross_term = cfg.get("w_strip_cross", 35.0) * strip_cross_penalty
        strip_loop_term = cfg.get("w_strip_loop", 20.0) * strip_loop_penalty
        immediate_backtrack_term = cfg.get("w_immediate_backtrack", 1000.0) * immediate_backtrack_penalty
        score = (
            new_coverage_score
            + activity_score
            + direction_term
            + structure_term
            + strip_forward_score
            + strip_transition_score
            - turn_penalty
            - repeat_penalty
            - dead_zone_penalty
            - obstacle_penalty
            - loop_penalty
            - strip_reverse_term
            - strip_cross_term
            - strip_loop_term
            - immediate_backtrack_term
        )
        return {
            "branch_score": float(score),
            "new_coverage_score": float(new_coverage_score),
            "activity_score": float(activity_score),
            "direction_score": float(direction_term),
            "structure_score": float(structure_term),
            "turn_penalty": float(turn_penalty),
            "repeat_penalty": float(repeat_penalty),
            "dead_zone_penalty": float(dead_zone_penalty),
            "obstacle_penalty": float(obstacle_penalty),
            "strip_forward_score": float(strip_forward_score),
            "strip_transition_score": float(strip_transition_score),
            "strip_reverse_penalty": float(strip_reverse_term),
            "strip_cross_penalty": float(strip_cross_term),
            "strip_loop_penalty": float(strip_loop_term),
            "immediate_backtrack_penalty": float(immediate_backtrack_term),
        }

    def current_strip_has_forward_uncovered(self, usv: USV, cell_map: CellMap) -> bool:
        y = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if y < 0 or y >= cell_map.height:
            return False
        direction = 1 if usv.strip_direction >= 0 else -1
        x = usv.current_cell[0] + direction
        while 0 <= x < cell_map.width:
            if not cell_map.is_traversable((x, y)):
                return False
            if cell_map.is_uncovered((x, y)):
                return True
            x += direction
        return False

    def current_strip_has_any_uncovered(self, usv: USV, cell_map: CellMap) -> bool:
        y = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if y < 0 or y >= cell_map.height:
            return False
        for x in range(cell_map.width):
            if cell_map.is_uncovered((x, y)):
                return True
        return False

    def current_strip_uncovered_direction(self, usv: USV, cell_map: CellMap) -> int | None:
        y = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if y < 0 or y >= cell_map.height:
            return None
        xs = [x for x in range(cell_map.width) if cell_map.is_uncovered((x, y))]
        if not xs:
            return None
        current_x = usv.current_cell[0]
        nearest = min(xs, key=lambda x: abs(x - current_x))
        if nearest == current_x:
            return 1 if usv.strip_direction >= 0 else -1
        return 1 if nearest > current_x else -1

    def current_or_adjacent_strip_has_continuation(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> bool:
        self._feature_cache = {}
        current_strip = usv.current_strip_id if usv.current_strip_id is not None else usv.current_cell[1]
        if self.current_strip_has_forward_uncovered(usv, cell_map):
            return True
        for candidate in self.build_candidate_tree(usv, cell_map, gbnn_field):
            for cell in candidate.branch:
                if abs(cell[1] - current_strip) <= 1 and cell_map.is_uncovered(cell):
                    return True
        return False

    def _local_strip_fallback(self, usv: USV, cell_map: CellMap, gbnn_field: GBNNField) -> list[Cell] | None:
        neighbors = cell_map.neighbors8(usv.current_cell)
        if not neighbors:
            return None
        scored = []
        for n in neighbors:
            branch = [n]
            details = self.score_branch(usv, cell_map, gbnn_field, branch)
            scored.append((details["branch_score"], n))
        scored.sort(reverse=True)
        return [scored[0][1]] if scored else None

    def estimate_dead_zone_risk(self, cell_map: CellMap, cell: Cell) -> float:
        ns = cell_map.neighbors8(cell)
        if not ns:
            return 1.0
        uncovered = sum(1 for n in ns if cell_map.is_uncovered(n))
        return 1.0 / (1.0 + uncovered)

    def _obstacle_risk(self, cell_map: CellMap, cell: Cell) -> float:
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

    def _local_structure_score(self, cell_map: CellMap, cell: Cell) -> float:
        ns = cell_map.neighbors8(cell)
        if not ns:
            return -1.0
        uncovered = sum(1 for n in ns if cell_map.is_uncovered(n))
        return min(uncovered, 4) / 4.0

    def _cell_features(self, cell_map: CellMap, gbnn_field: GBNNField, cell: Cell) -> dict:
        cached = self._feature_cache.get(cell)
        if cached is not None:
            return cached
        ns = cell_map.neighbors8(cell)
        uncovered_neighbors = 0
        obstacle_neighbors = 0
        x, y = cell
        for n in ns:
            if cell_map.grid[n[1], n[0]] == 0:
                uncovered_neighbors += 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if not cell_map.in_bounds(n) or cell_map.grid[n[1], n[0]] == 2:
                    obstacle_neighbors += 1
        features = {
            "uncovered": bool(cell_map.grid[y, x] == 0),
            "visited": bool(cell_map.visit_count[y, x] > 0),
            "activity": max(0.0, gbnn_field.get_activity(cell)),
            "structure": min(uncovered_neighbors, 4) / 4.0 if ns else -1.0,
            "dead_zone": 1.0 / (1.0 + uncovered_neighbors) if ns else 1.0,
            "obstacle": obstacle_neighbors / 8.0,
        }
        self._feature_cache[cell] = features
        return features


def _heading_between(a: Cell, b: Cell) -> int | None:
    dx = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
    dy = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
    if dx == 0 and dy == 0:
        return None
    dirs = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
    return dirs[(dx, dy)]


def _heading_delta(a: int, b: int) -> int:
    diff = abs(a - b) % 8
    return min(diff, 8 - diff)


def _serialize_candidates(candidates: list[BranchState], limit: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda b: b.score, reverse=True)[: max(1, limit)]
    return [
        {
            "type": "rolling",
            "score": float(candidate.score),
            "path": [[int(x), int(y)] for x, y in candidate.branch],
        }
        for candidate in ordered
    ]


def _serialize_tree(levels: list[list[BranchState]], limit: int) -> dict:
    return {
        "levels": [
            {
                "depth": depth,
                "branches": _serialize_candidates(level, limit),
            }
            for depth, level in enumerate(levels, start=1)
        ]
    }
