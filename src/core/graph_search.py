from __future__ import annotations

import heapq
import math
from typing import Callable

from src.core.cell_map import Cell, CellMap

CostFn = Callable[[Cell, Cell], float]
TargetCondition = Callable[[Cell], bool]


def move_cost(a: Cell, b: Cell) -> float:
    return math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0


def reconstruct_path(came_from: dict[Cell, Cell], end: Cell) -> list[Cell]:
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def dijkstra(
    cell_map: CellMap,
    start: Cell,
    goal: Cell | None = None,
    target_condition: TargetCondition | None = None,
    cost_fn: CostFn | None = None,
    max_expansion: int | None = None,
) -> tuple[list[Cell] | None, float, dict]:
    if not cell_map.is_traversable(start):
        return None, math.inf, {"expanded": 0}
    cost_fn = cost_fn or move_cost
    pq: list[tuple[float, Cell]] = [(0.0, start)]
    dist: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    expanded = 0
    while pq:
        cost, cell = heapq.heappop(pq)
        if cost > dist[cell]:
            continue
        expanded += 1
        if goal is not None and cell == goal:
            return reconstruct_path(came_from, cell), cost, {"expanded": expanded, "dist": dist}
        if target_condition is not None and cell != start and target_condition(cell):
            return reconstruct_path(came_from, cell), cost, {"expanded": expanded, "dist": dist}
        if max_expansion is not None and expanded >= max_expansion:
            break
        for n in cell_map.neighbors8(cell):
            new_cost = cost + float(cost_fn(cell, n))
            if new_cost < dist.get(n, math.inf):
                dist[n] = new_cost
                came_from[n] = cell
                heapq.heappush(pq, (new_cost, n))
    return None, math.inf, {"expanded": expanded, "dist": dist}


def astar(
    cell_map: CellMap,
    start: Cell,
    goal: Cell,
    cost_fn: CostFn | None = None,
) -> tuple[list[Cell] | None, float, dict]:
    if not cell_map.is_traversable(start) or not cell_map.is_traversable(goal):
        return None, math.inf, {"expanded": 0}
    cost_fn = cost_fn or move_cost

    def heuristic(c: Cell) -> float:
        return max(abs(c[0] - goal[0]), abs(c[1] - goal[1]))

    pq: list[tuple[float, float, Cell]] = [(heuristic(start), 0.0, start)]
    dist: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    expanded = 0
    while pq:
        _, cost, cell = heapq.heappop(pq)
        if cost > dist[cell]:
            continue
        expanded += 1
        if cell == goal:
            return reconstruct_path(came_from, cell), cost, {"expanded": expanded, "dist": dist}
        for n in cell_map.neighbors8(cell):
            new_cost = cost + float(cost_fn(cell, n))
            if new_cost < dist.get(n, math.inf):
                dist[n] = new_cost
                came_from[n] = cell
                heapq.heappush(pq, (new_cost + heuristic(n), new_cost, n))
    return None, math.inf, {"expanded": expanded, "dist": dist}
