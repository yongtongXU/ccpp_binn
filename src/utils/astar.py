from __future__ import annotations

import heapq
from typing import Optional

from src.utils.geometry import Cell, euclidean


def astar(grid_map, start: Cell, goal: Cell) -> Optional[list[Cell]]:
    if start == goal:
        return [start]
    if not grid_map.is_free_cell(start) or not grid_map.is_free_cell(goal):
        return None

    open_heap: list[tuple[float, int, Cell]] = []
    heapq.heappush(open_heap, (euclidean(start, goal), 0, start))
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    counter = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        for nb in grid_map.neighbors(current, diagonal=True):
            step = euclidean(current, nb)
            tentative = g_score[current] + step
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                counter += 1
                f = tentative + euclidean(nb, goal)
                heapq.heappush(open_heap, (f, counter, nb))
    return None


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
