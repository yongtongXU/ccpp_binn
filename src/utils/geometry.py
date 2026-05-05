from __future__ import annotations

import math
from typing import Iterable, Sequence


Point = tuple[float, float]
Cell = tuple[int, int]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length(path: Sequence[Sequence[float]]) -> float:
    return sum(euclidean(path[i - 1], path[i]) for i in range(1, len(path)))


def heading(a: Sequence[float], b: Sequence[float]) -> float:
    return math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def angle_diff(a: float, b: float) -> float:
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def count_turns(path: Sequence[Sequence[float]], threshold_deg: float = 20.0) -> int:
    if len(path) < 3:
        return 0
    threshold = math.radians(threshold_deg)
    turns = 0
    prev = heading(path[0], path[1])
    for i in range(2, len(path)):
        cur = heading(path[i - 1], path[i])
        if angle_diff(prev, cur) > threshold:
            turns += 1
        prev = cur
    return turns


def cumulative_heading_change(path: Sequence[Sequence[float]]) -> float:
    if len(path) < 3:
        return 0.0
    total = 0.0
    prev = heading(path[0], path[1])
    for i in range(2, len(path)):
        cur = heading(path[i - 1], path[i])
        total += angle_diff(prev, cur)
        prev = cur
    return math.degrees(total)


def dedupe_path(path: Iterable[Cell]) -> list[Cell]:
    result: list[Cell] = []
    for cell in path:
        if not result or result[-1] != cell:
            result.append(cell)
    return result
