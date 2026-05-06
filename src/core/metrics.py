from __future__ import annotations

from src.core.cell_map import CellMap
from src.core.usv import USV

SUMMARY_COLUMNS = [
    "scenario",
    "success",
    "coverage_rate",
    "repeated_coverage_rate",
    "path_length",
    "number_of_turns",
    "total_steps",
    "normal_steps",
    "escape_steps",
    "deadlock_count",
    "backtracking_escape_count",
    "dijkstra_escape_count",
    "average_escape_length",
    "max_escape_length",
    "residual_components",
    "failure_reason",
]


def number_of_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        heading = (0 if b[0] == a[0] else (1 if b[0] > a[0] else -1), 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1))
        if prev is not None and heading != prev:
            turns += 1
        prev = heading
    return turns


def compute_metrics(
    scenario: str,
    success: bool,
    cell_map: CellMap,
    usv: USV,
    deadlock_count: int,
    escape_records: list[dict],
    failure_reason: str = "",
) -> dict:
    escape_lengths = [int(e.get("path_length", 0)) for e in escape_records]
    normal_steps = sum(1 for m in usv.mode_history[1:] if m == "normal")
    escape_steps = sum(1 for m in usv.mode_history[1:] if m == "escape")
    return {
        "scenario": scenario,
        "success": bool(success),
        "coverage_rate": cell_map.coverage_rate(),
        "repeated_coverage_rate": cell_map.repeated_coverage_rate(),
        "path_length": max(0, len(usv.path) - 1),
        "number_of_turns": number_of_turns(usv.path),
        "total_steps": max(0, len(usv.path) - 1),
        "normal_steps": normal_steps,
        "escape_steps": escape_steps,
        "deadlock_count": deadlock_count,
        "backtracking_escape_count": sum(1 for e in escape_records if e.get("escape_type") == "backtracking"),
        "dijkstra_escape_count": sum(1 for e in escape_records if e.get("escape_type") == "dijkstra"),
        "average_escape_length": sum(escape_lengths) / len(escape_lengths) if escape_lengths else 0.0,
        "max_escape_length": max(escape_lengths) if escape_lengths else 0,
        "residual_components": len(cell_map.get_uncovered_components()),
        "failure_reason": failure_reason,
    }
