from __future__ import annotations

from src.utils.geometry import count_turns, cumulative_heading_change, path_length


def compute_metrics(
    grid_map,
    coverage_state,
    paths_by_usv: dict[str, list],
    number_of_segments: int,
    astar_connection_count: int,
    residual_before: int,
    residual_after: int,
    recovery_path_length: float,
    scenario: str,
    mode: str,
    best_scan_angle: float,
) -> dict:
    path_lengths = {uid: path_length(path) for uid, path in paths_by_usv.items()}
    total_path = sum(path_lengths.values())
    all_turns = sum(count_turns(path) for path in paths_by_usv.values())
    all_heading = sum(cumulative_heading_change(path) for path in paths_by_usv.values())
    load_balance = _load_balance_index(list(path_lengths.values()))
    return {
        "scenario": scenario,
        "mode": mode,
        "success": bool(coverage_state.coverage_rate >= 0.95),
        "best_scan_angle": best_scan_angle,
        "coverage_rate": coverage_state.coverage_rate,
        "repeated_coverage_rate": coverage_state.repeated_coverage_rate,
        "total_path_length": total_path,
        "path_length_each_usv": path_lengths,
        "number_of_turns": all_turns,
        "cumulative_heading_change": all_heading,
        "number_of_segments": number_of_segments,
        "astar_connection_count": astar_connection_count,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "recovery_path_length": recovery_path_length,
        "load_balance_index": load_balance,
    }


def _load_balance_index(values: list[float]) -> float:
    if not values:
        return 1.0
    total = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0
    return (total * total) / (len(values) * sq)
