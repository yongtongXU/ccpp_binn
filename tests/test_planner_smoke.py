from src.core.planner import CoveragePlanner


def test_20x20_open_water_planner_smoke():
    cfg = {
        "scenario_name": "smoke",
        "map": {"width": 20, "height": 20, "resolution": 1, "obstacles": []},
        "planning": {"scan_angles": [0, 90], "target_coverage_rate": 0.95, "min_residual_size": 3, "allow_astar_connection": True},
        "sweep": {"coverage_width": 4, "min_segment_length": 3},
        "cost": {"w_path_length": 1.0, "w_turns": 2.0, "w_connection": 1.5, "w_astar": 3.0, "w_residual": 4.0, "w_obstacle": 1.0},
        "gbnn": {"enabled": True, "iterations": 2},
        "cluster": {"enabled": False},
        "usvs": [{"id": "USV1", "start": [1.5, 1.5], "coverage_width": 4, "coverage_radius": 2, "max_speed": 1.0, "endurance": 10000}],
    }
    result = CoveragePlanner(cfg).run()
    assert result.metrics["coverage_rate"] > 0.95
    assert result.metrics["number_of_segments"] > 0
    jumps = []
    path = result.paths_by_usv["USV1"]
    for i in range(1, len(path)):
        jumps.append(((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5)
    assert max(jumps) < 30
