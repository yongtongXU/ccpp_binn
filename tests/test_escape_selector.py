from src.core.cell_map import COVERED, OBSTACLE, UNCOVERED, CellMap
from src.core.escape_selector import EscapeSelector
from src.core.usv import USV


def test_backtracking_candidate_finds_history_cell_with_potential():
    cm = CellMap(6, 4)
    usv = USV((1, 1))
    for c in [(2, 1), (3, 1), (4, 1)]:
        usv.move_to(c, "normal")
        cm.mark_covered(c)
    cm.mark_covered((1, 1))
    cm.mark_covered((2, 2))
    cand = EscapeSelector({"backtracking_max_steps": 10}).find_backtracking_candidate(usv, cm)
    assert cand is not None
    assert cm.is_traversable(cand.target)


def test_dijkstra_candidate_does_not_cross_obstacle_wall():
    cm = CellMap(7, 5)
    cm.grid[:, 3] = OBSTACLE
    cm.grid[4, 3] = 0
    usv = USV((1, 2))
    for y in range(5):
        for x in range(3):
            cm.mark_covered((x, y))
    cand = EscapeSelector({"dijkstra_max_expansion": 100}).find_dijkstra_candidate(usv, cm)
    assert cand is not None
    assert all(cm.is_traversable(c) for c in cand.path)
    assert (3, 2) not in cand.path


def test_dijkstra_uses_reachable_path_cost_not_geometry():
    cm = CellMap(8, 5)
    cm.grid[1:5, 3] = OBSTACLE
    cm.grid[4, 3] = 0
    usv = USV((2, 2))
    for y in range(5):
        for x in range(8):
            if cm.is_traversable((x, y)):
                cm.mark_covered((x, y))
    cm.grid[2, 4] = 0
    cm.visit_count[2, 4] = 0
    cand = EscapeSelector({"dijkstra_max_expansion": 100}).find_dijkstra_candidate(usv, cm)
    assert cand is not None
    assert len(cand.path) > 2
    assert all(cm.is_traversable(c) for c in cand.path)


def test_fusion_returns_legal_escape_target():
    cm = CellMap(5, 5)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    target, escape_type, path, _ = EscapeSelector({}).select_escape_target(usv, cm)
    assert target is not None
    assert escape_type in {"backtracking", "dijkstra"}
    assert path[0] == usv.current_cell
    assert cm.is_traversable(target)


def test_dijkstra_prefers_axis_aligned_uncovered_entry_on_tie():
    cm = CellMap(4, 5)
    cm.grid[:, :] = COVERED
    cm.visit_count[:, :] = 1
    cm.grid[4, 1] = UNCOVERED
    cm.grid[4, 2] = UNCOVERED
    usv = USV((2, 1))
    cand = EscapeSelector({"dijkstra_max_expansion": 100}).find_dijkstra_candidate(usv, cm)
    assert cand is not None
    assert cand.target == (2, 4)
