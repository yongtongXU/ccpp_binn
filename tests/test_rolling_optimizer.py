from src.core.cell_map import CellMap, OBSTACLE
from src.core.gbnn_field import GBNNField
from src.core.rolling_optimizer import RollingOptimizer
from src.core.usv import USV


def make_optimizer():
    return RollingOptimizer({"horizon": 3, "beam_width": 10, "allow_immediate_backtrack": False})


def test_open_map_prefers_new_coverage_neighbor():
    cm = CellMap(6, 6)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    nxt, branch, _ = make_optimizer().select_next_cell(usv, cm, gbnn)
    assert nxt in cm.neighbors8((2, 2))
    assert cm.is_uncovered(nxt)
    assert branch


def test_candidate_tree_expands_all_eight_directions_per_step():
    cm = CellMap(5, 5)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer({"horizon": 1, "beam_width": 8, "record_candidate_count": 8})
    _, _, details = opt.select_next_cell(usv, cm, gbnn)
    first_steps = {tuple(candidate["path"][0]) for candidate in details["candidate_branches"]}
    assert first_steps == set(cm.neighbors8((2, 2)))


def test_avoids_obstacle_and_next_is_neighbor():
    cm = CellMap(5, 5)
    cm.grid[2, 3] = OBSTACLE
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    nxt, _, _ = make_optimizer().select_next_cell(usv, cm, gbnn)
    assert nxt in cm.neighbors8((2, 2))
    assert nxt != (3, 2)


def test_allows_limited_repeat_but_not_immediate_backtrack():
    cm = CellMap(4, 4)
    usv = USV((1, 1))
    usv.move_to((2, 1), "normal")
    cm.mark_covered((1, 1))
    cm.mark_covered((2, 1))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    nxt, _, _ = make_optimizer().select_next_cell(usv, cm, gbnn)
    assert nxt != (1, 1)
    assert nxt in cm.neighbors8((2, 1))
