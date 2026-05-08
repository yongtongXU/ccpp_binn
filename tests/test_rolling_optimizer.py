from src.core.cell_map import COVERED, CellMap, OBSTACLE, UNCOVERED
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


def test_candidate_tree_expands_all_axis_directions_by_default():
    cm = CellMap(5, 5)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer({"horizon": 1, "beam_width": 8, "record_candidate_count": 8})
    _, _, details = opt.select_next_cell(usv, cm, gbnn)
    first_steps = {tuple(candidate["path"][0]) for candidate in details["candidate_branches"]}
    assert first_steps == {(1, 2), (2, 1), (2, 3), (3, 2)}


def test_candidate_tree_can_include_diagonal_moves_when_enabled():
    cm = CellMap(5, 5)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer({"horizon": 1, "beam_width": 8, "record_candidate_count": 8, "allow_diagonal_normal": True})
    _, _, details = opt.select_next_cell(usv, cm, gbnn)
    first_steps = {tuple(candidate["path"][0]) for candidate in details["candidate_branches"]}
    assert first_steps == set(cm.neighbors8((2, 2)))


def test_candidate_tree_records_each_depth_level():
    cm = CellMap(5, 5)
    usv = USV((2, 2))
    cm.mark_covered((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer({"horizon": 3, "beam_width": 8, "record_candidate_count": 8, "record_tree_count": 64, "use_priority_strip": False})
    _, _, details = opt.select_next_cell(usv, cm, gbnn)
    levels = details["candidate_tree"]["levels"]
    assert [level["depth"] for level in levels] == [1, 2, 3]
    assert all(level["branches"] for level in levels)
    assert all(len(branch["path"]) == level["depth"] for level in levels for branch in level["branches"])
    assert details["planning_mode"] == "open_water"
    assert all(branch["planning_mode"] == "open_water" for branch in levels[0]["branches"])


def test_planning_state_detects_junction_search():
    cm = CellMap(5, 5)
    cm.grid[:, :] = COVERED
    cm.visit_count[:, :] = 1
    for cell in [(2, 1), (1, 2), (3, 2), (2, 3)]:
        cm.grid[cell[1], cell[0]] = UNCOVERED
        cm.visit_count[cell[1], cell[0]] = 0
    usv = USV((2, 2))
    opt = RollingOptimizer({"mode_junction_min_branches": 3})
    state = opt.classify_planning_state(usv, cm)
    assert state.mode == "junction_search"
    assert state.uncovered_neighbors == 4


def test_planning_state_detects_dead_zone():
    cm = CellMap(4, 4)
    cm.mark_covered((1, 1))
    for n in cm.neighbors8((1, 1)):
        cm.mark_covered(n)
    usv = USV((1, 1))
    state = RollingOptimizer({}).classify_planning_state(usv, cm)
    assert state.mode == "dead_zone"
    assert state.reason == "no_uncovered_neighbor"


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


def test_fork_prefers_one_entry_branch_that_would_need_backfill():
    cm = CellMap(7, 5)
    cm.grid[:, :] = OBSTACLE
    cm.grid[2, 2] = COVERED
    cm.visit_count[2, 2] = 1
    cm.grid[2, 1] = UNCOVERED
    cm.grid[2, 3] = UNCOVERED
    cm.grid[2, 4] = UNCOVERED
    cm.grid[2, 5] = UNCOVERED
    cm.grid[1, 4] = COVERED
    cm.visit_count[1, 4] = 1

    usv = USV((2, 2))
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer(
        {
            "horizon": 1,
            "use_priority_strip": False,
            "w_new_coverage": 1.0,
            "w_activity": 0.0,
            "w_direction": 0.0,
            "w_structure": 0.0,
            "w_branch_urgency": 100.0,
            "w_strip_forward": 0.0,
            "w_strip_reverse": 0.0,
            "w_strip_cross": 0.0,
            "w_turn": 0.0,
            "w_dead_zone": 0.0,
            "w_obstacle": 0.0,
        }
    )
    nxt, _, details = opt.select_next_cell(usv, cm, gbnn)
    assert nxt == (1, 2)
    assert details["branch_urgency_score"] > 0


def test_missed_branch_bonus_enters_side_strip_before_passing_it():
    cm = CellMap(8, 5)
    cm.grid[:, :] = OBSTACLE
    cm.grid[2, 0:8] = UNCOVERED
    cm.grid[3, 0:8] = UNCOVERED
    cm.grid[1, 0:4] = OBSTACLE
    cm.grid[1, 4:8] = UNCOVERED
    cm.mark_covered((3, 2))
    usv = USV((3, 2), heading=0, current_strip_id=2, strip_direction=1)
    gbnn = GBNNField({})
    gbnn.initialize(cm)
    opt = RollingOptimizer(
        {
            "horizon": 1,
            "use_priority_strip": True,
            "w_new_coverage": 8.0,
            "w_activity": 0.0,
            "w_direction": 4.0,
            "w_structure": 0.0,
            "w_missed_branch": 80.0,
            "missed_branch_min_behind": 3,
            "missed_branch_full_behind": 4,
            "w_strip_forward": 18.0,
            "w_strip_transition": 14.0,
            "w_strip_cross": 35.0,
            "w_turn": 0.0,
            "w_dead_zone": 0.0,
            "w_obstacle": 0.0,
        }
    )
    nxt, _, details = opt.select_next_cell(usv, cm, gbnn)
    assert nxt[1] == 3
    assert details["missed_branch_score"] > 0
