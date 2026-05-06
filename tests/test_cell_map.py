from src.core.cell_map import CellMap, OBSTACLE


def test_map_initialization_and_obstacle_creation():
    cm = CellMap.from_config({"map": {"width": 10, "height": 8, "obstacles": [{"type": "rectangle", "x": 2, "y": 2, "width": 3, "height": 2}]}})
    assert cm.width == 10
    assert cm.height == 8
    assert cm.grid[2, 2] == OBSTACLE
    assert cm.is_obstacle((4, 3))
    assert not cm.is_traversable((2, 2))


def test_neighbors8_and_coverage_rate():
    cm = CellMap(4, 4)
    cm.grid[1, 1] = OBSTACLE
    ns = cm.neighbors8((0, 0))
    assert (1, 0) in ns
    assert (0, 1) in ns
    assert (1, 1) not in ns
    cm.mark_covered((0, 0))
    assert cm.coverage_rate() == 1 / 15
    cm.mark_covered((0, 0))
    assert cm.repeated_coverage_rate() > 0


def test_uncovered_components():
    cm = CellMap(5, 3)
    cm.grid[:, 2] = OBSTACLE
    comps = cm.get_uncovered_components()
    assert sorted(len(c) for c in comps) == [6, 6]
