import numpy as np

from src.core.grid_map import GridMap


def test_map_size_and_obstacle():
    gm = GridMap.from_config({"map": {"width": 20, "height": 10, "obstacles": [{"type": "rectangle", "x": 5, "y": 2, "width": 4, "height": 3}]}})
    assert gm.grid.shape == (10, 20)
    assert gm.grid[3, 6] == 1
    assert gm.grid[0, 0] == 0


def test_line_is_free():
    gm = GridMap.from_config({"map": {"width": 20, "height": 20, "obstacles": [{"type": "rectangle", "x": 8, "y": 0, "width": 2, "height": 18}]}})
    assert not gm.line_is_free((1, 1), (1, 15))
    assert gm.line_is_free((19, 1), (19, 15))


def test_connected_components():
    gm = GridMap(6, 4)
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[3, 5] = True
    comps = gm.connected_components(mask)
    assert sorted(len(c) for c in comps) == [1, 2]
