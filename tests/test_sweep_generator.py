from src.core.grid_map import GridMap
from src.core.sweep_generator import SweepGenerator


def test_open_water_segments_ordered_parallel():
    gm = GridMap(20, 12)
    segments = SweepGenerator(gm, coverage_width=4, scan_angle=0, min_segment_length=3).generate()
    assert len(segments) > 0
    assert [s.line_index for s in segments] == sorted(s.line_index for s in segments)
    assert all(abs(p[1] - s.sampled_points[0][1]) < 1e-9 for s in segments for p in s.sampled_points)
    line_y = [segments[i].sampled_points[0][1] for i in range(min(3, len(segments)))]
    assert all(abs((line_y[i] - line_y[i - 1]) - 4) < 1 for i in range(1, len(line_y)))


def test_single_obstacle_cuts_scanline():
    gm = GridMap.from_config({"map": {"width": 30, "height": 20, "obstacles": [{"type": "rectangle", "x": 12, "y": 0, "width": 5, "height": 20}]}})
    segments = SweepGenerator(gm, coverage_width=4, scan_angle=0, min_segment_length=3).generate()
    split_lines = {}
    for seg in segments:
        split_lines.setdefault(seg.line_index, 0)
        split_lines[seg.line_index] += 1
    assert any(count >= 2 for count in split_lines.values())


def test_segments_do_not_cross_obstacles():
    gm = GridMap.from_config({"map": {"width": 30, "height": 20, "obstacles": [{"type": "rectangle", "x": 12, "y": 6, "width": 5, "height": 8}]}})
    segments = SweepGenerator(gm, coverage_width=4, scan_angle=0, min_segment_length=3).generate()
    assert all(gm.is_free_cell(cell) for seg in segments for cell in seg.sampled_cells)
