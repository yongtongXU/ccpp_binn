from __future__ import annotations

import argparse
from pathlib import Path

from src.core.planner import CoveragePlanner
from src.utils.config import load_config
from src.utils.io import ensure_output_dirs, write_csv
from src.visualization.animation import save_coverage_gif
from src.visualization.plotter import plot_coverage, plot_gbnn, plot_metrics_curve, plot_segments, plot_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connectivity-preserving full-coverage path planner")
    parser.add_argument("--scenario", required=True, help="Path to scenario yaml")
    parser.add_argument("--output", default=None, help="Output root")
    parser.add_argument("--animate", action="store_true", help="Save coverage animation")
    parser.add_argument("--animation-step", type=int, default=1, help="Render every N path samples in the GIF")
    parser.add_argument("--no-gbnn", action="store_true", help="Disable GBNN residual demand field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.scenario)
    if args.output:
        cfg.setdefault("output", {})["root"] = args.output
    output_root = cfg.get("output", {}).get("root", "outputs")
    dirs = ensure_output_dirs(output_root)

    result = CoveragePlanner(cfg).run(gbnn_enabled=not args.no_gbnn)
    plot_trajectory(result, dirs["figures"] / "trajectory.png")
    plot_segments(result, dirs["figures"] / "segments.png")
    plot_coverage(result, dirs["figures"] / "coverage_map.png")
    plot_gbnn(result, dirs["figures"] / "gbnn_activity.png")
    plot_metrics_curve(result, dirs["figures"] / "metrics_curve.png")
    if args.animate:
        save_coverage_gif(result, dirs["animations"] / "coverage.gif", step=args.animation_step)

    write_csv(dirs["data"] / "paths.csv", _path_rows(result), ["usv_id", "order", "x", "y", "mode"])
    write_csv(
        dirs["data"] / "segments.csv",
        _segment_rows(result),
        ["segment_id", "line_index", "start_x", "start_y", "end_x", "end_y", "length", "assigned_usv"],
    )
    metric_cols = [
        "scenario",
        "mode",
        "success",
        "best_scan_angle",
        "coverage_rate",
        "repeated_coverage_rate",
        "total_path_length",
        "number_of_turns",
        "cumulative_heading_change",
        "number_of_segments",
        "astar_connection_count",
        "residual_before",
        "residual_after",
        "recovery_path_length",
        "load_balance_index",
    ]
    write_csv(dirs["data"] / "final_metrics.csv", [result.metrics], metric_cols)
    print(f"scenario={result.metrics['scenario']}")
    print(f"mode={result.metrics['mode']} best_scan_angle={result.best_scan_angle}")
    print(f"coverage_rate={result.metrics['coverage_rate']:.4f} residual_after={result.metrics['residual_after']}")
    print(f"outputs={Path(output_root).resolve()}")


def _path_rows(result):
    rows = []
    for uid, path in result.paths_by_usv.items():
        modes = result.modes_by_usv.get(uid, [])
        for i, point in enumerate(path):
            rows.append({"usv_id": uid, "order": i, "x": point[0], "y": point[1], "mode": modes[i] if i < len(modes) else "idle"})
    return rows


def _segment_rows(result):
    rows = []
    for seg in result.segments:
        sx, sy = seg.start_point
        ex, ey = seg.end_point
        rows.append(
            {
                "segment_id": seg.id,
                "line_index": seg.line_index,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "length": seg.length,
                "assigned_usv": seg.assigned_usv or "",
            }
        )
    return rows


if __name__ == "__main__":
    main()
