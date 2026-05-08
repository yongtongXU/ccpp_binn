from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.coverage_planner import CoveragePlanner
from src.core.metrics import SUMMARY_COLUMNS
from src.utils.config import apply_cli_overrides, load_config

SCENARIOS = [
    "configs/scenarios/open_water.yaml",
    "configs/scenarios/single_obstacle.yaml",
    "configs/scenarios/island_obstacles.yaml",
    "configs/scenarios/concave_area.yaml",
]


def run_scenario(path: str, args: argparse.Namespace) -> dict:
    config = apply_cli_overrides(load_config(path), args)
    if args.output:
        config.setdefault("output", {})["root"] = args.output
    planner = CoveragePlanner(config)
    output_root = config.get("output", {}).get("root", "outputs")
    print(
        f"[start] scenario={planner.scenario} method={planner.strategy.name} "
        f"map={planner.cell_map.width}x{planner.cell_map.height} output={output_root}",
        flush=True,
    )
    metrics = planner.run(output_root)
    print(
        f"[done] scenario={planner.scenario} success={metrics['success']} "
        f"coverage={metrics['coverage_rate']:.4f} steps={metrics['total_steps']} "
        f"turns={metrics['number_of_turns']} escapes={metrics['escape_steps']} "
        f"failure={metrics['failure_reason'] or 'none'}",
        flush=True,
    )
    print(f"[output] {Path(output_root) / planner.scenario}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-USV cell-level GBNN assisted coverage planner")
    parser.add_argument("--scenario", help="Path to scenario YAML")
    parser.add_argument("--all", action="store_true", help="Run all fixed scenarios")
    parser.add_argument("--output", help="Output root")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--method", choices=["rolling_gbnn", "gbnn_greedy", "original_binn", "improved_binn"], help="Coverage method for comparison runs")
    parser.add_argument("--no-gbnn", action="store_true")
    parser.add_argument("--no-rolling", action="store_true")
    parser.add_argument("--no-escape", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.scenario:
        parser.error("Use --scenario <yaml> or --all")

    scenario_paths = SCENARIOS if args.all else [args.scenario]
    print(f"[run] planning {len(scenario_paths)} scenario(s)", flush=True)
    rows = [run_scenario(path, args) for path in scenario_paths]
    output_root = args.output or "outputs"
    Path(output_root).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df[SUMMARY_COLUMNS].to_csv(Path(output_root) / "summary.csv", index=False)
    print(f"[summary] {Path(output_root) / 'summary.csv'}", flush=True)
    print(df[SUMMARY_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
