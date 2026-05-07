from pathlib import Path

from src.core.coverage_planner import CoveragePlanner
from src.utils.config import load_config


def test_open_water_20x20_smoke(tmp_path: Path):
    cfg = load_config()
    cfg["scenario"] = {"name": "smoke_open_water"}
    cfg["map"]["width"] = 20
    cfg["map"]["height"] = 20
    cfg["map"]["obstacles"] = []
    cfg["start"] = {"x": 1, "y": 1}
    cfg["planner"]["max_steps"] = 5000
    cfg.setdefault("output", {}).setdefault("animation", {})["enabled"] = False
    metrics = CoveragePlanner(cfg).run(tmp_path)
    assert metrics["success"] is True
    assert metrics["coverage_rate"] == 1.0
    assert (tmp_path / "smoke_open_water" / "data" / "path.csv").exists()
    assert (tmp_path / "smoke_open_water" / "data" / "metrics.csv").exists()
