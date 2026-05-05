from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(scenario_path: str | Path, default_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    default_cfg = load_yaml(default_path)
    scenario_cfg = load_yaml(scenario_path)
    cfg = deep_update(default_cfg, scenario_cfg)
    cfg["scenario_path"] = str(scenario_path)
    cfg["scenario_name"] = scenario_cfg.get("scenario", {}).get("name", Path(scenario_path).stem)
    return cfg
