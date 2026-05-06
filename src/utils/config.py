from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(scenario_path: str | Path | None = None, default_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config = load_yaml(default_path)
    if scenario_path:
        scenario = load_yaml(scenario_path)
        config = deep_merge(config, scenario)
        config["scenario_path"] = str(scenario_path)
    return config


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    config = deepcopy(config)
    if getattr(args, "output", None):
        config.setdefault("output", {})["root"] = args.output
    if getattr(args, "max_steps", None):
        config.setdefault("planner", {})["max_steps"] = args.max_steps
    if getattr(args, "no_gbnn", False):
        config.setdefault("gbnn", {})["enabled"] = False
    if getattr(args, "no_rolling", False):
        config.setdefault("rolling_optimizer", {})["enabled"] = False
    if getattr(args, "no_escape", False):
        config.setdefault("escape", {})["enabled"] = False
    return config
