from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_scenario_dirs(root: str | Path, scenario_name: str) -> dict[str, Path]:
    base = Path(root) / scenario_name
    figures = base / "figures"
    animations = base / "animations"
    data = base / "data"
    figures.mkdir(parents=True, exist_ok=True)
    animations.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return {"base": base, "figures": figures, "animations": animations, "data": data}


def write_rows_csv(path: str | Path, rows: Iterable[dict], columns: list[str] | None = None) -> None:
    df = pd.DataFrame(list(rows))
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
