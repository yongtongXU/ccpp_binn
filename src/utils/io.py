from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_output_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "figures": root / "figures",
        "data": root / "data",
        "animations": root / "animations",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: str | Path, rows: Iterable[dict], columns: list[str]) -> None:
    df = pd.DataFrame(list(rows), columns=columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
