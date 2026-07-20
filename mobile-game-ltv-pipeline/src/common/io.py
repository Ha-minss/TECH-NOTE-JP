from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    frame.to_parquet(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["ensure_parent", "read_parquet", "write_parquet", "write_json", "read_json"]
