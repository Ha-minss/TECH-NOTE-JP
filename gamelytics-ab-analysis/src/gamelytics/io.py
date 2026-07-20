from pathlib import Path

import pandas as pd


def read_semicolon_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", **kwargs)


def load_ab_data(data_dir: str | Path) -> pd.DataFrame:
    return read_semicolon_csv(Path(data_dir) / "ab_test.csv")


def load_retention_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    return read_semicolon_csv(base / "reg_data.csv"), read_semicolon_csv(base / "auth_data.csv")
