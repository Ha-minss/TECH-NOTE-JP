from __future__ import annotations

import pandas as pd


def equal_blend(frame: pd.DataFrame, prediction_cols: list[str], output_col: str = "final_score") -> pd.DataFrame:
    """Average verified OOF prediction columns without fabricating model-specific scores."""
    if not prediction_cols:
        raise ValueError("prediction_cols must not be empty")
    missing = [col for col in prediction_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    out = frame.copy()
    out[output_col] = out[prediction_cols].mean(axis=1)
    return out
