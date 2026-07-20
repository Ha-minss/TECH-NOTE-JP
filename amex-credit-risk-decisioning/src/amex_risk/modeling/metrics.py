from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if labels.shape[0] == 0:
        raise ValueError("AMEX metric requires at least one row")
    unique = set(labels.tolist())
    if unique != {0, 1}:
        raise ValueError("AMEX metric requires both default and non-default targets")
    return labels, scores


def top_four_percent_captured(y_true, y_score) -> float:
    """Competition top-four-percent capture component with AMEX class weights."""
    labels, scores = _as_arrays(y_true, y_score)
    df = pd.DataFrame({"target": labels, "prediction": scores})
    df = df.sort_values("prediction", ascending=False, kind="mergesort").reset_index(drop=True)
    df["weight"] = np.where(df["target"].eq(0), 20, 1)
    cutoff = int(0.04 * df["weight"].sum())
    cutoff = max(cutoff, 1)
    selected = df.loc[df["weight"].cumsum().le(cutoff)]
    captured_default_weight = selected.loc[selected["target"].eq(1), "weight"].sum()
    total_default_weight = df.loc[df["target"].eq(1), "weight"].sum()
    return float(captured_default_weight / total_default_weight)


def weighted_gini(y_true, y_score) -> float:
    """Weighted Gini component used by the AMEX competition metric."""
    labels, scores = _as_arrays(y_true, y_score)
    df = pd.DataFrame({"target": labels, "prediction": scores})
    df["weight"] = np.where(df["target"].eq(0), 20, 1)
    df = df.sort_values("prediction", ascending=False, kind="mergesort").reset_index(drop=True)
    random = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    lorentz = (df["target"] * df["weight"]).cumsum() / total_pos
    return float(((lorentz - random) * df["weight"]).sum())


def normalized_gini(y_true, y_score) -> float:
    labels, scores = _as_arrays(y_true, y_score)
    return weighted_gini(labels, scores) / weighted_gini(labels, labels)


def amex_metric(y_true, y_score) -> float:
    """Final AMEX metric: average of normalized Gini and top-four capture."""
    return 0.5 * (normalized_gini(y_true, y_score) + top_four_percent_captured(y_true, y_score))

