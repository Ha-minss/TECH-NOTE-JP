from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-6) -> pd.Series:
    return numerator / denominator.where(denominator.abs().gt(eps), np.nan)


def build_customer_aggregate_features(
    frame: pd.DataFrame,
    customer_col: str = "customer_ID",
    date_col: str = "S_2",
    target_col: str | None = "target",
) -> pd.DataFrame:
    """Module form of the original customer-month to customer-level aggregate logic."""
    sorted_frame = frame.sort_values([customer_col, date_col], kind="mergesort")
    excluded = {customer_col, date_col}
    if target_col:
        excluded.add(target_col)
    numeric_cols = [col for col in sorted_frame.select_dtypes(include="number").columns if col not in excluded]
    categorical_cols = [
        col for col in sorted_frame.columns if col not in excluded and col not in numeric_cols
    ]

    numeric_agg = sorted_frame.groupby(customer_col)[numeric_cols].agg(["last", "first", "mean", "std", "min", "max", "sum", "median", "count"])
    numeric_agg.columns = ["_".join(col).strip() for col in numeric_agg.columns]

    pieces = [numeric_agg]
    if categorical_cols:
        cat_agg = sorted_frame.groupby(customer_col)[categorical_cols].agg(["last", "first", "nunique"])
        cat_agg.columns = ["_".join(col).strip() for col in cat_agg.columns]
        pieces.append(cat_agg)

    features = pd.concat(pieces, axis=1)
    for col in numeric_cols:
        last = features[f"{col}_last"]
        first = features[f"{col}_first"]
        mean = features[f"{col}_mean"]
        features[f"{col}_last_minus_mean"] = last - mean
        features[f"{col}_last_minus_first"] = last - first
        features[f"{col}_last_div_mean"] = safe_divide(last, mean)
        features[f"{col}_last_div_first"] = safe_divide(last, first)

    features["missing_count_total"] = sorted_frame.groupby(customer_col)[numeric_cols].apply(lambda x: x.isna().sum().sum())
    features["missing_ratio_total"] = sorted_frame.groupby(customer_col)[numeric_cols].apply(lambda x: x.isna().mean().mean())

    if target_col and target_col in sorted_frame.columns:
        features[target_col] = sorted_frame.groupby(customer_col)[target_col].last()

    return features.reset_index()

