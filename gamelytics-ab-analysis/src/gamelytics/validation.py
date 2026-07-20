from __future__ import annotations

from math import erfc, sqrt
from typing import Mapping

import pandas as pd


def assert_single_group_per_user(df: pd.DataFrame) -> None:
    groups_per_user = df.groupby("user_id")["testgroup"].nunique()
    bad_users = groups_per_user[groups_per_user > 1]
    if not bad_users.empty:
        raise ValueError(f"{len(bad_users)} users appear in multiple test groups")


def sample_ratio_mismatch_test(counts: Mapping[str, int]) -> dict[str, float]:
    observed = pd.Series(counts, dtype="float64")
    expected = observed.sum() / len(observed)
    chi_square = float(((observed - expected) ** 2 / expected).sum())
    if len(observed) == 2:
        p_value = erfc(sqrt(chi_square / 2.0))
    else:
        p_value = float("nan")
    return {"chi_square": chi_square, "p_value": p_value}


def audit_ab_data(df: pd.DataFrame) -> dict[str, object]:
    assert_single_group_per_user(df)
    grouped_revenue = df.groupby("testgroup")["revenue"]
    return {
        "rows": len(df),
        "unique_users": df["user_id"].nunique(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_user_ids": int(df["user_id"].duplicated().sum()),
        "users_in_multiple_groups": 0,
        "group_counts": df["testgroup"].value_counts().sort_index().to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "negative_revenue_rows": int((df["revenue"] < 0).sum()),
        "zero_revenue_share": float((df["revenue"] == 0).mean()),
        "payers_by_group": grouped_revenue.apply(lambda s: int((s > 0).sum())).to_dict(),
        "revenue_quantiles_by_group": grouped_revenue.quantile(
            [0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
        ).unstack().to_dict("index"),
    }
