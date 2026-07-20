from __future__ import annotations

import pandas as pd


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("testgroup").agg(
        users=("user_id", "count"),
        payers=("revenue", lambda s: int((s > 0).sum())),
        total_revenue=("revenue", "sum"),
        arpu=("revenue", "mean"),
        median_revenue=("revenue", "median"),
        max_revenue=("revenue", "max"),
    )
    summary["conversion_rate"] = summary["payers"] / summary["users"]
    summary["arppu"] = summary["total_revenue"] / summary["payers"]
    summary["arpu_identity"] = summary["conversion_rate"] * summary["arppu"]
    return summary


def revenue_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_df in df.groupby("testgroup"):
        payers = group_df[group_df["revenue"] > 0].copy()
        total_revenue = float(group_df["revenue"].sum())
        payer_revenue = payers["revenue"].sort_values(ascending=False).reset_index(drop=True)
        row = {
            "testgroup": group,
            "users": len(group_df),
            "payers": len(payers),
            "zero_revenue_share": float((group_df["revenue"] == 0).mean()),
            "payer_revenue_mean": float(payers["revenue"].mean()),
            "payer_revenue_median": float(payers["revenue"].median()),
            "payer_revenue_p90": float(payers["revenue"].quantile(0.9)),
            "payer_revenue_p95": float(payers["revenue"].quantile(0.95)),
            "payer_revenue_p99": float(payers["revenue"].quantile(0.99)),
            "max_revenue": float(group_df["revenue"].max()),
        }
        for pct in (0.01, 0.05, 0.10):
            n = max(1, int(len(payer_revenue) * pct))
            row[f"top_{int(pct * 100)}pct_payer_revenue_share"] = (
                float(payer_revenue.iloc[:n].sum() / total_revenue) if total_revenue else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("testgroup")


def compare_groups(summary: pd.DataFrame, control: str = "a", treatment: str = "b") -> dict[str, float]:
    a = summary.loc[control]
    b = summary.loc[treatment]
    return {
        "arpu_diff": float(b["arpu"] - a["arpu"]),
        "arpu_lift": float(b["arpu"] / a["arpu"] - 1),
        "conversion_diff_pp": float((b["conversion_rate"] - a["conversion_rate"]) * 100),
        "conversion_relative_diff": float(b["conversion_rate"] / a["conversion_rate"] - 1),
        "arppu_diff": float(b["arppu"] - a["arppu"]),
        "total_revenue_diff": float(b["total_revenue"] - a["total_revenue"]),
    }
