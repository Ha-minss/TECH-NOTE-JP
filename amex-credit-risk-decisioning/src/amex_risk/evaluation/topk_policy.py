from __future__ import annotations

import math

import pandas as pd


def simulate_topk_policy(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    review_rates: list[float],
) -> pd.DataFrame:
    ranked = frame.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    total_defaults = int(ranked[target_col].sum())
    observed_default_rate = float(ranked[target_col].mean())
    source = ranked["data_kind"].iloc[0] if "data_kind" in ranked.columns and ranked["data_kind"].nunique() == 1 else "original_oof_or_aggregate"
    if source == "synthetic_test_fixture":
        source = "synthetic_test_fixture_not_portfolio_performance"

    rows = []
    for rate in review_rates:
        review_count = math.ceil(len(ranked) * rate)
        reviewed = ranked.head(review_count)
        tp = int(reviewed[target_col].sum())
        fp = int(review_count - tp)
        observed_precision = tp / review_count if review_count else 0.0
        rows.append(
            {
                "policy": f"Top {int(round(rate * 100))}%",
                "review_rate": rate,
                "review_count": review_count,
                "tp": tp,
                "fp": fp,
                "observed_precision": observed_precision,
                "capture_rate": tp / total_defaults if total_defaults else 0.0,
                "lift": observed_precision / observed_default_rate if observed_default_rate else 0.0,
                "missed_default": total_defaults - tp,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def apply_nondefault_weight_scenario(policy_table: pd.DataFrame, nondefault_weight: int = 20) -> pd.DataFrame:
    out = policy_table.copy()
    out["effective_fp"] = out["fp"] * nondefault_weight
    out["effective_review_count"] = out["tp"] + out["effective_fp"]
    out["weighted_scenario_precision"] = out["tp"] / out["effective_review_count"]
    out["nondefault_weight"] = nondefault_weight
    out["weighting_label"] = "AMEX competition sampling-adjusted scenario"
    out["weighting_label_ko"] = f"비부도 고객 {nondefault_weight}배 가중 시나리오"
    return out

