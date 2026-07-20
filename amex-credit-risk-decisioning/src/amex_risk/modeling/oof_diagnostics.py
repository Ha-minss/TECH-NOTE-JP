from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from amex_risk.modeling.metrics import amex_metric


def compute_model_correlation(frame: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in prediction_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    return frame[prediction_cols].corr(method="pearson")


def _score(frame: pd.DataFrame, target_col: str, score_col: str) -> dict[str, float]:
    return {
        "roc_auc": _roc_auc(frame[target_col], frame[score_col]),
        "amex_metric": amex_metric(frame[target_col], frame[score_col]),
    }


def _roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    ranked = pd.DataFrame({"target": y_true, "score": y_score}).sort_values("score", kind="mergesort")
    n_pos = int(ranked["target"].sum())
    n_neg = int(len(ranked) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires both classes")
    ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    pos_rank_sum = ranks[ranked["target"].eq(1)].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_single_and_leave_one_out_blends(
    frame: pd.DataFrame,
    target_col: str,
    prediction_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for col in prediction_cols:
        scored = _score(frame, target_col, col)
        rows.append(
            {
                "evaluation_type": "single_model",
                "model_name": col,
                "removed_model": "",
                "n_models": 1,
                **scored,
            }
        )

    all_col = "__all_model_equal_blend"
    frame = frame.copy()
    frame[all_col] = frame[prediction_cols].mean(axis=1)
    all_scored = _score(frame, target_col, all_col)
    rows.append(
        {
            "evaluation_type": "all_model_equal_blend",
            "model_name": " + ".join(prediction_cols),
            "removed_model": "",
            "n_models": len(prediction_cols),
            **all_scored,
        }
    )

    for removed in prediction_cols:
        kept = [col for col in prediction_cols if col != removed]
        blend_col = f"__without_{removed}"
        frame[blend_col] = frame[kept].mean(axis=1)
        scored = _score(frame, target_col, blend_col)
        rows.append(
            {
                "evaluation_type": "leave_one_out_blend",
                "model_name": " + ".join(kept),
                "removed_model": removed,
                "n_models": len(kept),
                **scored,
            }
        )

    return pd.DataFrame(rows)


def compute_incremental_gain(comparison: pd.DataFrame) -> pd.DataFrame:
    all_metric = comparison.loc[comparison["evaluation_type"].eq("all_model_equal_blend"), "amex_metric"].iloc[0]
    best_single = comparison.loc[comparison["evaluation_type"].eq("single_model"), "amex_metric"].max()
    single = comparison[comparison["evaluation_type"].eq("single_model")][["model_name", "amex_metric", "roc_auc"]].copy()
    single = single.rename(columns={"amex_metric": "single_model_amex", "roc_auc": "single_model_roc_auc"})

    loo = comparison[comparison["evaluation_type"].eq("leave_one_out_blend")][["removed_model", "amex_metric"]].copy()
    loo["leave_one_out_delta_vs_all"] = all_metric - loo["amex_metric"]
    loo = loo.rename(columns={"removed_model": "model_name", "amex_metric": "blend_without_model_amex"})

    out = single.merge(loo, on="model_name", how="left")
    out["all_model_blend_amex"] = all_metric
    out["best_single_amex"] = best_single
    out["best_single_delta_vs_model"] = out["single_model_amex"] - best_single
    out["all_blend_delta_vs_model"] = all_metric - out["single_model_amex"]
    return out.sort_values("leave_one_out_delta_vs_all", ascending=False).reset_index(drop=True)


def load_oof_prediction_matrix(paths: dict[str, str | Path], target_col: str = "target") -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model_name, path in paths.items():
        df = pd.read_csv(path)
        pred_candidates = [c for c in df.columns if c not in {"customer_ID", target_col}]
        if len(pred_candidates) != 1:
            raise ValueError(f"{path} must contain customer_ID, {target_col}, and exactly one prediction column")
        pred_col = pred_candidates[0]
        df = df[["customer_ID", target_col, pred_col]].rename(columns={pred_col: model_name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["customer_ID", target_col], how="inner")
    if merged is None:
        raise ValueError("No OOF paths were provided")
    return merged

