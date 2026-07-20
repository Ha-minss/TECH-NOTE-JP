from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.run_feature_ablation import (
    build_time_bucket_features_from_zip,
    fit_xgboost_feature_set,
    get_feature_set_frame,
)
from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN
from experiments.train_two_stage_model import fit_two_stage_feature_set


FOLDS = [
    {"fold": 1, "train_min": 0, "train_max": 13, "valid_min": 14, "valid_max": 16},
    {"fold": 2, "train_min": 0, "train_max": 16, "valid_min": 17, "valid_max": 19},
    {"fold": 3, "train_min": 0, "train_max": 19, "valid_min": 20, "valid_max": 23},
    {"fold": 4, "train_min": 0, "train_max": 23, "valid_min": 24, "valid_max": 30},
]

MODEL_SPECS = [
    {
        "model_name": "single_stage_xgb_target_encoding_features",
        "model_family": "single_stage",
        "feature_set": "xgb_target_encoding_features",
    },
    {
        "model_name": "single_stage_xgb_velocity_ratio_features",
        "model_family": "single_stage",
        "feature_set": "xgb_velocity_ratio_features",
    },
    {
        "model_name": "two_stage_xgb_target_encoding_features",
        "model_family": "two_stage",
        "feature_set": "xgb_target_encoding_features",
    },
    {
        "model_name": "two_stage_xgb_velocity_ratio_features",
        "model_family": "two_stage",
        "feature_set": "xgb_velocity_ratio_features",
    },
]


def make_expanding_time_splits(frame: pd.DataFrame) -> list[tuple[dict[str, int], pd.DataFrame, pd.DataFrame]]:
    splits = []
    for fold in FOLDS:
        train_split = frame[frame["install_day"].between(fold["train_min"], fold["train_max"])].copy()
        valid_split = frame[frame["install_day"].between(fold["valid_min"], fold["valid_max"])].copy()
        if train_split.empty or valid_split.empty:
            raise ValueError(f"Fold {fold['fold']} has empty train or validation split.")
        splits.append((fold, train_split, valid_split))
    return splits


def prepare_fold_features(
    feature_set: str,
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    bucket_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return get_feature_set_frame(feature_set, train_split, valid_split, bucket_features)


def validate_fold_predictions(prediction: pd.Series, valid_rows: int) -> dict[str, Any]:
    values = pd.Series(prediction, dtype=float)
    return {
        "rows_match_valid": int(len(values)) == int(valid_rows),
        "prediction_rows": int(len(values)),
        "valid_rows": int(valid_rows),
        "null_predictions": int(values.isna().sum()),
        "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
        "negative_predictions": int((values < 0).sum()),
    }


def _fit_model_for_fold(
    spec: dict[str, str],
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
):
    if spec["model_family"] == "single_stage":
        result = fit_xgboost_feature_set(
            spec["feature_set"],
            train_split,
            valid_split,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        return result.prediction, result.metrics, {
            "numeric_feature_count": result.metadata.get("numeric_feature_count"),
            "categorical_feature_count": result.metadata.get("categorical_feature_count"),
            "best_iteration": result.metadata.get("best_iteration"),
            "best_score": result.metadata.get("best_score"),
        }
    if spec["model_family"] == "two_stage":
        result = fit_two_stage_feature_set(
            spec["feature_set"],
            train_split,
            valid_split,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        return result.final_pred, result.metrics, {
            "numeric_feature_count": result.metadata.get("numeric_feature_count"),
            "categorical_feature_count": result.metadata.get("categorical_feature_count"),
            "stage1_best_iteration": result.metadata.get("stage1_best_iteration"),
            "stage2_best_iteration": result.metadata.get("stage2_best_iteration"),
            "stage1_best_score": result.metadata.get("stage1_best_score"),
            "stage2_best_score": result.metadata.get("stage2_best_score"),
        }
    raise ValueError(f"Unsupported model family: {spec['model_family']}")


def run_rolling_validation(
    train_model_input: pd.DataFrame,
    bucket_features: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"folds": [], "prediction_checks": []}

    for fold, train_split, valid_split in make_expanding_time_splits(train_model_input):
        diagnostics["folds"].append(
            {
                **fold,
                "train_rows": int(len(train_split)),
                "valid_rows": int(len(valid_split)),
                "train_target_positive_rate": float((train_split[TARGET_COLUMN] > 0).mean()),
                "valid_target_positive_rate": float((valid_split[TARGET_COLUMN] > 0).mean()),
            }
        )
        for spec in MODEL_SPECS:
            fs_train, fs_valid, feature_metadata = prepare_fold_features(
                spec["feature_set"],
                train_split,
                valid_split,
                bucket_features,
            )
            prediction, metrics, model_metadata = _fit_model_for_fold(
                spec,
                fs_train,
                fs_valid,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
            )
            check = validate_fold_predictions(prediction, len(valid_split))
            model_name = spec["model_name"]
            metrics_rows.append(
                {
                    "fold": fold["fold"],
                    "model": model_name,
                    "model_family": spec["model_family"],
                    "feature_set": spec["feature_set"],
                    "train_day_min": fold["train_min"],
                    "train_day_max": fold["train_max"],
                    "valid_day_min": fold["valid_min"],
                    "valid_day_max": fold["valid_max"],
                    "train_rows": int(len(train_split)),
                    "valid_rows": int(len(valid_split)),
                    **metrics,
                    **model_metadata,
                }
            )
            diagnostics["prediction_checks"].append(
                {
                    "fold": fold["fold"],
                    "model": model_name,
                    **check,
                    "target_encoding_fit_scope": feature_metadata.get("note", "no target encoding for this feature set"),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = build_summary(metrics_df)
    return metrics_df, summary_df, diagnostics


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mae",
        "rmse",
        "rmsle",
        "spearman_corr",
        "positive_ltv_rate_pred_top_decile",
        "top_10pct_revenue_capture",
        "top_decile_lift",
    ]
    summary = metrics.groupby(["model", "model_family", "feature_set"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index(drop=True)
    summary["fold_count"] = metrics.groupby("model").size().reindex(summary["model"]).to_numpy()
    summary["best_rmsle_mean"] = summary["rmsle_mean"] == summary["rmsle_mean"].min()
    summary["best_top_capture_mean"] = (
        summary["top_10pct_revenue_capture_mean"] == summary["top_10pct_revenue_capture_mean"].max()
    )
    summary["rmsle_cv"] = summary["rmsle_std"] / summary["rmsle_mean"].replace(0, np.nan)
    summary["top_capture_cv"] = summary["top_10pct_revenue_capture_std"] / summary[
        "top_10pct_revenue_capture_mean"
    ].replace(0, np.nan)
    return summary.sort_values(["rmsle_mean", "top_10pct_revenue_capture_mean"], ascending=[True, False]).reset_index(drop=True)


def _single_holdout_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["fold"] == 4].sort_values("rmsle", ascending=True)


def write_report(path: Path, metrics: pd.DataFrame, summary: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    best_rmsle = summary.sort_values("rmsle_mean", ascending=True).iloc[0]
    best_capture = summary.sort_values("top_10pct_revenue_capture_mean", ascending=False).iloc[0]
    holdout = _single_holdout_rows(metrics)
    holdout_best_rmsle = holdout.iloc[0]
    holdout_best_capture = metrics[metrics["fold"] == 4].sort_values("top_10pct_revenue_capture", ascending=False).iloc[0]
    per_fold_best = metrics.loc[metrics.groupby("fold")["rmsle"].idxmin(), ["fold", "model", "rmsle"]]
    per_fold_capture_best = metrics.loc[
        metrics.groupby("fold")["top_10pct_revenue_capture"].idxmax(),
        ["fold", "model", "top_10pct_revenue_capture"],
    ]

    lines = [
        "# Rolling Time Validation Results",
        "",
        "## Scope",
        "",
        "This report evaluates model stability with expanding install_day time folds. It does not use random KFold, OOF, Optuna, or LightGBM.",
        "",
        "## Folds",
        "",
        "| fold | train days | valid days | train rows | valid rows | train positive rate | valid positive rate |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for fold in diagnostics["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_min']}-{fold['train_max']} | {fold['valid_min']}-{fold['valid_max']} | "
            f"{fold['train_rows']:,} | {fold['valid_rows']:,} | {fold['train_target_positive_rate']:.2%} | {fold['valid_target_positive_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Rolling Summary",
            "",
            "| model | RMSLE mean | RMSLE std | top 10% capture mean | top 10% capture std | Spearman mean | fold count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['rmsle_mean']:.4f} | {row['rmsle_std']:.4f} | "
            f"{row['top_10pct_revenue_capture_mean']:.2%} | {row['top_10pct_revenue_capture_std']:.2%} | "
            f"{row['spearman_corr_mean']:.4f} | {int(row['fold_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| fold | model | RMSLE | RMSE | Spearman | top 10% revenue capture | top-decile lift |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.sort_values(["fold", "model"]).to_dict(orient="records"):
        lines.append(
            f"| {row['fold']} | {row['model']} | {row['rmsle']:.4f} | {row['rmse']:.4f} | "
            f"{row['spearman_corr']:.4f} | {row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"- Single holdout consistency: fold 4 RMSLE winner is `{holdout_best_rmsle['model']}`; rolling mean RMSLE winner is `{best_rmsle['model']}`. Fold 4 top-capture winner is `{holdout_best_capture['model']}`; rolling mean top-capture winner is `{best_capture['model']}`.",
            f"- Most stable by RMSLE mean: `{best_rmsle['model']}` with mean RMSLE {best_rmsle['rmsle_mean']:.6f} and std {best_rmsle['rmsle_std']:.6f}.",
            f"- Most stable by top 10% capture mean: `{best_capture['model']}` with mean capture {best_capture['top_10pct_revenue_capture_mean']:.2%} and std {best_capture['top_10pct_revenue_capture_std']:.2%}.",
            "- Fold-specific RMSLE winners: "
            + "; ".join(f"fold {int(r.fold)} `{r.model}` ({r.rmsle:.4f})" for r in per_fold_best.itertuples(index=False))
            + ".",
            "- Fold-specific top-capture winners: "
            + "; ".join(
                f"fold {int(r.fold)} `{r.model}` ({r.top_10pct_revenue_capture:.2%})"
                for r in per_fold_capture_best.itertuples(index=False)
            )
            + ".",
            f"- Optuna recommendation: tune `{best_rmsle['model']}` for RMSLE and `{best_capture['model']}` for business ranking. If they are the same model, use two objectives on that model first before broadening the search.",
            "",
            "## Leakage Controls",
            "",
            "- Each fold rebuilds target encoding using only that fold's train rows, then maps encodings to the valid rows.",
            "- Each model fit constructs preprocessing and p1/p99 clipping from the fold train matrix only.",
            "- Validation target is used only for early stopping and metric calculation, not for target encoding fit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    bucket_features = build_time_bucket_features_from_zip(Path(args.zip_path), "train.csv")
    metrics, summary, diagnostics = run_rolling_validation(train_model_input, bucket_features)

    metrics.to_csv(processed_dir / "rolling_validation_metrics.csv", index=False)
    summary.to_csv(processed_dir / "rolling_validation_summary.csv", index=False)
    write_report(reports_dir / "rolling_validation_results.md", metrics, summary, diagnostics)


if __name__ == "__main__":
    main()
