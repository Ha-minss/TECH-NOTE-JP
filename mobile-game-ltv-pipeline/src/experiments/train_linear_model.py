from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN, evaluate_predictions


CATEGORICAL_FEATURES = ["platform", "country_tier", "channel_tier", "top_network", "top_ad_placement"]
EXCLUDED_FEATURES = ["user_id", "install_day", TARGET_COLUMN]
MODEL_NAMES = ["ridge_log_linear", "elasticnet_log_linear"]


@dataclass
class LinearModelResult:
    predictions: dict[str, pd.Series]
    metrics: pd.DataFrame
    feature_config: dict[str, Any]


class QuantileClipper:
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None

    def fit(self, x, y=None):
        frame = pd.DataFrame(x)
        self.lower_ = frame.quantile(self.lower_quantile)
        self.upper_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, x):
        frame = pd.DataFrame(x)
        clipped = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return clipped.to_numpy(dtype=float)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


def make_time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = frame[frame["install_day"].between(0, 23)].copy()
    valid_split = frame[frame["install_day"].between(24, 30)].copy()
    return train_split, valid_split


def build_feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in frame.columns]
    excluded = set(EXCLUDED_FEATURES)
    numeric_features = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col not in excluded and col not in categorical_features
    ]
    return numeric_features, categorical_features


def restore_ltv_scale(log_predictions) -> np.ndarray:
    raw = np.expm1(np.asarray(log_predictions, dtype=float))
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(raw, 0, None)


def build_pipeline(model, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("clip", QuantileClipper(0.01, 0.99)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def fit_linear_models(train_split: pd.DataFrame, valid_split: pd.DataFrame) -> LinearModelResult:
    numeric_features, categorical_features = build_feature_lists(train_split)
    x_train = train_split[numeric_features + categorical_features]
    x_valid = valid_split[numeric_features + categorical_features]
    y_train_log = np.log1p(train_split[TARGET_COLUMN].clip(lower=0).astype(float))

    model_specs = {
        "ridge_log_linear": Ridge(alpha=1.0, random_state=42),
        "elasticnet_log_linear": ElasticNet(alpha=0.01, l1_ratio=0.15, max_iter=20000, tol=1e-4, random_state=42),
    }
    predictions: dict[str, pd.Series] = {}
    metrics_rows: list[dict[str, Any]] = []

    for name, model in model_specs.items():
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        pipeline.fit(x_train, y_train_log)
        log_pred = pipeline.predict(x_valid)
        pred = pd.Series(restore_ltv_scale(log_pred), index=valid_split.index, name=name)
        predictions[name] = pred
        metrics_rows.append({"model": name, **evaluate_predictions(valid_split[TARGET_COLUMN], pred)})

    feature_config = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "excluded_features": EXCLUDED_FEATURES,
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1_then_clip_at_zero",
        "numeric_preprocessing": "p1/p99 clipping on train split, then StandardScaler",
        "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore')",
        "validation_split": "train install_day 0-23, valid install_day 24-30",
    }
    return LinearModelResult(predictions=predictions, metrics=pd.DataFrame(metrics_rows), feature_config=feature_config)


def train_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_split, valid_split = make_time_split(frame)
    if train_split.empty or valid_split.empty:
        raise ValueError("Time-based split failed: train or validation split is empty.")

    result = fit_linear_models(train_split, valid_split)
    prediction_frame = valid_split[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
    for name, pred in result.predictions.items():
        prediction_frame[f"pred_{name}"] = pred.values

    prediction_checks = {}
    for name, pred in result.predictions.items():
        values = pd.Series(pred, dtype=float)
        prediction_checks[name] = {
            "rows_match_valid": int(len(values)) == int(len(valid_split)),
            "prediction_rows": int(len(values)),
            "valid_rows": int(len(valid_split)),
            "null_predictions": int(values.isna().sum()),
            "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
            "negative_predictions": int((values < 0).sum()),
        }

    diagnostics = {
        "split": {
            "train_rows": int(len(train_split)),
            "valid_rows": int(len(valid_split)),
            "train_install_day_min": int(train_split["install_day"].min()),
            "train_install_day_max": int(train_split["install_day"].max()),
            "valid_install_day_min": int(valid_split["install_day"].min()),
            "valid_install_day_max": int(valid_split["install_day"].max()),
        },
        "prediction_checks": prediction_checks,
        "feature_config": result.feature_config,
    }
    return result.metrics, prediction_frame, diagnostics


def write_report(path: Path, metrics_df: pd.DataFrame, baseline_metrics: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    lines = [
        "# Linear Log-Target Model Results",
        "",
        "## Scope",
        "",
        "This is the first supervised ML baseline after non-ML baselines. It uses only time-based validation and does not use OOF, random KFold, LightGBM, XGBoost, Optuna, or two-stage modeling.",
        "",
        "## Validation Split",
        "",
        f"- Train split: install_day {diagnostics['split']['train_install_day_min']} to {diagnostics['split']['train_install_day_max']} ({diagnostics['split']['train_rows']:,} rows)",
        f"- Valid split: install_day {diagnostics['split']['valid_install_day_min']} to {diagnostics['split']['valid_install_day_max']} ({diagnostics['split']['valid_rows']:,} rows)",
        "",
        "## Feature Setup",
        "",
        f"- Categorical: `{diagnostics['feature_config']['categorical_features']}`",
        f"- Numeric feature count: {len(diagnostics['feature_config']['numeric_features'])}",
        f"- Excluded from model features: `{diagnostics['feature_config']['excluded_features']}`",
        f"- Numeric preprocessing: {diagnostics['feature_config']['numeric_preprocessing']}",
        f"- Categorical preprocessing: {diagnostics['feature_config']['categorical_preprocessing']}",
        f"- Target transform: {diagnostics['feature_config']['target_transform']}; predictions use {diagnostics['feature_config']['prediction_inverse_transform']}",
        "",
        "## Metrics",
        "",
        "| model | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
            f"{row['spearman_corr']:.4f} | {row['positive_ltv_rate_pred_top_decile']:.2%} | "
            f"{row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
        )

    if not baseline_metrics.empty:
        lines.extend(
            [
                "",
                "## Baseline Reference",
                "",
                "| baseline | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture | top-decile lift |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in baseline_metrics.to_dict(orient="records"):
            lines.append(
                f"| {row['baseline']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
                f"{row['spearman_corr']:.4f} | {row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Prediction Checks",
            "",
            "| model | rows match valid | null predictions | inf predictions | negative predictions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model, check in diagnostics["prediction_checks"].items():
        lines.append(
            f"| {model} | {check['rows_match_valid']} | {check['null_predictions']:,} | "
            f"{check['inf_predictions']:,} | {check['negative_predictions']:,} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    baseline_path = processed_dir / "baseline_metrics.csv"
    baseline_metrics = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    metrics_df, prediction_frame, diagnostics = train_and_evaluate(train_model_input)

    metrics_df.to_csv(processed_dir / "linear_model_metrics.csv", index=False)
    prediction_frame.to_parquet(processed_dir / "linear_valid_predictions.parquet", index=False)
    (processed_dir / "linear_model_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(reports_dir / "linear_model_results.md", metrics_df, baseline_metrics, diagnostics)


if __name__ == "__main__":
    main()

