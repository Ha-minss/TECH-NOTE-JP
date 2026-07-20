from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN, evaluate_predictions


CATEGORICAL_FEATURES = ["platform", "country_tier", "channel_tier", "top_network", "top_ad_placement"]
EXCLUDED_FEATURES = ["user_id", "install_day", TARGET_COLUMN]
MODEL_NAME = "xgboost_log_target"
class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_ = None
        self.upper_ = None

    def fit(self, x, y=None):
        frame = pd.DataFrame(x)
        self.lower_ = frame.quantile(self.lower_quantile)
        self.upper_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, x):
        frame = pd.DataFrame(x)
        clipped = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return clipped.to_numpy(dtype=float)

@dataclass
class XGBoostResult:
    prediction: pd.Series
    metrics: pd.DataFrame
    pipeline: Pipeline
    feature_config: dict[str, Any]


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


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", QuantileClipper(0.01, 0.99), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


def build_xgb_regressor(n_estimators: int = 500, early_stopping_rounds: int = 50) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        n_jobs=0,
        verbosity=0,
    )


def fit_xgboost_model(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> XGBoostResult:
    numeric_features, categorical_features = build_feature_lists(train_split)
    feature_columns = numeric_features + categorical_features
    x_train = train_split[feature_columns]
    x_valid = valid_split[feature_columns]
    y_train_log = np.log1p(train_split[TARGET_COLUMN].clip(lower=0).astype(float))
    y_valid_log = np.log1p(valid_split[TARGET_COLUMN].clip(lower=0).astype(float))

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_valid_transformed = preprocessor.transform(x_valid)

    model = build_xgb_regressor(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds)
    model.fit(
        x_train_transformed,
        y_train_log,
        eval_set=[(x_valid_transformed, y_valid_log)],
        verbose=False,
    )
    log_pred = model.predict(x_valid_transformed)
    prediction = pd.Series(restore_ltv_scale(log_pred), index=valid_split.index, name=MODEL_NAME)
    metrics = pd.DataFrame([{"model": MODEL_NAME, **evaluate_predictions(valid_split[TARGET_COLUMN], prediction)}])
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    feature_config = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "excluded_features": EXCLUDED_FEATURES,
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1_then_clip_at_zero",
        "numeric_preprocessing": "p1/p99 clipping on train split; no StandardScaler",
        "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore')",
        "validation_split": "train install_day 0-23, valid install_day 24-30",
        "model_params": model.get_params(),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)) if getattr(model, "best_score", None) is not None else None,
    }
    return XGBoostResult(prediction=prediction, metrics=metrics, pipeline=pipeline, feature_config=feature_config)



def build_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            feature_names.extend([str(col) for col in columns])
        elif name == "cat":
            try:
                feature_names.extend(transformer.get_feature_names_out(columns).astype(str).tolist())
            except Exception:
                feature_names.extend([str(col) for col in columns])

    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def write_feature_importance(path: Path, importance: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)


def train_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_split, valid_split = make_time_split(frame)
    if train_split.empty or valid_split.empty:
        raise ValueError("Time-based split failed: train or validation split is empty.")

    result = fit_xgboost_model(train_split, valid_split)
    prediction_frame = valid_split[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
    prediction_frame[f"pred_{MODEL_NAME}"] = result.prediction.values

    values = pd.Series(result.prediction, dtype=float)
    diagnostics = {
        "split": {
            "train_rows": int(len(train_split)),
            "valid_rows": int(len(valid_split)),
            "train_install_day_min": int(train_split["install_day"].min()),
            "train_install_day_max": int(train_split["install_day"].max()),
            "valid_install_day_min": int(valid_split["install_day"].min()),
            "valid_install_day_max": int(valid_split["install_day"].max()),
        },
        "prediction_checks": {
            MODEL_NAME: {
                "rows_match_valid": int(len(values)) == int(len(valid_split)),
                "prediction_rows": int(len(values)),
                "valid_rows": int(len(valid_split)),
                "null_predictions": int(values.isna().sum()),
                "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                "negative_predictions": int((values < 0).sum()),
            }
        },
        "feature_config": result.feature_config,
    }
    importance = build_feature_importance(result.pipeline)
    return result.metrics, prediction_frame, importance, diagnostics


def write_report(
    path: Path,
    metrics_df: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    linear_metrics: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    lines = [
        "# XGBoost Log-Target Baseline Results",
        "",
        "## Scope",
        "",
        "This is the first nonlinear tree-based supervised ML baseline. It uses only time-based validation and does not use LightGBM, Optuna, OOF, KFold, or two-stage modeling.",
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
        "- Model: XGBRegressor objective=`reg:squarederror`, tree_method=`hist`, n_estimators=500, learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, early stopping on valid split.",
        f"- Best iteration: {diagnostics['feature_config']['best_iteration']}",
        f"- Best validation log-RMSE: {diagnostics['feature_config']['best_score']}",
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

    if not linear_metrics.empty:
        lines.extend(["", "## Linear Model Reference", "", "| model | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture |", "|---|---:|---:|---:|---:|---:|"])
        for row in linear_metrics.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
                f"{row['spearman_corr']:.4f} | {row['top_10pct_revenue_capture']:.2%} |"
            )

    if not baseline_metrics.empty:
        lines.extend(["", "## Baseline Reference", "", "| baseline | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture |", "|---|---:|---:|---:|---:|---:|"])
        for row in baseline_metrics.to_dict(orient="records"):
            lines.append(
                f"| {row['baseline']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
                f"{row['spearman_corr']:.4f} | {row['top_10pct_revenue_capture']:.2%} |"
            )

    lines.extend(["", "## Prediction Checks", "", "| model | rows match valid | null predictions | inf predictions | negative predictions |", "|---|---|---:|---:|---:|"])
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
    baseline_metrics_path = processed_dir / "baseline_metrics.csv"
    linear_metrics_path = processed_dir / "linear_model_metrics.csv"
    baseline_metrics = pd.read_csv(baseline_metrics_path) if baseline_metrics_path.exists() else pd.DataFrame()
    linear_metrics = pd.read_csv(linear_metrics_path) if linear_metrics_path.exists() else pd.DataFrame()

    metrics_df, prediction_frame, importance, diagnostics = train_and_evaluate(train_model_input)
    metrics_df.to_csv(processed_dir / "xgboost_model_metrics.csv", index=False)
    prediction_frame.to_parquet(processed_dir / "xgboost_valid_predictions.parquet", index=False)
    write_feature_importance(processed_dir / "xgboost_feature_importance.csv", importance)
    (processed_dir / "xgboost_model_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(reports_dir / "xgboost_model_results.md", metrics_df, baseline_metrics, linear_metrics, diagnostics)


if __name__ == "__main__":
    main()




