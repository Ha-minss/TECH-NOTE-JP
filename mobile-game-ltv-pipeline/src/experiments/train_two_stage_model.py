from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from experiments.run_feature_ablation import (
    build_feature_importance as build_xgb_feature_importance,
    build_preprocessor,
    build_time_bucket_features_from_zip,
    get_feature_set_frame,
    build_feature_lists,
)
from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN, evaluate_predictions


TWO_STAGE_FEATURE_SETS = ["xgb_velocity_ratio_features", "xgb_target_encoding_features"]
EXCLUDED_FEATURES = ["user_id", "install_day", TARGET_COLUMN]


@dataclass
class TwoStageResult:
    feature_set: str
    p_positive: pd.Series
    positive_ltv_pred: pd.Series
    final_pred: pd.Series
    metrics: dict[str, float]
    stage1_diagnostics: dict[str, float]
    stage2_diagnostics: dict[str, float]
    classifier_pipeline: Pipeline
    regressor_pipeline: Pipeline
    metadata: dict[str, Any]


def make_time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = frame[frame["install_day"].between(0, 23)].copy()
    valid_split = frame[frame["install_day"].between(24, 30)].copy()
    return train_split, valid_split


def restore_ltv_scale(log_predictions) -> np.ndarray:
    raw = np.expm1(np.asarray(log_predictions, dtype=float))
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(raw, 0, None)


def build_final_prediction(p_positive: pd.Series, positive_ltv_pred: pd.Series) -> pd.Series:
    p = pd.Series(p_positive, dtype=float).clip(lower=0, upper=1).reset_index(drop=True)
    amount = pd.Series(positive_ltv_pred, dtype=float).clip(lower=0).reset_index(drop=True)
    final = (p * amount).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return final.clip(lower=0)


def build_stage2_training_frame(train_split: pd.DataFrame) -> pd.DataFrame:
    return train_split[train_split[TARGET_COLUMN] > 0].copy()


def prepare_two_stage_feature_frames(
    feature_set: str,
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    bucket_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if feature_set not in TWO_STAGE_FEATURE_SETS:
        raise ValueError(f"Unsupported two-stage feature set: {feature_set}")
    return get_feature_set_frame(feature_set, train_split, valid_split, bucket_features)


def build_xgb_classifier(n_estimators: int = 500, early_stopping_rounds: int = 50) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="logloss",
        n_jobs=0,
        verbosity=0,
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


def _safe_metric(default: float, fn, *args, **kwargs) -> float:
    try:
        value = float(fn(*args, **kwargs))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def compute_stage1_diagnostics(model_name: str, y_true: pd.Series, p_positive: pd.Series) -> dict[str, float | str]:
    y = pd.Series(y_true, dtype=int).reset_index(drop=True)
    p = pd.Series(p_positive, dtype=float).clip(lower=0, upper=1).reset_index(drop=True)
    pred_label = (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred_label,
        average="binary",
        zero_division=0,
    )
    return {
        "model": model_name,
        "roc_auc": _safe_metric(0.0, roc_auc_score, y, p),
        "pr_auc": _safe_metric(float(y.mean()), average_precision_score, y, p),
        "logloss": _safe_metric(0.0, log_loss, y, p, labels=[0, 1]),
        "brier_score": _safe_metric(0.0, brier_score_loss, y, p),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "positive_rate_actual": float(y.mean()),
        "positive_rate_pred_0_5": float(pred_label.mean()),
    }


def compute_stage2_diagnostics(model_name: str, y_true_positive: pd.Series, pred_positive: pd.Series) -> dict[str, float | str]:
    actual = pd.Series(y_true_positive, dtype=float).reset_index(drop=True)
    pred = pd.Series(pred_positive, dtype=float).clip(lower=0).reset_index(drop=True)
    if len(actual) == 0:
        return {
            "model": model_name,
            "positive_valid_rows": 0,
            "positive_only_mae": 0.0,
            "positive_only_rmse": 0.0,
            "positive_only_rmsle": 0.0,
        }
    error = pred - actual
    return {
        "model": model_name,
        "positive_valid_rows": int(len(actual)),
        "positive_only_mae": float(np.mean(np.abs(error))),
        "positive_only_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "positive_only_rmsle": float(np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(actual.clip(lower=0)))))),
    }


def validate_prediction_frame(frame: pd.DataFrame, model_name: str, valid_rows: int) -> dict[str, Any]:
    pred_col = f"pred_{model_name}"
    values = pd.Series(frame[pred_col], dtype=float)
    probabilities = pd.Series(frame["p_positive"], dtype=float)
    return {
        "rows_match_valid": int(len(frame)) == int(valid_rows),
        "prediction_rows": int(len(frame)),
        "valid_rows": int(valid_rows),
        "null_predictions": int(values.isna().sum()),
        "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
        "negative_predictions": int((values < 0).sum()),
        "probability_out_of_range": int(((probabilities < 0) | (probabilities > 1)).sum()),
    }


def _fit_stage1_classifier(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[pd.Series, Pipeline, dict[str, Any]]:
    x_train = train_split[feature_columns]
    x_valid = valid_split[feature_columns]
    y_train = (train_split[TARGET_COLUMN] > 0).astype(int)
    y_valid = (valid_split[TARGET_COLUMN] > 0).astype(int)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_valid_transformed = preprocessor.transform(x_valid)
    classifier = build_xgb_classifier(n_estimators, early_stopping_rounds)
    classifier.fit(x_train_transformed, y_train, eval_set=[(x_valid_transformed, y_valid)], verbose=False)
    p_positive = pd.Series(classifier.predict_proba(x_valid_transformed)[:, 1], index=valid_split.index, name="p_positive")
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])
    metadata = {
        "stage1_best_iteration": int(getattr(classifier, "best_iteration", -1)),
        "stage1_best_score": float(getattr(classifier, "best_score", np.nan)) if getattr(classifier, "best_score", None) is not None else None,
    }
    return p_positive, pipeline, metadata


def _fit_stage2_regressor(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[pd.Series, Pipeline, dict[str, Any]]:
    stage2_train = build_stage2_training_frame(train_split)
    stage2_valid = valid_split[valid_split[TARGET_COLUMN] > 0].copy()
    if stage2_train.empty:
        raise ValueError("Stage 2 training split has no positive LTV rows.")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_transformed = preprocessor.fit_transform(stage2_train[feature_columns])
    x_valid_all_transformed = preprocessor.transform(valid_split[feature_columns])
    y_train_log = np.log1p(stage2_train[TARGET_COLUMN].clip(lower=0).astype(float))

    regressor = build_xgb_regressor(n_estimators, early_stopping_rounds)
    if stage2_valid.empty:
        regressor.fit(x_train_transformed, y_train_log, eval_set=[(x_train_transformed, y_train_log)], verbose=False)
    else:
        x_valid_positive_transformed = preprocessor.transform(stage2_valid[feature_columns])
        y_valid_positive_log = np.log1p(stage2_valid[TARGET_COLUMN].clip(lower=0).astype(float))
        regressor.fit(
            x_train_transformed,
            y_train_log,
            eval_set=[(x_valid_positive_transformed, y_valid_positive_log)],
            verbose=False,
        )

    positive_ltv_pred = pd.Series(
        restore_ltv_scale(regressor.predict(x_valid_all_transformed)),
        index=valid_split.index,
        name="positive_ltv_pred",
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])
    metadata = {
        "stage2_train_positive_rows": int(len(stage2_train)),
        "stage2_valid_positive_rows": int(len(stage2_valid)),
        "stage2_best_iteration": int(getattr(regressor, "best_iteration", -1)),
        "stage2_best_score": float(getattr(regressor, "best_score", np.nan)) if getattr(regressor, "best_score", None) is not None else None,
    }
    return positive_ltv_pred, pipeline, metadata


def fit_two_stage_feature_set(
    feature_set: str,
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> TwoStageResult:
    numeric_features, categorical_features = build_feature_lists(train_split)
    feature_columns = numeric_features + categorical_features
    model_name = f"two_stage_{feature_set}"

    p_positive, classifier_pipeline, stage1_meta = _fit_stage1_classifier(
        train_split,
        valid_split,
        feature_columns,
        numeric_features,
        categorical_features,
        n_estimators,
        early_stopping_rounds,
    )
    positive_ltv_pred, regressor_pipeline, stage2_meta = _fit_stage2_regressor(
        train_split,
        valid_split,
        feature_columns,
        numeric_features,
        categorical_features,
        n_estimators,
        early_stopping_rounds,
    )
    final_pred = build_final_prediction(p_positive, positive_ltv_pred)
    final_pred.index = valid_split.index
    final_pred.name = model_name

    positive_valid_mask = valid_split[TARGET_COLUMN] > 0
    stage2_valid_pred = pd.Series(positive_ltv_pred, index=valid_split.index).loc[positive_valid_mask]
    metadata = {
        "feature_set": feature_set,
        "model_name": model_name,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        **stage1_meta,
        **stage2_meta,
    }
    return TwoStageResult(
        feature_set=feature_set,
        p_positive=p_positive,
        positive_ltv_pred=positive_ltv_pred,
        final_pred=final_pred,
        metrics=evaluate_predictions(valid_split[TARGET_COLUMN], final_pred),
        stage1_diagnostics=compute_stage1_diagnostics(model_name, positive_valid_mask.astype(int), p_positive),
        stage2_diagnostics=compute_stage2_diagnostics(
            model_name,
            valid_split.loc[positive_valid_mask, TARGET_COLUMN],
            stage2_valid_pred,
        ),
        classifier_pipeline=classifier_pipeline,
        regressor_pipeline=regressor_pipeline,
        metadata=metadata,
    )


def build_two_stage_feature_importance(result: TwoStageResult) -> pd.DataFrame:
    stage1 = build_xgb_feature_importance(result.classifier_pipeline, result.feature_set).assign(stage="stage1_classifier")
    stage2 = build_xgb_feature_importance(result.regressor_pipeline, result.feature_set).assign(stage="stage2_regressor")
    stage1["model"] = f"two_stage_{result.feature_set}"
    stage2["model"] = f"two_stage_{result.feature_set}"
    return pd.concat([stage1, stage2], ignore_index=True)[["model", "feature_set", "stage", "feature", "importance"]]


def run_two_stage_models(
    train_model_input: pd.DataFrame,
    bucket_features: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_split, valid_split = make_time_split(train_model_input)
    if train_split.empty or valid_split.empty:
        raise ValueError("Time-based split failed: train or validation split is empty.")

    prediction_frame = valid_split[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
    metrics_rows: list[dict[str, Any]] = []
    stage1_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "split": {
            "train_rows": int(len(train_split)),
            "valid_rows": int(len(valid_split)),
            "train_install_day_min": int(train_split["install_day"].min()),
            "train_install_day_max": int(train_split["install_day"].max()),
            "valid_install_day_min": int(valid_split["install_day"].min()),
            "valid_install_day_max": int(valid_split["install_day"].max()),
        },
        "feature_sets": {},
        "prediction_checks": {},
    }

    for feature_set in TWO_STAGE_FEATURE_SETS:
        fs_train, fs_valid, feature_meta = prepare_two_stage_feature_frames(feature_set, train_split, valid_split, bucket_features)
        result = fit_two_stage_feature_set(feature_set, fs_train, fs_valid, n_estimators, early_stopping_rounds)
        model_name = f"two_stage_{feature_set}"

        metrics_rows.append({"model": model_name, "feature_set": feature_set, **result.metrics})
        stage1_rows.append(result.stage1_diagnostics)
        stage2_rows.append(result.stage2_diagnostics)
        prediction_frame[f"p_positive_{feature_set}"] = result.p_positive.values
        prediction_frame[f"positive_ltv_pred_{feature_set}"] = result.positive_ltv_pred.values
        prediction_frame[f"pred_{model_name}"] = result.final_pred.values
        importance_frames.append(build_two_stage_feature_importance(result))

        check_frame = pd.DataFrame(
            {
                "p_positive": result.p_positive.values,
                "positive_ltv_pred": result.positive_ltv_pred.values,
                f"pred_{model_name}": result.final_pred.values,
            }
        )
        diagnostics["prediction_checks"][model_name] = validate_prediction_frame(check_frame, model_name, len(valid_split))
        diagnostics["feature_sets"][feature_set] = {**feature_meta, **result.metadata}

    return (
        pd.DataFrame(metrics_rows),
        prediction_frame,
        pd.DataFrame(stage1_rows),
        pd.DataFrame(stage2_rows),
        pd.concat(importance_frames, ignore_index=True),
        diagnostics,
    )


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _comparison_rows(processed_dir: Path, two_stage_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_metrics = _read_csv_if_exists(processed_dir / "feature_engineering_metrics.csv")
    if not feature_metrics.empty:
        for feature_set in ["xgb_current_full", "xgb_velocity_ratio_features", "xgb_target_encoding_features"]:
            match = feature_metrics[feature_metrics["feature_set"] == feature_set]
            if not match.empty:
                row = match.iloc[0].to_dict()
                rows.append({"model": feature_set, "group": "single_stage_xgboost", **row})

    linear_metrics = _read_csv_if_exists(processed_dir / "linear_model_metrics.csv")
    if not linear_metrics.empty:
        match = linear_metrics[linear_metrics["model"] == "ridge_log_linear"]
        if not match.empty:
            rows.append({"group": "linear", **match.iloc[0].to_dict()})

    baseline_metrics = _read_csv_if_exists(processed_dir / "baseline_metrics.csv")
    if not baseline_metrics.empty:
        match = baseline_metrics[baseline_metrics["baseline"] == "early_revenue_multiplier"]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["model"] = row.pop("baseline")
            rows.append({"group": "baseline", **row})

    for row in two_stage_metrics.to_dict(orient="records"):
        rows.append({"group": "two_stage", **row})
    return rows


def write_report(
    path: Path,
    two_stage_metrics: pd.DataFrame,
    stage1_diagnostics: pd.DataFrame,
    stage2_diagnostics: pd.DataFrame,
    diagnostics: dict[str, Any],
    processed_dir: Path,
) -> None:
    comparison = pd.DataFrame(_comparison_rows(processed_dir, two_stage_metrics))
    best_two_stage_rmsle = two_stage_metrics.sort_values("rmsle", ascending=True).iloc[0]
    best_two_stage_capture = two_stage_metrics.sort_values("top_10pct_revenue_capture", ascending=False).iloc[0]
    best_single_rmsle = 0.5404
    best_single_capture = 0.7660

    lines = [
        "# Two-Stage LTV Model Results",
        "",
        "## Scope",
        "",
        "This compares a two-stage XGBoost model against existing baselines on the same time-based validation split. It does not use Optuna, OOF, random KFold, or LightGBM.",
        "",
        "## Validation Split",
        "",
        f"- Train split: install_day {diagnostics['split']['train_install_day_min']} to {diagnostics['split']['train_install_day_max']} ({diagnostics['split']['train_rows']:,} rows)",
        f"- Valid split: install_day {diagnostics['split']['valid_install_day_min']} to {diagnostics['split']['valid_install_day_max']} ({diagnostics['split']['valid_rows']:,} rows)",
        "",
        "## Final Metrics",
        "",
        "| model | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in two_stage_metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
            f"{row['spearman_corr']:.4f} | {row['positive_ltv_rate_pred_top_decile']:.2%} | "
            f"{row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
        )

    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Comparison Reference",
                "",
                "| group | model | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture | top-decile lift |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparison.to_dict(orient="records"):
            model = row.get("model", row.get("feature_set", "unknown"))
            lines.append(
                f"| {row.get('group', '')} | {model} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
                f"{row['spearman_corr']:.4f} | {row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
            )

    lines.extend(["", "## Stage 1 Diagnostics", "", "| model | ROC-AUC | PR-AUC | LogLoss | Brier | Precision@0.5 | Recall@0.5 | F1@0.5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in stage1_diagnostics.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['logloss']:.4f} | "
            f"{row['brier_score']:.4f} | {row['precision_at_0_5']:.4f} | {row['recall_at_0_5']:.4f} | {row['f1_at_0_5']:.4f} |"
        )

    lines.extend(["", "## Stage 2 Diagnostics", "", "| model | positive valid rows | positive-only MAE | positive-only RMSE | positive-only RMSLE |", "|---|---:|---:|---:|---:|"])
    for row in stage2_diagnostics.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {int(row['positive_valid_rows']):,} | {row['positive_only_mae']:.4f} | "
            f"{row['positive_only_rmse']:.4f} | {row['positive_only_rmsle']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"- Is two-stage better than best single-stage RMSLE 0.5404? {'Yes' if best_two_stage_rmsle['rmsle'] < best_single_rmsle else 'No'}; best two-stage RMSLE is {best_two_stage_rmsle['rmsle']:.6f} from `{best_two_stage_rmsle['model']}`.",
            f"- Is two-stage better than best single-stage top 10% revenue capture 76.60%? {'Yes' if best_two_stage_capture['top_10pct_revenue_capture'] > best_single_capture else 'No'}; best two-stage capture is {best_two_stage_capture['top_10pct_revenue_capture']:.2%} from `{best_two_stage_capture['model']}`.",
            "- Zero-heavy target impact: compare RMSLE, Spearman, and top-decile capture together; final prediction multiplies propensity by conditional positive value, so it may improve calibration while sometimes softening high-value ranking.",
            "- Stage 1 quality: ROC-AUC/PR-AUC show whether positive LTV users are separable before amount prediction.",
            "- Stage 2 stability: positive-only MAE/RMSE/RMSLE describe amount prediction only among positive valid users, separated from the zero classification task.",
            "",
            "## Prediction Checks",
            "",
            "| model | rows match valid | null predictions | inf predictions | negative predictions | probability out of range |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model, check in diagnostics["prediction_checks"].items():
        lines.append(
            f"| {model} | {check['rows_match_valid']} | {check['null_predictions']:,} | "
            f"{check['inf_predictions']:,} | {check['negative_predictions']:,} | {check['probability_out_of_range']:,} |"
        )
    lines.append("")
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
    metrics, predictions, stage1, stage2, importance, diagnostics = run_two_stage_models(train_model_input, bucket_features)

    metrics.to_csv(processed_dir / "two_stage_metrics.csv", index=False)
    predictions.to_parquet(processed_dir / "two_stage_valid_predictions.parquet", index=False)
    stage1.to_csv(processed_dir / "two_stage_stage1_diagnostics.csv", index=False)
    stage2.to_csv(processed_dir / "two_stage_stage2_diagnostics.csv", index=False)
    importance.to_csv(processed_dir / "two_stage_feature_importance.csv", index=False)
    (processed_dir / "two_stage_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(reports_dir / "two_stage_model_results.md", metrics, stage1, stage2, diagnostics, processed_dir)


if __name__ == "__main__":
    main()
