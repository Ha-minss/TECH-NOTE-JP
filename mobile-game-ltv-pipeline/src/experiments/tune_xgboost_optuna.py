from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from experiments.run_feature_ablation import (
    build_feature_importance,
    build_feature_lists,
    build_preprocessor,
    build_time_bucket_features_from_zip,
    get_feature_set_frame,
    restore_ltv_scale,
)
from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN, evaluate_predictions
from experiments.train_two_stage_model import (
    build_final_prediction,
    compute_stage1_diagnostics,
    compute_stage2_diagnostics,
)


TARGET_FEATURE_SET = "xgb_target_encoding_features"
BASELINE_SINGLE_STAGE_RMSLE = 0.5404
BASELINE_TWO_STAGE_TOP_CAPTURE = 0.7766
TUNING_FOLDS = [
    {"fold": 1, "train_min": 0, "train_max": 13, "valid_min": 14, "valid_max": 16},
    {"fold": 2, "train_min": 0, "train_max": 16, "valid_min": 17, "valid_max": 19},
    {"fold": 3, "train_min": 0, "train_max": 19, "valid_min": 20, "valid_max": 23},
]
FINAL_HOLDOUT = {"fold": 4, "train_min": 0, "train_max": 23, "valid_min": 24, "valid_max": 30}
MODEL_OBJECTIVES = [
    {"model": "optuna_single_stage_rmsle", "model_family": "single_stage", "objective": "rmsle"},
    {"model": "optuna_two_stage_top_capture", "model_family": "two_stage", "objective": "top_capture"},
]


@dataclass
class FoldData:
    fold: int
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    x_train: Any
    x_valid: Any
    y_train_raw: pd.Series
    y_valid_raw: pd.Series
    y_train_log: pd.Series
    y_valid_log: pd.Series
    y_train_positive: pd.Series
    y_valid_positive: pd.Series
    preprocessor: Any


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required. Install it with `python -m pip install optuna`.") from exc
    return optuna


def build_param_dict(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "tree_method": "hist",
        "random_state": 42,
    }


def objective_value_from_metrics(metrics: dict[str, float], objective: str) -> float:
    if objective == "rmsle":
        return float(metrics["rmsle"])
    if objective == "top_capture":
        return -float(metrics["top_10pct_revenue_capture"])
    raise ValueError(f"Unknown objective: {objective}")


def prepare_target_encoding_splits(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    bucket_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return get_feature_set_frame(TARGET_FEATURE_SET, train_split, valid_split, bucket_features)


def make_time_split_for_fold(frame: pd.DataFrame, fold: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = frame[frame["install_day"].between(fold["train_min"], fold["train_max"])].copy()
    valid_split = frame[frame["install_day"].between(fold["valid_min"], fold["valid_max"])].copy()
    if train_split.empty or valid_split.empty:
        raise ValueError(f"Fold {fold['fold']} has empty train or validation split.")
    return train_split, valid_split


def prepare_fold_data(
    frame: pd.DataFrame,
    bucket_features: pd.DataFrame,
    folds: list[dict[str, int]],
) -> list[FoldData]:
    fold_data: list[FoldData] = []
    for fold in folds:
        raw_train, raw_valid = make_time_split_for_fold(frame, fold)
        train_split, valid_split, _ = prepare_target_encoding_splits(raw_train, raw_valid, bucket_features)
        numeric_features, categorical_features = build_feature_lists(train_split)
        feature_columns = numeric_features + categorical_features
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        x_train = preprocessor.fit_transform(train_split[feature_columns])
        x_valid = preprocessor.transform(valid_split[feature_columns])
        y_train_raw = train_split[TARGET_COLUMN].clip(lower=0).astype(float)
        y_valid_raw = valid_split[TARGET_COLUMN].clip(lower=0).astype(float)
        fold_data.append(
            FoldData(
                fold=fold["fold"],
                train_frame=train_split,
                valid_frame=valid_split,
                feature_columns=feature_columns,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                x_train=x_train,
                x_valid=x_valid,
                y_train_raw=y_train_raw,
                y_valid_raw=y_valid_raw,
                y_train_log=np.log1p(y_train_raw),
                y_valid_log=np.log1p(y_valid_raw),
                y_train_positive=(y_train_raw > 0).astype(int),
                y_valid_positive=(y_valid_raw > 0).astype(int),
                preprocessor=preprocessor,
            )
        )
    return fold_data


def build_regressor(params: dict[str, Any], early_stopping_rounds: int) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        n_jobs=0,
        verbosity=0,
    )


def build_classifier(params: dict[str, Any], early_stopping_rounds: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="logloss",
        n_jobs=0,
        verbosity=0,
    )


def fit_predict_single_stage(
    fold_data: FoldData,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> pd.Series:
    model = build_regressor(params, early_stopping_rounds)
    model.fit(fold_data.x_train, fold_data.y_train_log, eval_set=[(fold_data.x_valid, fold_data.y_valid_log)], verbose=False)
    return pd.Series(restore_ltv_scale(model.predict(fold_data.x_valid)), index=fold_data.valid_frame.index)


def fit_predict_two_stage(
    fold_data: FoldData,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    classifier = build_classifier(params, early_stopping_rounds)
    classifier.fit(
        fold_data.x_train,
        fold_data.y_train_positive,
        eval_set=[(fold_data.x_valid, fold_data.y_valid_positive)],
        verbose=False,
    )
    p_positive = pd.Series(classifier.predict_proba(fold_data.x_valid)[:, 1], index=fold_data.valid_frame.index)

    positive_train_mask = fold_data.y_train_raw > 0
    positive_valid_mask = fold_data.y_valid_raw > 0
    if not positive_train_mask.any():
        raise ValueError(f"Fold {fold_data.fold} has no positive train rows for stage 2.")
    x_stage2_train = fold_data.x_train[positive_train_mask.to_numpy()]
    y_stage2_train = fold_data.y_train_log[positive_train_mask]
    x_stage2_valid = fold_data.x_valid[positive_valid_mask.to_numpy()] if positive_valid_mask.any() else x_stage2_train
    y_stage2_valid = fold_data.y_valid_log[positive_valid_mask] if positive_valid_mask.any() else y_stage2_train
    regressor = build_regressor(params, early_stopping_rounds)
    regressor.fit(x_stage2_train, y_stage2_train, eval_set=[(x_stage2_valid, y_stage2_valid)], verbose=False)
    positive_pred = pd.Series(restore_ltv_scale(regressor.predict(fold_data.x_valid)), index=fold_data.valid_frame.index)
    final_pred = build_final_prediction(p_positive, positive_pred)
    final_pred.index = fold_data.valid_frame.index
    return p_positive, positive_pred, final_pred


def evaluate_params_on_folds(
    model_family: str,
    objective: str,
    params: dict[str, Any],
    folds: list[FoldData],
    early_stopping_rounds: int,
) -> tuple[float, list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for fold_data in folds:
        if model_family == "single_stage":
            pred = fit_predict_single_stage(fold_data, params, early_stopping_rounds)
        elif model_family == "two_stage":
            _, _, pred = fit_predict_two_stage(fold_data, params, early_stopping_rounds)
        else:
            raise ValueError(f"Unknown model family: {model_family}")
        metrics = evaluate_predictions(fold_data.y_valid_raw, pred)
        fold_rows.append({"fold": fold_data.fold, **metrics})

    mean_metrics = {
        key: float(np.mean([row[key] for row in fold_rows]))
        for key in ["mae", "rmse", "rmsle", "spearman_corr", "positive_ltv_rate_pred_top_decile", "top_10pct_revenue_capture", "top_decile_lift"]
    }
    return objective_value_from_metrics(mean_metrics, objective), fold_rows


def run_study(
    model_name: str,
    model_family: str,
    objective: str,
    folds: list[FoldData],
    n_trials: int,
    early_stopping_rounds: int,
    seed: int,
):
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=model_name)

    def objective_fn(trial):
        params = build_param_dict(trial)
        value, fold_rows = evaluate_params_on_folds(model_family, objective, params, folds, early_stopping_rounds)
        for metric_name in ["rmsle", "top_10pct_revenue_capture"]:
            trial.set_user_attr(f"mean_{metric_name}", float(np.mean([row[metric_name] for row in fold_rows])))
        return value

    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study


def _trial_rows(study, model_name: str, model_family: str, objective: str) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {
            "model": model_name,
            "model_family": model_family,
            "objective": objective,
            "trial_number": trial.number,
            "objective_value": trial.value,
            "state": str(trial.state),
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final_single_stage(
    fold_data: FoldData,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> tuple[pd.Series, Pipeline]:
    model = build_regressor(params, early_stopping_rounds)
    model.fit(fold_data.x_train, fold_data.y_train_log, eval_set=[(fold_data.x_valid, fold_data.y_valid_log)], verbose=False)
    pred = pd.Series(restore_ltv_scale(model.predict(fold_data.x_valid)), index=fold_data.valid_frame.index)
    pipeline = Pipeline(steps=[("preprocess", fold_data.preprocessor), ("model", model)])
    return pred, pipeline


def fit_final_two_stage(
    fold_data: FoldData,
    params: dict[str, Any],
    early_stopping_rounds: int,
) -> tuple[pd.Series, pd.Series, pd.Series, Pipeline, Pipeline]:
    classifier = build_classifier(params, early_stopping_rounds)
    classifier.fit(
        fold_data.x_train,
        fold_data.y_train_positive,
        eval_set=[(fold_data.x_valid, fold_data.y_valid_positive)],
        verbose=False,
    )
    p_positive = pd.Series(classifier.predict_proba(fold_data.x_valid)[:, 1], index=fold_data.valid_frame.index)
    positive_train_mask = fold_data.y_train_raw > 0
    positive_valid_mask = fold_data.y_valid_raw > 0
    regressor = build_regressor(params, early_stopping_rounds)
    regressor.fit(
        fold_data.x_train[positive_train_mask.to_numpy()],
        fold_data.y_train_log[positive_train_mask],
        eval_set=[(fold_data.x_valid[positive_valid_mask.to_numpy()], fold_data.y_valid_log[positive_valid_mask])],
        verbose=False,
    )
    positive_pred = pd.Series(restore_ltv_scale(regressor.predict(fold_data.x_valid)), index=fold_data.valid_frame.index)
    final_pred = build_final_prediction(p_positive, positive_pred)
    final_pred.index = fold_data.valid_frame.index
    return (
        p_positive,
        positive_pred,
        final_pred,
        Pipeline(steps=[("preprocess", fold_data.preprocessor), ("model", classifier)]),
        Pipeline(steps=[("preprocess", fold_data.preprocessor), ("model", regressor)]),
    )


def build_valid_predictions_frame(valid_frame: pd.DataFrame, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    out = valid_frame[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
    for name, pred in predictions.items():
        out[f"pred_{name}"] = pd.Series(pred).to_numpy()
    return out


def summarize_best_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    single = metrics[metrics["model"] == "optuna_single_stage_rmsle"].iloc[0]
    two_stage = metrics[metrics["model"] == "optuna_two_stage_top_capture"].iloc[0]
    return {
        "single_stage_improved_rmsle": bool(single["rmsle"] < BASELINE_SINGLE_STAGE_RMSLE),
        "two_stage_improved_top_capture": bool(two_stage["top_10pct_revenue_capture"] > BASELINE_TWO_STAGE_TOP_CAPTURE),
        "single_stage_rmsle_delta": float(BASELINE_SINGLE_STAGE_RMSLE - single["rmsle"]),
        "two_stage_top_capture_delta": float(two_stage["top_10pct_revenue_capture"] - BASELINE_TWO_STAGE_TOP_CAPTURE),
    }


def build_best_params_payload(best_params: dict[str, dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "settings": settings,
        "best_params": best_params,
        "fixed": {
            "feature_set": TARGET_FEATURE_SET,
            "tree_method": "hist",
            "random_state": 42,
            "single_stage_objective": "reg:squarederror",
            "two_stage_classifier_objective": "binary:logistic",
            "two_stage_regressor_objective": "reg:squarederror",
            "validation": "Optuna folds 1-3 only; fold 4 final holdout only",
        },
    }


def run_tuning(
    train_model_input: pd.DataFrame,
    bucket_features: pd.DataFrame,
    n_trials: int,
    early_stopping_rounds: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tuning_folds = prepare_fold_data(train_model_input, bucket_features, TUNING_FOLDS)
    final_fold = prepare_fold_data(train_model_input, bucket_features, [FINAL_HOLDOUT])[0]

    trial_frames: list[pd.DataFrame] = []
    final_metric_rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.Series] = {}
    best_params: dict[str, dict[str, Any]] = {}

    for spec in MODEL_OBJECTIVES:
        study = run_study(
            spec["model"],
            spec["model_family"],
            spec["objective"],
            tuning_folds,
            n_trials=n_trials,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        params = {**build_param_dict_from_best(study.best_params)}
        best_params[spec["model"]] = params
        trial_frames.append(_trial_rows(study, spec["model"], spec["model_family"], spec["objective"]))

        if spec["model_family"] == "single_stage":
            pred, _ = fit_final_single_stage(final_fold, params, early_stopping_rounds)
            metrics = evaluate_predictions(final_fold.y_valid_raw, pred)
            final_metric_rows.append({"model": spec["model"], "model_family": spec["model_family"], "objective": spec["objective"], **metrics})
            predictions[spec["model"]] = pred
        else:
            p_positive, positive_pred, pred, _, _ = fit_final_two_stage(final_fold, params, early_stopping_rounds)
            metrics = evaluate_predictions(final_fold.y_valid_raw, pred)
            final_metric_rows.append({"model": spec["model"], "model_family": spec["model_family"], "objective": spec["objective"], **metrics})
            predictions[spec["model"]] = pred
            predictions[f"{spec['model']}_p_positive"] = p_positive
            predictions[f"{spec['model']}_positive_ltv_pred"] = positive_pred

    metrics_df = pd.DataFrame(final_metric_rows)
    trials_df = pd.concat(trial_frames, ignore_index=True)
    prediction_frame = build_valid_predictions_frame(final_fold.valid_frame, predictions)
    payload = build_best_params_payload(
        best_params,
        {"n_trials": n_trials, "early_stopping_rounds": early_stopping_rounds, "seed": seed},
    )
    return trials_df, metrics_df, prediction_frame, payload


def build_param_dict_from_best(best_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(best_params)
    params["tree_method"] = "hist"
    params["random_state"] = 42
    return params


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _comparison_rows(processed_dir: Path, tuned_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature = _read_csv_if_exists(processed_dir / "feature_engineering_metrics.csv")
    if not feature.empty:
        for fs in ["xgb_current_full", "xgb_target_encoding_features"]:
            match = feature[feature["feature_set"] == fs]
            if not match.empty:
                row = match.iloc[0].to_dict()
                rows.append({"group": "untuned_xgboost", "model": fs, **row})
    two_stage = _read_csv_if_exists(processed_dir / "two_stage_metrics.csv")
    if not two_stage.empty:
        for model in ["two_stage_xgb_target_encoding_features"]:
            match = two_stage[two_stage["model"] == model]
            if not match.empty:
                rows.append({"group": "untuned_two_stage", **match.iloc[0].to_dict()})
    linear = _read_csv_if_exists(processed_dir / "linear_model_metrics.csv")
    if not linear.empty:
        for model in ["ridge_log_linear", "elasticnet_log_linear"]:
            match = linear[linear["model"] == model]
            if not match.empty:
                rows.append({"group": "linear", **match.iloc[0].to_dict()})
    baseline = _read_csv_if_exists(processed_dir / "baseline_metrics.csv")
    if not baseline.empty:
        match = baseline[baseline["baseline"] == "early_revenue_multiplier"]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["model"] = row.pop("baseline")
            rows.append({"group": "baseline", **row})
    for row in tuned_metrics.to_dict(orient="records"):
        rows.append({"group": "optuna_tuned", **row})
    return rows


def write_report(path: Path, trials: pd.DataFrame, metrics: pd.DataFrame, payload: dict[str, Any], processed_dir: Path) -> None:
    summary = summarize_best_metrics(metrics)
    comparison = pd.DataFrame(_comparison_rows(processed_dir, metrics))
    single_trials = trials[trials["model"] == "optuna_single_stage_rmsle"].sort_values("objective_value")
    two_trials = trials[trials["model"] == "optuna_two_stage_top_capture"].sort_values("objective_value")
    single_best_cv = single_trials.iloc[0]
    two_best_cv = two_trials.iloc[0]

    lines = [
        "# Optuna XGBoost Tuning Results",
        "",
        "## Scope",
        "",
        "This tunes only the two rolling-validated candidate models. It does not use AutoML, random KFold, OOF, or LightGBM.",
        "",
        "## Tuning Setup",
        "",
        f"- Trials per study: {payload['settings']['n_trials']}",
        f"- Early stopping rounds: {payload['settings']['early_stopping_rounds']}",
        "- Tuning folds: install_day 0-13 -> 14-16, 0-16 -> 17-19, 0-19 -> 20-23.",
        "- Final holdout: install_day 0-23 -> 24-30, not used in Optuna objective.",
        "- Feature set: target_encoding_features; target encodings are fit within each train fold only.",
        "",
        "## Final Holdout Metrics",
        "",
        "| model | objective | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {row['objective']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
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
            lines.append(
                f"| {row.get('group', '')} | {row.get('model', row.get('feature_set', 'unknown'))} | {row['mae']:.4f} | "
                f"{row['rmse']:.4f} | {row['rmsle']:.4f} | {row['spearman_corr']:.4f} | "
                f"{row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"- Did tuned single-stage improve RMSLE 0.5404? {'Yes' if summary['single_stage_improved_rmsle'] else 'No'}; delta={summary['single_stage_rmsle_delta']:.6f}.",
            f"- Did tuned two-stage improve top 10% capture 77.66%? {'Yes' if summary['two_stage_improved_top_capture'] else 'No'}; delta={summary['two_stage_top_capture_delta']:.6f}.",
            f"- Rolling objective vs holdout consistency: best CV single-stage mean RMSLE={single_best_cv.get('mean_rmsle', np.nan):.6f}; final holdout RMSLE={metrics.loc[metrics['model']=='optuna_single_stage_rmsle','rmsle'].iloc[0]:.6f}. Best CV two-stage mean capture={two_best_cv.get('mean_top_10pct_revenue_capture', np.nan):.2%}; final holdout capture={metrics.loc[metrics['model']=='optuna_two_stage_top_capture','top_10pct_revenue_capture'].iloc[0]:.2%}.",
            "- RMSLE and top-capture objectives select different model families by design: single-stage for RMSLE, two-stage for business ranking.",
            "- If improvements are small, the bottleneck is more likely feature/model structure and target formulation than basic hyperparameter settings.",
            "",
            "## Best Params",
            "",
            "```json",
            json.dumps(payload["best_params"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    bucket_features = build_time_bucket_features_from_zip(Path(args.zip_path), "train.csv")
    trials, metrics, predictions, payload = run_tuning(
        train_model_input,
        bucket_features,
        n_trials=args.n_trials,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )
    trials.to_csv(processed_dir / "optuna_trials.csv", index=False)
    metrics.to_csv(processed_dir / "optuna_best_metrics.csv", index=False)
    predictions.to_parquet(processed_dir / "optuna_valid_predictions.parquet", index=False)
    (processed_dir / "optuna_best_params.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(reports_dir / "optuna_tuning_results.md", trials, metrics, payload, processed_dir)


if __name__ == "__main__":
    main()
