from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from experiments.run_feature_ablation import (
    build_feature_lists,
    build_preprocessor,
    build_time_bucket_features_from_zip,
    get_feature_set_frame,
    restore_ltv_scale,
)
from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN
from experiments.train_two_stage_model import build_final_prediction
from experiments.tune_xgboost_optuna import TARGET_FEATURE_SET, build_param_dict_from_best


FINAL_MODEL = "optuna_two_stage_top_capture"
PREDICTION_COL = "ltv_d8_d180"
EXPECTED_FALLBACK_COUNT = 0


def load_final_params_payload(processed_dir: Path) -> dict[str, Any]:
    final_path = processed_dir / "final_model_params.json"
    legacy_path = processed_dir / "optuna_best_params.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    return json.loads(legacy_path.read_text(encoding="utf-8"))


def load_sample_submission(path: Path | None, zip_path: Path) -> pd.DataFrame:
    if path is not None and path.exists():
        return pd.read_csv(path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("sample_submission.csv") as handle:
            return pd.read_csv(handle)


def load_expected_test_user_ids(zip_path: Path) -> pd.Series:
    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("test.csv") as handle:
            for chunk in pd.read_csv(handle, usecols=["user_id"], chunksize=500_000):
                chunks.append(chunk["user_id"])
    ids = pd.concat(chunks, ignore_index=True).drop_duplicates(keep="first").reset_index(drop=True)
    ids.name = "user_id"
    return ids


def _build_classifier(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,
        verbosity=0,
    )


def _build_regressor(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=0,
        verbosity=0,
    )


def prepare_full_train_test_features(
    train_model_input: pd.DataFrame,
    test_model_input: pd.DataFrame,
    train_bucket_features: pd.DataFrame,
    test_bucket_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Any]:
    bucket_features = pd.concat([train_bucket_features, test_bucket_features], ignore_index=True)
    bucket_value_cols = [c for c in bucket_features.columns if c not in GRAIN_COLUMNS]
    bucket_features = bucket_features.groupby(GRAIN_COLUMNS, dropna=False, sort=False)[bucket_value_cols].sum().reset_index()
    train_features, test_features, _ = get_feature_set_frame(
        TARGET_FEATURE_SET,
        train_model_input.copy(),
        test_model_input.copy(),
        bucket_features,
    )
    numeric_features, categorical_features = build_feature_lists(train_features)
    feature_columns = numeric_features + categorical_features
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(train_features[feature_columns])
    return train_features, test_features, feature_columns, preprocessor


def fit_full_two_stage_model(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: list[str],
    preprocessor: Any,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, XGBClassifier, XGBRegressor, Any]:
    x_train = preprocessor.transform(train_features[feature_columns])
    x_test = preprocessor.transform(test_features[feature_columns])
    y_raw = train_features[TARGET_COLUMN].clip(lower=0).astype(float)
    y_positive = (y_raw > 0).astype(int)
    y_log = np.log1p(y_raw)

    classifier = _build_classifier(params)
    classifier.fit(x_train, y_positive, verbose=False)
    p_positive = pd.Series(classifier.predict_proba(x_test)[:, 1], index=test_features.index, name="p_positive")

    positive_mask = y_raw > 0
    if not positive_mask.any():
        raise ValueError("Full train data has no positive target rows for stage 2.")
    regressor = _build_regressor(params)
    regressor.fit(x_train[positive_mask.to_numpy()], y_log[positive_mask], verbose=False)
    positive_pred = pd.Series(restore_ltv_scale(regressor.predict(x_test)), index=test_features.index, name="positive_ltv_pred")
    final_pred = build_final_prediction(p_positive, positive_pred)

    prediction_frame = test_features[GRAIN_COLUMNS].copy()
    prediction_frame["p_positive"] = p_positive.to_numpy()
    prediction_frame["positive_ltv_pred"] = positive_pred.to_numpy()
    prediction_frame["pred_ltv"] = final_pred.to_numpy()
    prediction_frame["pred_ltv"] = prediction_frame["pred_ltv"].astype(float).clip(lower=0)
    return prediction_frame, classifier, regressor, preprocessor


def aggregate_test_predictions(row_predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        row_predictions.groupby("user_id", as_index=False, dropna=False, sort=False)
        .agg(
            pred_ltv=("pred_ltv", "mean"),
            p_positive=("p_positive", "mean"),
            positive_ltv_pred=("positive_ltv_pred", "mean"),
            model_row_count=("pred_ltv", "size"),
            platform=("platform", "first"),
            country_tier=("country_tier", "first"),
            channel_tier=("channel_tier", "first"),
        )
        .reset_index(drop=True)
    )


def build_fallback_predictions(missing_rows: pd.DataFrame, train_model_input: pd.DataFrame) -> pd.DataFrame:
    if missing_rows.empty:
        return pd.DataFrame(columns=list(missing_rows.columns) + [PREDICTION_COL, "fallback_method"])

    train = train_model_input.copy()
    target = train[TARGET_COLUMN].clip(lower=0).astype(float)
    global_mean = float(target.mean())
    platform_country_channel = train.assign(_target=target).groupby(
        ["platform", "country_tier", "channel_tier"], dropna=False
    )["_target"].mean()
    country_channel = train.assign(_target=target).groupby(["country_tier", "channel_tier"], dropna=False)["_target"].mean()

    rows: list[dict[str, Any]] = []
    for row in missing_rows.to_dict(orient="records"):
        key1 = (row.get("platform"), row.get("country_tier"), row.get("channel_tier"))
        key2 = (row.get("country_tier"), row.get("channel_tier"))
        if key1 in platform_country_channel.index:
            value = float(platform_country_channel.loc[key1])
            method = "platform_country_channel_mean"
        elif key2 in country_channel.index:
            value = float(country_channel.loc[key2])
            method = "country_channel_mean"
        else:
            value = global_mean
            method = "global_mean"
        out = dict(row)
        out[PREDICTION_COL] = max(0.0, value)
        out["fallback_method"] = method
        rows.append(out)
    return pd.DataFrame(rows)


def _as_expected_user_ids(expected_user_ids: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(expected_user_ids, pd.DataFrame):
        ids = expected_user_ids["user_id"].copy()
    else:
        ids = expected_user_ids.copy()
    ids.name = "user_id"
    return ids.reset_index(drop=True)


def build_submission_frame(
    expected_user_ids: pd.Series | pd.DataFrame,
    user_predictions: pd.DataFrame,
    test_model_input: pd.DataFrame,
    train_model_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_ids = _as_expected_user_ids(expected_user_ids)
    expected_frame = pd.DataFrame({"user_id": expected_ids})
    prediction_lookup = user_predictions[["user_id", "pred_ltv"]].rename(columns={"pred_ltv": PREDICTION_COL})
    submission = expected_frame.merge(prediction_lookup, on="user_id", how="left")

    missing_mask = submission[PREDICTION_COL].isna()
    missing_context = expected_frame.loc[missing_mask].merge(
        test_model_input[["user_id", "platform", "country_tier", "channel_tier"]].drop_duplicates("user_id"),
        on="user_id",
        how="left",
    )
    for col in ["platform", "country_tier", "channel_tier"]:
        if col not in missing_context.columns:
            missing_context[col] = np.nan
    fallback_rows = build_fallback_predictions(missing_context, train_model_input)
    if not fallback_rows.empty:
        fallback_values = fallback_rows.set_index("user_id")[PREDICTION_COL]
        submission.loc[missing_mask, PREDICTION_COL] = submission.loc[missing_mask, "user_id"].map(fallback_values).to_numpy()

    submission[PREDICTION_COL] = submission[PREDICTION_COL].astype(float).clip(lower=0)
    return submission[["user_id", PREDICTION_COL]], fallback_rows.sort_values("user_id").reset_index(drop=True)


def validate_submission_outputs(
    expected_user_ids: pd.Series | pd.DataFrame,
    submission: pd.DataFrame,
    fallback_rows: pd.DataFrame,
    expected_fallback_count: int,
    artifacts: dict[str, bool],
) -> dict[str, bool]:
    expected_ids = _as_expected_user_ids(expected_user_ids)
    values = submission[PREDICTION_COL].to_numpy(dtype=float)
    return {
        "row_count_matches_expected_users": len(submission) == len(expected_ids),
        "id_order_matches_expected_users": submission["user_id"].tolist() == expected_ids.tolist(),
        "prediction_not_null": not submission[PREDICTION_COL].isna().any(),
        "prediction_not_inf": bool(np.isfinite(values).all()),
        "prediction_non_negative": bool((values >= 0).all()),
        "fallback_count_matches_expected": len(fallback_rows) == expected_fallback_count,
        "model_artifacts_exist": bool(all(artifacts.values())),
    }


def _summary_stats(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def _fallback_method_summary(fallback_rows: pd.DataFrame) -> pd.DataFrame:
    if fallback_rows.empty:
        return pd.DataFrame(columns=["fallback_method", "row_count"])
    return fallback_rows["fallback_method"].value_counts().rename_axis("fallback_method").reset_index(name="row_count")


def write_prediction_report(
    path: Path,
    checks: dict[str, bool],
    train_rows: int,
    test_feature_rows: int,
    test_unique_users: int,
    sample_rows: int,
    matched_rows: int,
    fallback_rows: pd.DataFrame,
    user_predictions: pd.DataFrame,
) -> None:
    prediction_summary = _summary_stats(user_predictions["pred_ltv"])
    p_positive_summary = {
        "mean": float(user_predictions["p_positive"].mean()),
        "median": float(user_predictions["p_positive"].median()),
        "p90": float(user_predictions["p_positive"].quantile(0.90)),
        "p99": float(user_predictions["p_positive"].quantile(0.99)),
    }
    fallback_summary = _fallback_method_summary(fallback_rows)

    lines = [
        "# Test Prediction Report",
        "",
        "## Scope",
        "",
        f"- Final model: `{FINAL_MODEL}` refit on all labeled train_model_input rows.",
        "- Submission grain: one row per unique `user_id` in `test.csv`.",
        "- Test labels are unavailable, so test MAE/RMSE/RMSLE are not calculated.",
        "- Kaggle hidden test score is available only after submitting `submission.csv` to Kaggle.",
        "",
        "## Counts",
        "",
        f"- Train row count: {train_rows:,}",
        f"- Test feature row count: {test_feature_rows:,}",
        f"- Test unique user count / submission row count: {test_unique_users:,}",
        f"- sample_submission row count in provided zip: {sample_rows:,}",
        f"- Matched prediction row count: {matched_rows:,}",
        f"- Fallback prediction row count: {len(fallback_rows):,}",
        "",
        "## Fallback Method Summary",
        "",
        "| fallback method | row count |",
        "|---|---:|",
    ]
    if fallback_summary.empty:
        lines.append("| none | 0 |")
    else:
        for row in fallback_summary.to_dict(orient="records"):
            lines.append(f"| {row['fallback_method']} | {int(row['row_count'])} |")

    lines.extend(["", "## Prediction Checks", "", "| check | passed |", "|---|---|"])
    for key, value in checks.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Test Prediction Summary", "", "| metric | value |", "|---|---:|"])
    for key, value in prediction_summary.items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## p_positive Summary", "", "| metric | value |", "|---|---:|"])
    for key, value in p_positive_summary.items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Submission Notes",
            "",
            "- Kaggle rejected the provided sample_submission row count; the accepted contract is the competition description: one row per unique test user_id.",
            "- Context-level predictions are averaged to user-level LTV before writing `submission.csv`.",
            "- Test targets are not available in the public files; hidden test score can only be checked after Kaggle submission.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_submission(project_root: Path, zip_path: Path, sample_submission_path: Path | None = None) -> dict[str, bool]:
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    test_model_input = pd.read_parquet(processed_dir / "test_model_input.parquet")
    params_payload = load_final_params_payload(processed_dir)
    params = build_param_dict_from_best(params_payload["best_params"][FINAL_MODEL])
    train_bucket = build_time_bucket_features_from_zip(zip_path, "train.csv")
    test_bucket = build_time_bucket_features_from_zip(zip_path, "test.csv")
    sample_submission = load_sample_submission(sample_submission_path, zip_path)
    expected_user_ids = load_expected_test_user_ids(zip_path)

    train_features, test_features, feature_columns, preprocessor = prepare_full_train_test_features(
        train_model_input,
        test_model_input,
        train_bucket,
        test_bucket,
    )
    row_predictions, classifier, regressor, preprocessor = fit_full_two_stage_model(
        train_features,
        test_features,
        feature_columns,
        preprocessor,
        params,
    )
    user_predictions = aggregate_test_predictions(row_predictions)
    submission, fallback_rows = build_submission_frame(expected_user_ids, user_predictions, test_model_input, train_model_input)

    stage1_path = models_dir / "final_two_stage_stage1.joblib"
    stage2_path = models_dir / "final_two_stage_stage2.joblib"
    preprocessor_path = models_dir / "final_preprocessor.joblib"
    joblib.dump(classifier, stage1_path)
    joblib.dump(regressor, stage2_path)
    joblib.dump(preprocessor, preprocessor_path)

    artifacts = {
        "stage1": stage1_path.exists() and stage1_path.stat().st_size > 0,
        "stage2": stage2_path.exists() and stage2_path.stat().st_size > 0,
        "preprocessor": preprocessor_path.exists() and preprocessor_path.stat().st_size > 0,
    }
    checks = validate_submission_outputs(expected_user_ids, submission, fallback_rows, EXPECTED_FALLBACK_COUNT, artifacts)

    user_predictions.to_parquet(processed_dir / "test_predictions.parquet", index=False)
    user_predictions.to_csv(processed_dir / "test_predictions.csv", index=False)
    submission.to_csv(processed_dir / "submission.csv", index=False)
    fallback_rows.to_csv(processed_dir / "submission_fallback_rows.csv", index=False)
    write_prediction_report(
        reports_dir / "test_prediction_report.md",
        checks,
        train_rows=len(train_model_input),
        test_feature_rows=len(test_model_input),
        test_unique_users=len(expected_user_ids),
        sample_rows=len(sample_submission),
        matched_rows=len(submission) - len(fallback_rows),
        fallback_rows=fallback_rows,
        user_predictions=user_predictions,
    )
    return checks



def write_final_prediction_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Final Prediction Report",
        "",
        "## Scope",
        "",
        f"- Final model: `{FINAL_MODEL}` refit on all labeled train rows.",
        "- Output grain: both user-context predictions and user-level aggregated predictions.",
        "- Test labels are unavailable, so test MAE/RMSE/RMSLE are not calculated.",
        "",
        "## Outputs",
        "",
        "- `data/processed/final_test_context_predictions.parquet`",
        "- `data/processed/final_test_user_predictions.parquet`",
        "- `models/final_two_stage_stage1.joblib`",
        "- `models/final_two_stage_stage2.joblib`",
        "- `models/final_preprocessor.joblib`",
        "",
        "## Prediction Checks",
        "",
        f"- Context prediction rows: {summary['context_prediction_rows']:,}",
        f"- User prediction rows: {summary['user_prediction_rows']:,}",
        f"- Test metrics: `{summary['test_metrics']}`",
        f"- Null predictions: {summary['prediction_null_count']:,}",
        f"- Inf predictions: {summary['prediction_inf_count']:,}",
        f"- Negative predictions: {summary['prediction_negative_count']:,}",
        "",
        "## Prediction Distribution",
        "",
        f"- Mean: {summary['prediction_mean']:.6f}",
        f"- Median: {summary['prediction_median']:.6f}",
        f"- P75: {summary['prediction_p75']:.6f}",
        f"- P90: {summary['prediction_p90']:.6f}",
        f"- P95: {summary['prediction_p95']:.6f}",
        f"- P99: {summary['prediction_p99']:.6f}",
        f"- Max: {summary['prediction_max']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def predict_submission(project_root: Path, zip_path: Path) -> dict[str, object]:
    """Create final portfolio prediction artifacts without calculating hidden-test metrics."""
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    test_model_input = pd.read_parquet(processed_dir / "test_model_input.parquet")
    metadata = json.loads((models_dir / "final_model_metadata.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    preprocessor = joblib.load(models_dir / "final_preprocessor.joblib")
    classifier = joblib.load(models_dir / "final_two_stage_stage1.joblib")
    regressor = joblib.load(models_dir / "final_two_stage_stage2.joblib")

    train_bucket = build_time_bucket_features_from_zip(zip_path, "train.csv")
    test_bucket = build_time_bucket_features_from_zip(zip_path, "test.csv")
    _, test_features, _, _ = prepare_full_train_test_features(train_model_input, test_model_input, train_bucket, test_bucket)
    x_test = preprocessor.transform(test_features[feature_columns])

    p_positive = pd.Series(classifier.predict_proba(x_test)[:, 1], index=test_features.index, name="p_positive")
    positive_ltv_pred = pd.Series(restore_ltv_scale(regressor.predict(x_test)), index=test_features.index, name="positive_ltv_pred")
    final_pred = build_final_prediction(p_positive, positive_ltv_pred)

    context_predictions = test_features[GRAIN_COLUMNS].copy()
    context_predictions["p_positive"] = p_positive.to_numpy()
    context_predictions["positive_ltv_pred"] = positive_ltv_pred.to_numpy()
    context_predictions["pred_ltv"] = final_pred.to_numpy()
    context_predictions["pred_ltv"] = context_predictions["pred_ltv"].astype(float).clip(lower=0)
    user_predictions = aggregate_test_predictions(context_predictions)

    context_predictions.to_parquet(processed_dir / "final_test_context_predictions.parquet", index=False)
    context_predictions.to_csv(processed_dir / "final_test_context_predictions.csv", index=False)
    user_predictions.to_parquet(processed_dir / "final_test_user_predictions.parquet", index=False)
    user_predictions.to_csv(processed_dir / "final_test_user_predictions.csv", index=False)

    pred = user_predictions["pred_ltv"].astype(float)
    values = pred.to_numpy(dtype=float)
    summary = {
        "context_prediction_rows": int(len(context_predictions)),
        "user_prediction_rows": int(len(user_predictions)),
        "test_metrics": "not_calculated_no_test_target",
        "prediction_null_count": int(pred.isna().sum()),
        "prediction_inf_count": int(np.isinf(values).sum()),
        "prediction_negative_count": int((pred < 0).sum()),
        "prediction_mean": float(pred.mean()),
        "prediction_median": float(pred.median()),
        "prediction_p75": float(pred.quantile(0.75)),
        "prediction_p90": float(pred.quantile(0.90)),
        "prediction_p95": float(pred.quantile(0.95)),
        "prediction_p99": float(pred.quantile(0.99)),
        "prediction_max": float(pred.max()),
    }
    (processed_dir / "final_prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_final_prediction_report(reports_dir / "final_prediction_report.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    parser.add_argument("--sample-submission-path", default=None)
    args = parser.parse_args()

    predict_submission(Path(args.project_root), Path(args.zip_path))


if __name__ == "__main__":
    main()

