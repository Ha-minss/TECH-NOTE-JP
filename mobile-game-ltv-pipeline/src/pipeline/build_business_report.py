from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.run_feature_ablation import build_time_bucket_features_from_zip
from experiments.train_baseline import GRAIN_COLUMNS, TARGET_COLUMN
from experiments.tune_xgboost_optuna import FINAL_HOLDOUT, build_param_dict_from_best, fit_final_two_stage, prepare_fold_data


FINAL_MODEL = "optuna_two_stage_top_capture"
PRED_COL = f"pred_{FINAL_MODEL}"
P_POSITIVE_COL = f"pred_{FINAL_MODEL}_p_positive"
POSITIVE_LTV_COL = f"pred_{FINAL_MODEL}_positive_ltv_pred"
ALLOWED_UA_DECISIONS = {"scale_up", "keep", "reduce"}
MIN_UA_SEGMENT_USERS = 100
FINAL_STAGE1_ARTIFACT = "final_optuna_two_stage_stage1_classifier.pkl"
FINAL_STAGE2_ARTIFACT = "final_optuna_two_stage_stage2_regressor.pkl"
IMPORTANCE_SOURCE = "final_tuned_optuna_two_stage_model_artifact_gain_importance"

SEGMENT_GROUPS = [
    ("platform", ["platform"]),
    ("country_tier", ["country_tier"]),
    ("channel_tier", ["channel_tier"]),
    ("install_week", ["install_week"]),
    ("platform_country_tier", ["platform", "country_tier"]),
    ("country_tier_channel_tier", ["country_tier", "channel_tier"]),
    ("platform_country_tier_channel_tier", ["platform", "country_tier", "channel_tier"]),
]

TOP_DECILE_NUMERIC_FEATURES = [
    "revenue_d0_d7",
    "ad_revenue_d0_d7",
    "iap_revenue_d0_d7",
    "active_days",
    "last_event_day",
    "event_count",
    "session_count",
    "ad_impression_count",
    "iap_count",
    "ads_per_session",
    "revenue_per_active_day",
    "early_payer_flag",
]
TOP_DECILE_CATEGORICAL_FEATURES = ["platform", "country_tier", "channel_tier", "top_network", "top_ad_placement"]


def ensure_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pred_ltv" not in out.columns and PRED_COL in out.columns:
        out["pred_ltv"] = out[PRED_COL].astype(float).clip(lower=0)
    if "actual_ltv" not in out.columns and TARGET_COLUMN in out.columns:
        out["actual_ltv"] = out[TARGET_COLUMN].astype(float).clip(lower=0)
    if "prediction_error" not in out.columns and {"pred_ltv", "actual_ltv"}.issubset(out.columns):
        out["prediction_error"] = out["pred_ltv"] - out["actual_ltv"]
    if "abs_error" not in out.columns and "prediction_error" in out.columns:
        out["abs_error"] = out["prediction_error"].abs()
    if "revenue_per_active_day" not in out.columns and {"revenue_d0_d7", "active_days"}.issubset(out.columns):
        active_days = out["active_days"].astype(float)
        out["revenue_per_active_day"] = np.where(
            active_days > 0,
            out["revenue_d0_d7"].astype(float) / active_days,
            0.0,
        )
    return out


def load_final_analysis_frame(processed_dir: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(processed_dir / "final_holdout_predictions.parquet")
    model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    valid_features = model_input[model_input["install_day"].between(24, 30)].copy()
    feature_cols = [c for c in valid_features.columns if c != TARGET_COLUMN]
    keep_prediction_cols = GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7", PRED_COL, P_POSITIVE_COL, POSITIVE_LTV_COL]
    base = predictions[keep_prediction_cols].merge(
        valid_features[feature_cols],
        on=GRAIN_COLUMNS,
        how="left",
        suffixes=("", "_feature"),
    )
    return add_top_decile_flag(ensure_analysis_columns(base), "pred_ltv")


def add_top_decile_flag(frame: pd.DataFrame, pred_col: str = "pred_ltv") -> pd.DataFrame:
    out = frame.copy()
    top_n = max(1, int(np.ceil(len(out) * 0.10))) if len(out) else 0
    out["is_pred_top_decile"] = False
    if top_n:
        top_index = out[pred_col].sort_values(ascending=False, kind="mergesort").head(top_n).index
        out.loc[top_index, "is_pred_top_decile"] = True
    return out


def build_segment_ltv_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_analysis_columns(frame)
    rows: list[dict[str, Any]] = []
    for segment_name, columns in SEGMENT_GROUPS:
        grouped = frame.groupby(columns, dropna=False)
        for keys, group in grouped:
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            row = {
                "segment_type": segment_name,
                "segment_key": " | ".join(str(v) for v in keys_tuple),
                "user_count": int(len(group)),
                "actual_ltv_mean": float(group["actual_ltv"].mean()),
                "predicted_ltv_mean": float(group["pred_ltv"].mean()),
                "actual_ltv_sum": float(group["actual_ltv"].sum()),
                "predicted_ltv_sum": float(group["pred_ltv"].sum()),
                "early_revenue_mean": float(group["revenue_d0_d7"].mean()),
                "positive_ltv_rate": float((group["actual_ltv"] > 0).mean()),
                "prediction_error_mean": float(group["prediction_error"].mean()),
                "abs_error_mean": float(group["abs_error"].mean()),
                "top_decile_user_share": float(group["is_pred_top_decile"].mean()),
            }
            for col, value in zip(columns, keys_tuple):
                row[col] = value
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["segment_type", "predicted_ltv_sum"], ascending=[True, False]).reset_index(drop=True)


def build_top_decile_numeric_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_analysis_columns(frame)
    rows = []
    top = frame[frame["is_pred_top_decile"]]
    rest = frame[~frame["is_pred_top_decile"]]
    for feature in TOP_DECILE_NUMERIC_FEATURES:
        if feature not in frame.columns:
            continue
        top_mean = float(top[feature].mean()) if len(top) else 0.0
        rest_mean = float(rest[feature].mean()) if len(rest) else 0.0
        rows.append(
            {
                "feature": feature,
                "top_decile_mean": top_mean,
                "non_top_decile_mean": rest_mean,
                "difference": top_mean - rest_mean,
                "ratio_top_vs_non_top": top_mean / rest_mean if rest_mean else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_top_decile_categorical_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in TOP_DECILE_CATEGORICAL_FEATURES:
        if feature not in frame.columns:
            continue
        for group_name, mask in [("top_decile", frame["is_pred_top_decile"]), ("non_top_decile", ~frame["is_pred_top_decile"])]:
            subset = frame[mask]
            values = subset[feature].fillna("missing")
            dist = values.value_counts(normalize=True, dropna=False)
            counts = values.value_counts(dropna=False)
            for category, share in dist.items():
                rows.append(
                    {
                        "feature": feature,
                        "group": group_name,
                        "category": category,
                        "user_count": int(counts[category]),
                        "share": float(share),
                    }
                )
    return pd.DataFrame(rows)


def build_synthetic_cpi_table(frame: pd.DataFrame) -> pd.DataFrame:
    country_base = {
        "US": 2.20,
        "KR": 1.55,
        "JP": 1.90,
        "DE": 1.65,
        "GB": 1.75,
        "FR": 1.55,
        "CA": 1.80,
        "AU": 1.85,
        "BR": 0.75,
        "MX": 0.65,
        "IN": 0.35,
        "OTHER": 0.85,
    }
    channel_multiplier = {
        "organic": 0.20,
        "paid": 1.00,
        "social": 0.90,
        "search": 1.20,
        "video": 1.10,
        "other": 0.80,
    }
    segments = frame[["country_tier", "channel_tier"]].drop_duplicates().copy()
    rows = []
    for row in segments.itertuples(index=False):
        country = str(row.country_tier)
        channel = str(row.channel_tier)
        base = country_base.get(country, country_base["OTHER"])
        multiplier = channel_multiplier.get(channel, 0.95)
        rows.append(
            {
                "country_tier": row.country_tier,
                "channel_tier": row.channel_tier,
                "synthetic_cpi": round(base * multiplier, 4),
                "cpi_source": "synthetic_deterministic_rule_for_decision_simulation",
            }
        )
    return pd.DataFrame(rows).sort_values(["country_tier", "channel_tier"]).reset_index(drop=True)


def _ua_decision(roas: float) -> str:
    if roas >= 1.5:
        return "scale_up"
    if roas >= 1.0:
        return "keep"
    return "reduce"


def build_ua_decision_simulation(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_analysis_columns(frame)
    cpi = build_synthetic_cpi_table(frame)
    segment = (
        frame.groupby(["country_tier", "channel_tier"], dropna=False)
        .agg(
            user_count=("user_id", "count"),
            predicted_ltv_mean=("pred_ltv", "mean"),
            actual_ltv_mean=("actual_ltv", "mean"),
        )
        .reset_index()
    )
    out = segment.merge(cpi, on=["country_tier", "channel_tier"], how="left")
    out["predicted_roas"] = out["predicted_ltv_mean"] / out["synthetic_cpi"]
    out["actual_roas"] = out["actual_ltv_mean"] / out["synthetic_cpi"]
    out["predicted_roas"] = out["predicted_roas"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    out["actual_roas"] = out["actual_roas"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    out["decision"] = np.where(
        out["user_count"] < MIN_UA_SEGMENT_USERS,
        "insufficient_sample",
        out["predicted_roas"].map(_ua_decision),
    )
    return out.sort_values(["user_count", "predicted_roas"], ascending=[False, False]).reset_index(drop=True)


def build_low_sample_segments(ua: pd.DataFrame) -> pd.DataFrame:
    low_sample = ua[ua["user_count"] < MIN_UA_SEGMENT_USERS].copy()
    low_sample["warning"] = "low_sample_warning"
    return low_sample.sort_values("predicted_roas", ascending=False).reset_index(drop=True)


def _feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            feature_names.extend([str(col) for col in columns])
        elif name == "cat":
            feature_names.extend(transformer.get_feature_names_out(columns).astype(str).tolist())
    return feature_names


def extract_gain_importance(pipeline: Any, stage: str) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = _feature_names_from_preprocessor(preprocessor)
    booster_scores = model.get_booster().get_score(importance_type="gain")
    rows = []
    for key, value in booster_scores.items():
        feature_index = int(key[1:]) if key.startswith("f") and key[1:].isdigit() else None
        feature = feature_names[feature_index] if feature_index is not None and feature_index < len(feature_names) else key
        rows.append(
            {
                "model": FINAL_MODEL,
                "stage": stage,
                "feature": feature,
                "importance_gain": float(value),
                "importance_type": "gain",
                "importance_source": IMPORTANCE_SOURCE,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["model", "stage", "feature", "importance_gain", "importance_type", "importance_source", "rank"])
    out = out.sort_values("importance_gain", ascending=False).head(30).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["model", "stage", "rank", "feature", "importance_gain", "importance_type", "importance_source"]]


def rebuild_final_model_artifacts_and_importance(
    processed_dir: Path,
    zip_path: Path,
    early_stopping_rounds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    params_payload = json.loads((processed_dir / "final_model_params.json").read_text(encoding="utf-8"))
    params = build_param_dict_from_best(params_payload["best_params"][FINAL_MODEL])
    bucket_features = build_time_bucket_features_from_zip(zip_path, "train.csv")
    final_fold = prepare_fold_data(train_model_input, bucket_features, [FINAL_HOLDOUT])[0]
    _, _, _, stage1_pipeline, stage2_pipeline = fit_final_two_stage(final_fold, params, early_stopping_rounds)

    with (processed_dir / FINAL_STAGE1_ARTIFACT).open("wb") as f:
        pickle.dump(stage1_pipeline, f)
    with (processed_dir / FINAL_STAGE2_ARTIFACT).open("wb") as f:
        pickle.dump(stage2_pipeline, f)

    stage1 = extract_gain_importance(stage1_pipeline, "stage1_classifier")
    stage2 = extract_gain_importance(stage2_pipeline, "stage2_regressor")
    stage1.to_csv(processed_dir / "final_feature_importance_stage1.csv", index=False)
    stage2.to_csv(processed_dir / "final_feature_importance_stage2.csv", index=False)
    return stage1, stage2, IMPORTANCE_SOURCE


def validate_final_artifacts(processed_dir: Path) -> dict[str, bool]:
    stage1_path = processed_dir / FINAL_STAGE1_ARTIFACT
    stage2_path = processed_dir / FINAL_STAGE2_ARTIFACT
    stage1_importance_path = processed_dir / "final_feature_importance_stage1.csv"
    stage2_importance_path = processed_dir / "final_feature_importance_stage2.csv"
    stage1_importance = pd.read_csv(stage1_importance_path) if stage1_importance_path.exists() else pd.DataFrame()
    stage2_importance = pd.read_csv(stage2_importance_path) if stage2_importance_path.exists() else pd.DataFrame()
    return {
        "final_stage1_artifact_exists": stage1_path.exists() and stage1_path.stat().st_size > 0,
        "final_stage2_artifact_exists": stage2_path.exists() and stage2_path.stat().st_size > 0,
        "stage1_importance_non_empty": len(stage1_importance) > 0,
        "stage2_importance_non_empty": len(stage2_importance) > 0,
    }


def validate_business_outputs(frame: pd.DataFrame, segment_summary: pd.DataFrame, ua: pd.DataFrame) -> dict[str, bool]:
    expected_top_n = max(1, int(np.ceil(len(frame) * 0.10))) if len(frame) else 0
    roas_values = ua["predicted_roas"].to_numpy(dtype=float) if len(ua) else np.asarray([])
    low_sample = ua[ua["user_count"] < MIN_UA_SEGMENT_USERS]
    high_sample = ua[ua["user_count"] >= MIN_UA_SEGMENT_USERS]
    return {
        "top_decile_count_matches": int(frame["is_pred_top_decile"].sum()) == expected_top_n,
        "segment_summary_non_empty": len(segment_summary) > 0,
        "predicted_roas_valid": bool(len(ua) > 0 and not ua["predicted_roas"].isna().any() and not np.isinf(roas_values).any() and (roas_values >= 0).all()),
        "ua_decision_allowed_values": set(high_sample["decision"]).issubset(ALLOWED_UA_DECISIONS) and set(low_sample["decision"]).issubset({"insufficient_sample"}),
        "low_sample_segments_insufficient": bool(low_sample.empty or (low_sample["decision"] == "insufficient_sample").all()),
        "high_sample_segments_have_business_decisions": bool(high_sample.empty or set(high_sample["decision"]).issubset(ALLOWED_UA_DECISIONS)),
        "synthetic_cpi_positive": bool(len(ua) > 0 and (ua["synthetic_cpi"] > 0).all()),
    }


def write_model_card(path: Path, metrics: pd.DataFrame, params_payload: dict[str, Any], importance_note: str) -> None:
    row = metrics[metrics["model"] == FINAL_MODEL].iloc[0]
    lines = [
        "# Final Model Card",
        "",
        "## Selected Model",
        "",
        f"- Model: `{FINAL_MODEL}`",
        "- Architecture: two-stage XGBoost. Stage 1 estimates positive LTV probability; Stage 2 estimates conditional positive LTV; final prediction is `p_positive * predicted_ltv_if_positive`.",
        "- Validation split: train install_day 0-23, valid install_day 24-30.",
        "- Selection reason: best overall final holdout result after rolling validation and Optuna tuning, with stronger RMSLE, MAE/RMSE, and top-decile revenue capture than previous candidates.",
        "- Persisted artifacts: `data/processed/final_optuna_two_stage_stage1_classifier.pkl`, `data/processed/final_optuna_two_stage_stage2_regressor.pkl`.",
        "",
        "## Holdout Metrics",
        "",
        f"- MAE: {row['mae']:.4f}",
        f"- RMSE: {row['rmse']:.4f}",
        f"- RMSLE: {row['rmsle']:.4f}",
        f"- Spearman correlation: {row['spearman_corr']:.4f}",
        f"- Top 10% revenue capture: {row['top_10pct_revenue_capture']:.2%}",
        f"- Positive LTV rate in predicted top decile: {row['positive_ltv_rate_pred_top_decile']:.2%}",
        "",
        "The positive-rate metric is not an accuracy score. It means the share of users inside the model's predicted top decile whose actual D8-D180 LTV is greater than zero.",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(params_payload["best_params"].get(FINAL_MODEL, {}), indent=2),
        "```",
        "",
        "## Feature Importance Note",
        "",
        f"Feature importance is extracted from the saved final tuned model artifacts using XGBoost gain importance: `{importance_note}`.",
        "This step does not rerun Optuna or reselect models; it reconstructs the already selected final model artifact from saved best parameters for persistence and interpretation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_business_report(
    path: Path,
    frame: pd.DataFrame,
    numeric_top: pd.DataFrame,
    categorical_top: pd.DataFrame,
    ua: pd.DataFrame,
    low_sample: pd.DataFrame,
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    checks: dict[str, bool],
    importance_note: str,
) -> None:
    top_count = int(frame["is_pred_top_decile"].sum())
    top_share = top_count / len(frame) if len(frame) else 0
    total_actual = frame["actual_ltv"].sum()
    top_actual_capture = frame.loc[frame["is_pred_top_decile"], "actual_ltv"].sum() / total_actual if total_actual else 0.0
    top_numeric = numeric_top.sort_values("difference", ascending=False).head(8)
    ua_main = ua[ua["user_count"] >= MIN_UA_SEGMENT_USERS].sort_values("predicted_roas", ascending=False).head(10)
    low_sample_top = low_sample.sort_values("predicted_roas", ascending=False).head(10)
    stage1_top = stage1.head(10)
    stage2_top = stage2.head(10)

    lines = [
        "# Business Analysis",
        "",
        "## Executive Summary",
        "",
        f"- Final model: `{FINAL_MODEL}`.",
        f"- Predicted top decile size: {top_count:,} users ({top_share:.2%} of validation rows).",
        f"- Actual D8-D180 revenue captured by predicted top decile: {top_actual_capture:.2%}.",
        f"- UA decision simulation uses `min_users={MIN_UA_SEGMENT_USERS}`; smaller segments are marked `insufficient_sample`.",
        "- Synthetic CPI in the UA simulation is not real ad spend. It is a deterministic example table for decision workflow demonstration.",
        "",
        "## Feature Importance Interpretation",
        "",
        f"- Importance source: `{importance_note}`.",
        "- Stage 1 classifier importance is extracted from the final tuned classifier artifact and should be read as signals for whether a user becomes positive-LTV.",
        "- Stage 2 regressor importance is extracted from the final tuned regressor artifact and should be read as signals for the expected amount among users modeled as future positive-LTV.",
        "",
        "### Stage 1 Top Features",
        "",
        "| rank | feature | gain importance |",
        "|---:|---|---:|",
    ]
    for row in stage1_top.to_dict(orient="records"):
        lines.append(f"| {int(row['rank'])} | {row['feature']} | {row['importance_gain']:.6f} |")

    lines.extend(["", "### Stage 2 Top Features", "", "| rank | feature | gain importance |", "|---:|---|---:|"])
    for row in stage2_top.to_dict(orient="records"):
        lines.append(f"| {int(row['rank'])} | {row['feature']} | {row['importance_gain']:.6f} |")

    lines.extend(
        [
            "",
            "## Predicted Top-Decile Behavior",
            "",
            "| feature | top decile mean | non-top mean | difference | ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_numeric.to_dict(orient="records"):
        ratio = row["ratio_top_vs_non_top"]
        ratio_text = "" if pd.isna(ratio) else f"{ratio:.2f}"
        lines.append(
            f"| {row['feature']} | {row['top_decile_mean']:.4f} | {row['non_top_decile_mean']:.4f} | {row['difference']:.4f} | {ratio_text} |"
        )

    lines.extend(
        [
            "",
            "Top-decile users are the users ranked highest by predicted LTV, not users known to be high-value in advance. Their profile is therefore useful for UA targeting hypotheses and lifecycle prioritization.",
            "",
            "## UA Decision Simulation",
            "",
            f"Main table includes only segments with at least {MIN_UA_SEGMENT_USERS} users, sorted by predicted ROAS. Segments below the threshold are not assigned scale/keep/reduce decisions.",
            "",
            "| country_tier | channel_tier | users | predicted LTV | actual LTV | synthetic CPI | predicted ROAS | actual ROAS | decision |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ua_main.to_dict(orient="records"):
        lines.append(
            f"| {row['country_tier']} | {row['channel_tier']} | {int(row['user_count'])} | {row['predicted_ltv_mean']:.4f} | "
            f"{row['actual_ltv_mean']:.4f} | {row['synthetic_cpi']:.4f} | {row['predicted_roas']:.2f} | {row['actual_roas']:.2f} | {row['decision']} |"
        )

    lines.extend(
        [
            "",
            "### Low-Sample Warning Segments",
            "",
            "These segments may show high predicted ROAS, but they are marked `insufficient_sample` because their validation sample size is below the minimum threshold.",
            "",
            "| country_tier | channel_tier | users | predicted LTV | synthetic CPI | predicted ROAS | decision | warning |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in low_sample_top.to_dict(orient="records"):
        lines.append(
            f"| {row['country_tier']} | {row['channel_tier']} | {int(row['user_count'])} | {row['predicted_ltv_mean']:.4f} | "
            f"{row['synthetic_cpi']:.4f} | {row['predicted_roas']:.2f} | {row['decision']} | {row['warning']} |"
        )

    lines.extend(["", "## Validation Checks", "", "| check | passed |", "|---|---|"])
    for key, value in checks.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_business_analysis(project_root: Path, zip_path: Path, early_stopping_rounds: int = 50) -> dict[str, Any]:
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = load_final_analysis_frame(processed_dir)
    metrics = pd.read_csv(processed_dir / "final_model_metrics.csv")
    params_payload = json.loads((processed_dir / "final_model_params.json").read_text(encoding="utf-8"))

    stage1, stage2, importance_note = rebuild_final_model_artifacts_and_importance(processed_dir, zip_path, early_stopping_rounds)
    segment = build_segment_ltv_summary(frame)
    numeric_top = build_top_decile_numeric_analysis(frame)
    categorical_top = build_top_decile_categorical_distribution(frame)
    ua = build_ua_decision_simulation(frame)
    low_sample = build_low_sample_segments(ua)
    checks = {
        **validate_business_outputs(frame, segment, ua),
        **validate_final_artifacts(processed_dir),
    }

    segment.to_csv(processed_dir / "segment_ltv_summary.csv", index=False)
    numeric_top.to_csv(processed_dir / "top_decile_analysis.csv", index=False)
    categorical_top.to_csv(processed_dir / "top_decile_categorical_distribution.csv", index=False)
    ua.to_csv(processed_dir / "ua_decision_simulation.csv", index=False)
    low_sample.to_csv(processed_dir / "low_sample_segments.csv", index=False)
    write_model_card(reports_dir / "final_model_card.md", metrics, params_payload, importance_note)
    write_business_report(
        reports_dir / "business_analysis.md",
        frame,
        numeric_top,
        categorical_top,
        ua,
        low_sample,
        stage1,
        stage2,
        checks,
        importance_note,
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    args = parser.parse_args()
    run_business_analysis(Path(args.project_root), Path(args.zip_path), args.early_stopping_rounds)


if __name__ == "__main__":
    main()
