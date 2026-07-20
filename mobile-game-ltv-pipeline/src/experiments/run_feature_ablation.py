from __future__ import annotations

import argparse
import json
import zipfile
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


FEATURE_SETS = [
    "xgb_current_full",
    "xgb_time_bucket_features",
    "xgb_velocity_ratio_features",
    "xgb_frequency_interaction_features",
    "xgb_target_encoding_features",
    "xgb_all_derived_features",
]
BASE_CATEGORICAL_FEATURES = ["platform", "country_tier", "channel_tier", "top_network", "top_ad_placement"]
EXCLUDED_FEATURES = ["user_id", "install_day", TARGET_COLUMN]
BUCKET_LABELS = {
    "d0": lambda s: s == 0,
    "d1": lambda s: s == 1,
    "d2_d3": lambda s: s.between(2, 3),
    "d4_d7": lambda s: s.between(4, 7),
}
EVENT_TYPES = {
    "event_count": None,
    "session_count": "session",
    "ad_impression_count": "ad_impression",
    "iap_count": "iap",
}
INTERACTION_COLUMNS = [
    "platform_country",
    "country_channel",
    "platform_channel",
    "network_placement",
    "country_network",
    "channel_network",
]
TARGET_ENCODING_SPECS = [
    ("channel_tier", "te_channel"),
    ("country_channel", "te_country_channel"),
    ("platform_country_channel", "te_platform_country_channel"),
    ("top_network", "te_top_network"),
    ("top_ad_placement", "te_top_ad_placement"),
]
POSITIVE_RATE_SPECS = [
    ("channel_tier", "te_channel_positive_rate"),
    ("country_channel", "te_country_channel_positive_rate"),
    ("top_network", "te_top_network_positive_rate"),
]


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

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


@dataclass
class ExperimentResult:
    feature_set: str
    prediction: pd.Series
    metrics: dict[str, float]
    pipeline: Pipeline
    metadata: dict[str, Any]


def make_time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = frame[frame["install_day"].between(0, 23)].copy()
    valid_split = frame[frame["install_day"].between(24, 30)].copy()
    return train_split, valid_split


def _safe_divide(numerator, denominator) -> pd.Series:
    num = pd.Series(numerator, dtype=float)
    den = pd.Series(denominator, dtype=float).replace(0, np.nan)
    out = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(float)


def _interaction(*series: pd.Series) -> pd.Series:
    cleaned = [s.astype("string").fillna("missing") for s in series]
    out = cleaned[0]
    for s in cleaned[1:]:
        out = out + "__" + s
    return out.astype(str)


def build_time_bucket_features(raw_events: pd.DataFrame) -> pd.DataFrame:
    work = raw_events.copy()
    work["revenue_usd"] = work["revenue_usd"].fillna(0.0)
    grouped_keys = GRAIN_COLUMNS
    base = work[grouped_keys].drop_duplicates().copy()

    for prefix, event_type in EVENT_TYPES.items():
        type_mask = pd.Series(True, index=work.index) if event_type is None else work["event_type"].eq(event_type)
        for label, bucket_fn in BUCKET_LABELS.items():
            col = f"{prefix}_{label}"
            mask = type_mask & bucket_fn(work["day_since_install"])
            counts = work.loc[mask].groupby(grouped_keys, dropna=False).size().rename(col).reset_index()
            base = base.merge(counts, on=grouped_keys, how="left")
            base[col] = base[col].fillna(0).astype(int)

    return base


def build_time_bucket_features_from_zip(zip_path: Path, csv_name: str = "train.csv") -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as handle:
            for chunk in pd.read_csv(handle, chunksize=500_000, low_memory=False):
                chunks.append(build_time_bucket_features(chunk))
    combined = pd.concat(chunks, ignore_index=True)
    bucket_cols = [c for c in combined.columns if c not in GRAIN_COLUMNS]
    return combined.groupby(GRAIN_COLUMNS, dropna=False, sort=False)[bucket_cols].sum().reset_index()


def attach_time_bucket_features(frame: pd.DataFrame, bucket_features: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(bucket_features, on=GRAIN_COLUMNS, how="left")
    bucket_cols = [c for c in bucket_features.columns if c not in GRAIN_COLUMNS]
    out[bucket_cols] = out[bucket_cols].fillna(0)
    return out


def add_velocity_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["late_event_share"] = _safe_divide(out["event_count_d4_d7"], out["event_count"])
    out["late_session_share"] = _safe_divide(out["session_count_d4_d7"], out["session_count"])
    out["late_ad_share"] = _safe_divide(out["ad_impression_count_d4_d7"], out["ad_impression_count"])
    out["late_revenue_share"] = _safe_divide(out["revenue_d4_d7"], out["revenue_d0_d7"])
    out["event_growth_d4_d7_vs_d0_d1"] = _safe_divide(out["event_count_d4_d7"], out["event_count_d0"] + out["event_count_d1"])
    out["session_growth_d4_d7_vs_d0_d1"] = _safe_divide(out["session_count_d4_d7"], out["session_count_d0"] + out["session_count_d1"])
    out["ad_growth_d4_d7_vs_d0_d1"] = _safe_divide(out["ad_impression_count_d4_d7"], out["ad_impression_count_d0"] + out["ad_impression_count_d1"])
    out["revenue_growth_d4_d7_vs_d0_d1"] = _safe_divide(out["revenue_d4_d7"], out["revenue_d0"] + out["revenue_d1"])
    return out


def _add_interaction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["platform_country"] = _interaction(out["platform"], out["country_tier"])
    out["country_channel"] = _interaction(out["country_tier"], out["channel_tier"])
    out["platform_channel"] = _interaction(out["platform"], out["channel_tier"])
    out["network_placement"] = _interaction(out["top_network"], out["top_ad_placement"])
    out["country_network"] = _interaction(out["country_tier"], out["top_network"])
    out["channel_network"] = _interaction(out["channel_tier"], out["top_network"])
    return out


def add_frequency_interaction_features(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_out = _add_interaction_columns(train_split)
    valid_out = _add_interaction_columns(valid_split)
    train_rows = max(1, len(train_out))

    for col in INTERACTION_COLUMNS:
        counts = train_out[col].value_counts(dropna=False)
        train_out[f"{col}_count"] = train_out[col].map(counts).fillna(0).astype(float)
        valid_out[f"{col}_count"] = valid_out[col].map(counts).fillna(0).astype(float)
        train_out[f"{col}_freq"] = train_out[f"{col}_count"] / train_rows
        valid_out[f"{col}_freq"] = valid_out[f"{col}_count"] / train_rows

    return train_out, valid_out, {"interaction_columns": INTERACTION_COLUMNS}


def add_target_encoding_features(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_out = train_split.copy()
    valid_out = valid_split.copy()
    for frame in (train_out, valid_out):
        if "country_channel" not in frame.columns:
            frame["country_channel"] = _interaction(frame["country_tier"], frame["channel_tier"])
        frame["platform_country_channel"] = _interaction(frame["platform"], frame["country_tier"], frame["channel_tier"])

    log_target = np.log1p(train_out[TARGET_COLUMN].clip(lower=0).astype(float))
    global_log_mean = float(log_target.mean())
    global_positive_rate = float((train_out[TARGET_COLUMN] > 0).mean())

    for source_col, prefix in TARGET_ENCODING_SPECS:
        encoded_name = f"{prefix}_ltv_log_mean"
        mapping = pd.DataFrame({"key": train_out[source_col], "value": log_target}).groupby("key", dropna=False)["value"].mean()
        train_out[encoded_name] = train_out[source_col].map(mapping).fillna(global_log_mean).astype(float)
        valid_out[encoded_name] = valid_out[source_col].map(mapping).fillna(global_log_mean).astype(float)

    positive = (train_out[TARGET_COLUMN] > 0).astype(float)
    for source_col, encoded_name in POSITIVE_RATE_SPECS:
        mapping = pd.DataFrame({"key": train_out[source_col], "value": positive}).groupby("key", dropna=False)["value"].mean()
        train_out[encoded_name] = train_out[source_col].map(mapping).fillna(global_positive_rate).astype(float)
        valid_out[encoded_name] = valid_out[source_col].map(mapping).fillna(global_positive_rate).astype(float)

    return train_out, valid_out, {
        "fallback_log_mean": global_log_mean,
        "fallback_positive_rate": global_positive_rate,
        "target_encoding_columns": [f"{prefix}_ltv_log_mean" for _, prefix in TARGET_ENCODING_SPECS]
        + [name for _, name in POSITIVE_RATE_SPECS],
        "note": "Encodings are fitted on train split only and mapped to valid with global fallback for unseen categories.",
    }


def get_feature_set_frame(
    feature_set: str,
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    bucket_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_out = train_split.copy()
    valid_out = valid_split.copy()
    metadata: dict[str, Any] = {"feature_set": feature_set}

    if feature_set == "xgb_current_full":
        return train_out, valid_out, metadata

    train_out = attach_time_bucket_features(train_out, bucket_features)
    valid_out = attach_time_bucket_features(valid_out, bucket_features)
    metadata["time_bucket_features"] = [c for c in train_out.columns if any(c.endswith(f"_{label}") for label in BUCKET_LABELS)]
    if feature_set == "xgb_time_bucket_features":
        return train_out, valid_out, metadata

    train_out = add_velocity_ratio_features(train_out)
    valid_out = add_velocity_ratio_features(valid_out)
    metadata["velocity_ratio_features"] = [
        "late_event_share",
        "late_session_share",
        "late_ad_share",
        "late_revenue_share",
        "event_growth_d4_d7_vs_d0_d1",
        "session_growth_d4_d7_vs_d0_d1",
        "ad_growth_d4_d7_vs_d0_d1",
        "revenue_growth_d4_d7_vs_d0_d1",
    ]
    if feature_set == "xgb_velocity_ratio_features":
        return train_out, valid_out, metadata

    train_out, valid_out, interaction_meta = add_frequency_interaction_features(train_out, valid_out)
    metadata.update(interaction_meta)
    if feature_set == "xgb_frequency_interaction_features":
        return train_out, valid_out, metadata

    if feature_set in {"xgb_target_encoding_features", "xgb_all_derived_features"}:
        train_out, valid_out, te_meta = add_target_encoding_features(train_out, valid_out)
        metadata.update(te_meta)
        if feature_set == "xgb_all_derived_features":
            metadata["note"] = (
                metadata.get("note", "")
                + " This explicit all-derived set includes time buckets, velocity ratios, "
                + "frequency/count interaction features, and leakage-safe target encodings."
            ).strip()
        return train_out, valid_out, metadata

    raise ValueError(f"Unknown feature set: {feature_set}")


def build_feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = set(EXCLUDED_FEATURES)
    categorical_features = [
        col
        for col in frame.columns
        if col not in excluded and (col in BASE_CATEGORICAL_FEATURES or frame[col].dtype == "object" or str(frame[col].dtype).startswith("string"))
    ]
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


def fit_xgboost_feature_set(
    feature_set: str,
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> ExperimentResult:
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
    model.fit(x_train_transformed, y_train_log, eval_set=[(x_valid_transformed, y_valid_log)], verbose=False)

    pred = pd.Series(restore_ltv_scale(model.predict(x_valid_transformed)), index=valid_split.index, name=feature_set)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)) if getattr(model, "best_score", None) is not None else None,
    }
    return ExperimentResult(feature_set, pred, evaluate_predictions(valid_split[TARGET_COLUMN], pred), pipeline, metadata)


def build_feature_importance(pipeline: Pipeline, feature_set: str) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            feature_names.extend([str(col) for col in columns])
        elif name == "cat":
            feature_names.extend(transformer.get_feature_names_out(columns).astype(str).tolist())
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    out = pd.DataFrame({"feature_set": feature_set, "feature": feature_names, "importance": importances})
    return out.sort_values(["feature_set", "importance"], ascending=[True, False]).reset_index(drop=True)


def run_experiments(
    train_model_input: pd.DataFrame,
    bucket_features: pd.DataFrame,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_split, valid_split = make_time_split(train_model_input)
    if train_split.empty or valid_split.empty:
        raise ValueError("Time-based split failed: train or validation split is empty.")

    metrics_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    prediction_frame = valid_split[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
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

    for feature_set in FEATURE_SETS:
        fs_train, fs_valid, feature_meta = get_feature_set_frame(feature_set, train_split, valid_split, bucket_features)
        result = fit_xgboost_feature_set(feature_set, fs_train, fs_valid, n_estimators, early_stopping_rounds)
        metrics_rows.append({"feature_set": feature_set, **result.metrics})
        prediction_frame[f"pred_{feature_set}"] = result.prediction.values
        importance_frames.append(build_feature_importance(result.pipeline, feature_set))

        values = pd.Series(result.prediction, dtype=float)
        diagnostics["feature_sets"][feature_set] = {**feature_meta, **result.metadata}
        diagnostics["prediction_checks"][feature_set] = {
            "rows_match_valid": int(len(values)) == int(len(valid_split)),
            "prediction_rows": int(len(values)),
            "valid_rows": int(len(valid_split)),
            "null_predictions": int(values.isna().sum()),
            "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
            "negative_predictions": int((values < 0).sum()),
        }

    metrics = pd.DataFrame(metrics_rows)
    importance = pd.concat(importance_frames, ignore_index=True)
    return metrics, prediction_frame, importance, diagnostics


def _improvement_text(metrics_df: pd.DataFrame, metric: str, lower_is_better: bool) -> str:
    current = metrics_df.loc[metrics_df["feature_set"] == "xgb_current_full", metric].iloc[0]
    if lower_is_better:
        best_row = metrics_df.sort_values(metric, ascending=True).iloc[0]
        delta = current - best_row[metric]
    else:
        best_row = metrics_df.sort_values(metric, ascending=False).iloc[0]
        delta = best_row[metric] - current
    return f"{best_row['feature_set']} ({metric}={best_row[metric]:.6f}, current_full delta={delta:.6f})"


def write_report(path: Path, metrics_df: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    rmsle_best = _improvement_text(metrics_df, "rmsle", lower_is_better=True)
    capture_best = _improvement_text(metrics_df, "top_10pct_revenue_capture", lower_is_better=False)
    current_rmsle = metrics_df.loc[metrics_df["feature_set"] == "xgb_current_full", "rmsle"].iloc[0]
    te_rmsle = metrics_df.loc[metrics_df["feature_set"] == "xgb_target_encoding_features", "rmsle"].iloc[0]
    te_capture = metrics_df.loc[metrics_df["feature_set"] == "xgb_target_encoding_features", "top_10pct_revenue_capture"].iloc[0]
    best_rmsle_feature_set = metrics_df.sort_values("rmsle", ascending=True).iloc[0]["feature_set"]

    lines = [
        "# XGBoost Feature Engineering Experiment Results",
        "",
        "## Scope",
        "",
        "This experiment keeps the XGBoost log-target model and time-based validation split fixed, then changes only the feature family. It does not use Optuna, OOF, random KFold, LightGBM, or two-stage modeling.",
        "",
        "## Validation Split",
        "",
        f"- Train split: install_day {diagnostics['split']['train_install_day_min']} to {diagnostics['split']['train_install_day_max']} ({diagnostics['split']['train_rows']:,} rows)",
        f"- Valid split: install_day {diagnostics['split']['valid_install_day_min']} to {diagnostics['split']['valid_install_day_max']} ({diagnostics['split']['valid_rows']:,} rows)",
        "",
        "## Metrics",
        "",
        "| feature set | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"| {row['feature_set']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
            f"{row['spearman_corr']:.4f} | {row['positive_ltv_rate_pred_top_decile']:.2%} | "
            f"{row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Questions",
            "",
            f"- Best RMSLE improvement versus current_full: {rmsle_best}.",
            f"- Best top 10% revenue capture improvement versus current_full: {capture_best}.",
            f"- Target encoding result: target encoding RMSLE={te_rmsle:.6f}, top 10% capture={te_capture:.2%}. Current full RMSLE={current_rmsle:.6f}. Because target encodings are train-split fitted and not OOF, validation mapping is leakage-safe but train-side overfit risk remains; this should be revisited with rolling/OOF encoding later.",
            f"- Final feature set candidate: `{best_rmsle_feature_set}` for the next modeling stage, unless the business goal prioritizes top-decile capture over RMSLE.",
            "- `xgb_all_derived_features` is an explicit full derived-feature set: time buckets + velocity ratios + frequency/count interactions + leakage-safe target encodings.",
            "",
            "## Prediction Checks",
            "",
            "| feature set | rows match valid | null predictions | inf predictions | negative predictions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for feature_set, check in diagnostics["prediction_checks"].items():
        lines.append(
            f"| {feature_set} | {check['rows_match_valid']} | {check['null_predictions']:,} | "
            f"{check['inf_predictions']:,} | {check['negative_predictions']:,} |"
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
    metrics, predictions, importance, diagnostics = run_experiments(train_model_input, bucket_features)

    metrics.to_csv(processed_dir / "feature_engineering_metrics.csv", index=False)
    predictions.to_parquet(processed_dir / "feature_engineering_valid_predictions.parquet", index=False)
    importance.to_csv(processed_dir / "feature_engineering_importance.csv", index=False)
    (processed_dir / "feature_engineering_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(reports_dir / "feature_engineering_results.md", metrics, diagnostics)


if __name__ == "__main__":
    main()

