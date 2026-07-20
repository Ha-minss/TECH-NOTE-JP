from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "ltv_d8_d180"
SEGMENT_COLUMNS = ["platform", "country_tier", "channel_tier", "install_week"]
GRAIN_COLUMNS = ["user_id", "platform", "country_tier", "channel_tier", "install_day"]
BASELINES = ["global_mean", "segment_mean", "early_revenue_multiplier"]


def make_time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split = frame[frame["install_day"].between(0, 23)].copy()
    valid_split = frame[frame["install_day"].between(24, 30)].copy()
    return train_split, valid_split


def predict_global_mean(train_split: pd.DataFrame, valid_split: pd.DataFrame) -> pd.Series:
    global_mean = float(train_split[TARGET_COLUMN].mean())
    return pd.Series(global_mean, index=valid_split.index, name="global_mean")


def predict_segment_mean(train_split: pd.DataFrame, valid_split: pd.DataFrame) -> pd.Series:
    global_mean = float(train_split[TARGET_COLUMN].mean())
    segment_mean = train_split.groupby(SEGMENT_COLUMNS, dropna=False)[TARGET_COLUMN].mean()
    valid_index = pd.MultiIndex.from_frame(valid_split[SEGMENT_COLUMNS])
    mapped = pd.Series(valid_index.map(segment_mean), index=valid_split.index, dtype=float)
    return mapped.fillna(global_mean).rename("segment_mean")


def predict_early_revenue_multiplier(train_split: pd.DataFrame, valid_split: pd.DataFrame) -> pd.Series:
    global_mean = float(train_split[TARGET_COLUMN].mean())
    early_revenue_mean = float(train_split["revenue_d0_d7"].mean())
    if early_revenue_mean == 0 or not np.isfinite(early_revenue_mean):
        return pd.Series(global_mean, index=valid_split.index, name="early_revenue_multiplier")
    multiplier = global_mean / early_revenue_mean
    prediction = valid_split["revenue_d0_d7"].astype(float) * multiplier
    prediction = prediction.clip(lower=0).replace([np.inf, -np.inf], np.nan).fillna(global_mean)
    return prediction.rename("early_revenue_multiplier")


def _top_decile_mask(y_pred: pd.Series) -> pd.Series:
    if len(y_pred) == 0:
        return pd.Series([], dtype=bool)
    top_n = max(1, int(np.ceil(len(y_pred) * 0.10)))
    ranked_index = y_pred.sort_values(ascending=False, kind="mergesort").head(top_n).index
    mask = pd.Series(False, index=y_pred.index)
    mask.loc[ranked_index] = True
    return mask


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = pd.Series(y_true, dtype=float).reset_index(drop=True)
    pred = pd.Series(y_pred, dtype=float).reset_index(drop=True).clip(lower=0)
    pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0)

    error = pred - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    rmsle = float(np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(actual.clip(lower=0))))))
    if actual.nunique(dropna=False) <= 1 or pred.nunique(dropna=False) <= 1:
        spearman = 0.0
    else:
        spearman = float(actual.rank(method="average").corr(pred.rank(method="average"), method="pearson"))
        if not np.isfinite(spearman):
            spearman = 0.0

    top_mask = _top_decile_mask(pred)
    top_actual = actual[top_mask]
    positive_rate_top = float((top_actual > 0).mean()) if len(top_actual) else 0.0
    total_revenue = float(actual.sum())
    top_revenue = float(top_actual.sum()) if len(top_actual) else 0.0
    capture = top_revenue / total_revenue if total_revenue > 0 else 0.0
    top_share = len(top_actual) / len(actual) if len(actual) else 0.0
    lift = capture / top_share if top_share > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "rmsle": rmsle,
        "spearman_corr": spearman,
        "positive_ltv_rate_pred_top_decile": positive_rate_top,
        "top_10pct_revenue_capture": float(capture),
        "top_decile_lift": float(lift),
    }


def validate_baseline_predictions(valid_split: pd.DataFrame, predictions: dict[str, pd.Series]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    valid_rows = len(valid_split)
    for name, pred in predictions.items():
        values = pd.Series(pred, dtype=float)
        checks[name] = {
            "row_count_matches_valid": int(len(values)) == int(valid_rows),
            "prediction_rows": int(len(values)),
            "valid_rows": int(valid_rows),
            "null_predictions": int(values.isna().sum()),
            "inf_predictions": int(np.isinf(values.to_numpy(dtype=float)).sum()),
            "negative_predictions": int((values < 0).sum()),
        }
    return checks


def run_baselines(train_model_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_split, valid_split = make_time_split(train_model_input)
    if train_split.empty or valid_split.empty:
        raise ValueError("Time-based split failed: train or validation split is empty.")

    predictions = {
        "global_mean": predict_global_mean(train_split, valid_split),
        "segment_mean": predict_segment_mean(train_split, valid_split),
        "early_revenue_multiplier": predict_early_revenue_multiplier(train_split, valid_split),
    }

    metrics_rows = []
    for name, pred in predictions.items():
        metrics = evaluate_predictions(valid_split[TARGET_COLUMN], pred)
        metrics_rows.append({"baseline": name, **metrics})
    metrics_df = pd.DataFrame(metrics_rows)

    prediction_frame = valid_split[GRAIN_COLUMNS + [TARGET_COLUMN, "revenue_d0_d7"]].copy()
    for name, pred in predictions.items():
        prediction_frame[f"pred_{name}"] = pd.Series(pred, index=valid_split.index).values

    diagnostics = {
        "split": {
            "train_rows": int(len(train_split)),
            "valid_rows": int(len(valid_split)),
            "train_install_day_min": int(train_split["install_day"].min()),
            "train_install_day_max": int(train_split["install_day"].max()),
            "valid_install_day_min": int(valid_split["install_day"].min()),
            "valid_install_day_max": int(valid_split["install_day"].max()),
        },
        "prediction_checks": validate_baseline_predictions(valid_split, predictions),
        "global_mean": float(train_split[TARGET_COLUMN].mean()),
        "early_revenue_multiplier": (
            float(train_split[TARGET_COLUMN].mean() / train_split["revenue_d0_d7"].mean())
            if float(train_split["revenue_d0_d7"].mean()) != 0
            else None
        ),
        "segment_count": int(train_split[SEGMENT_COLUMNS].drop_duplicates().shape[0]),
    }
    return metrics_df, prediction_frame, diagnostics


def write_report(path: Path, metrics_df: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    lines = [
        "# Baseline Results",
        "",
        "## Scope",
        "",
        "This report compares non-ML/early-stage baselines on the time-based validation split. No LightGBM, XGBoost, Optuna, or advanced model training was performed.",
        "",
        "## Validation Split",
        "",
        f"- Train split: install_day {diagnostics['split']['train_install_day_min']} to {diagnostics['split']['train_install_day_max']} ({diagnostics['split']['train_rows']:,} rows)",
        f"- Valid split: install_day {diagnostics['split']['valid_install_day_min']} to {diagnostics['split']['valid_install_day_max']} ({diagnostics['split']['valid_rows']:,} rows)",
        "",
        "## Baselines",
        "",
        "- Global Mean: train split mean LTV for every validation row.",
        "- Segment Mean: platform + country_tier + channel_tier + install_week mean, with global mean fallback for unseen segments.",
        "- Early Revenue Multiplier: validation D0-D7 revenue multiplied by train mean LTV / train mean D0-D7 revenue, clipped at 0.",
        "",
        "## Metrics",
        "",
        "| baseline | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"| {row['baseline']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['rmsle']:.4f} | "
            f"{row['spearman_corr']:.4f} | {row['positive_ltv_rate_pred_top_decile']:.2%} | "
            f"{row['top_10pct_revenue_capture']:.2%} | {row['top_decile_lift']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Prediction Checks",
            "",
            "| baseline | rows match valid | null predictions | inf predictions | negative predictions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for baseline, check in diagnostics["prediction_checks"].items():
        lines.append(
            f"| {baseline} | {check['row_count_matches_valid']} | {check['null_predictions']:,} | "
            f"{check['inf_predictions']:,} | {check['negative_predictions']:,} |"
        )

    lines.extend(
        [
            "",
            "## Training Constants",
            "",
            f"- Global mean LTV: {diagnostics['global_mean']:.6f}",
            f"- Early revenue multiplier: {diagnostics['early_revenue_multiplier']:.6f}" if diagnostics["early_revenue_multiplier"] is not None else "- Early revenue multiplier: global mean fallback was used.",
            f"- Segment count in train split: {diagnostics['segment_count']:,}",
            "",
            "## Interpretation",
            "",
            "These baselines define the first bar for later ML models. Advanced models should improve not only RMSE/RMSLE, but also ranking-oriented metrics such as top-decile revenue capture.",
            "",
        ]
    )
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
    metrics_df, prediction_frame, diagnostics = run_baselines(train_model_input)

    metrics_df.to_csv(processed_dir / "baseline_metrics.csv", index=False)
    prediction_frame.to_parquet(processed_dir / "baseline_valid_predictions.parquet", index=False)
    (processed_dir / "baseline_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(reports_dir / "baseline_results.md", metrics_df, diagnostics)


if __name__ == "__main__":
    main()


