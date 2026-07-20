from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "ltv_d8_d180"
GRAIN_COLUMNS = ["user_id", "platform", "country_tier", "channel_tier", "install_day"]
CATEGORICAL_SENTINELS = {
    "top_network": "no_ad_network",
    "top_ad_placement": "no_ad_placement",
}
NUMERIC_SENTINELS = {
    "days_to_first_iap": 99,
}


def apply_model_input_preprocessing(frame: pd.DataFrame) -> pd.DataFrame:
    processed = frame.copy()

    for column, value in CATEGORICAL_SENTINELS.items():
        if column in processed.columns:
            processed[column] = processed[column].fillna(value)

    for column, value in NUMERIC_SENTINELS.items():
        if column in processed.columns:
            processed[column] = processed[column].fillna(value)

    numeric_columns = processed.select_dtypes(include=[np.number]).columns
    processed[numeric_columns] = processed[numeric_columns].replace([np.inf, -np.inf], np.nan)
    processed[numeric_columns] = processed[numeric_columns].fillna(0)

    return processed


def _numeric_health(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return {
            "numeric_column_count": 0,
            "numeric_null_cells": 0,
            "numeric_inf_cells": 0,
            "columns_with_numeric_nulls": [],
            "columns_with_numeric_inf": [],
        }
    null_counts = numeric.isna().sum()
    inf_counts = pd.Series(
        np.isinf(numeric.to_numpy(dtype=float)).sum(axis=0),
        index=numeric.columns,
    )
    return {
        "numeric_column_count": int(len(numeric.columns)),
        "numeric_null_cells": int(null_counts.sum()),
        "numeric_inf_cells": int(inf_counts.sum()),
        "columns_with_numeric_nulls": null_counts[null_counts > 0].index.tolist(),
        "columns_with_numeric_inf": inf_counts[inf_counts > 0].index.tolist(),
    }


def _categorical_nulls(frame: pd.DataFrame) -> dict[str, Any]:
    object_columns = frame.select_dtypes(include=["object", "string", "category"]).columns
    null_counts = frame[object_columns].isna().sum() if len(object_columns) else pd.Series(dtype=int)
    return {
        "categorical_column_count": int(len(object_columns)),
        "categorical_null_cells": int(null_counts.sum()) if len(object_columns) else 0,
        "columns_with_categorical_nulls": null_counts[null_counts > 0].index.tolist() if len(object_columns) else [],
        "null_counts": {col: int(value) for col, value in null_counts.items() if value > 0},
    }


def _target_distribution(train: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    positive = target > 0
    quantiles = target.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "row_count": int(len(target)),
        "null_count": int(target.isna().sum()),
        "positive_rate": float(positive.mean()),
        "zero_rate": float((target == 0).mean()),
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(quantiles.loc[0]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
    }


def validate_feature_tables(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    train_feature_columns = [col for col in train.columns if col != TARGET_COLUMN]
    test_feature_columns = list(test.columns)
    train_set = set(train_feature_columns)
    test_set = set(test_feature_columns)

    split_train_rows = int((train["install_day"].between(0, 23)).sum())
    split_valid_rows = int((train["install_day"].between(24, 30)).sum())

    report = {
        "schema": {
            "train_columns": list(train.columns),
            "test_columns": list(test.columns),
            "feature_columns_match": train_feature_columns == test_feature_columns,
            "missing_in_test": sorted(train_set - test_set),
            "extra_in_test": sorted(test_set - train_set),
            "target_in_train": TARGET_COLUMN in train.columns,
            "target_in_test": TARGET_COLUMN in test.columns,
            "target_only_in_train": TARGET_COLUMN in train.columns and TARGET_COLUMN not in test.columns,
        },
        "grain": {
            "grain_columns": GRAIN_COLUMNS,
            "train_duplicate_grain_rows": int(train.duplicated(GRAIN_COLUMNS).sum()),
            "test_duplicate_grain_rows": int(test.duplicated(GRAIN_COLUMNS).sum()),
            "train_unique_grain_rows": int(train[GRAIN_COLUMNS].drop_duplicates().shape[0]),
            "test_unique_grain_rows": int(test[GRAIN_COLUMNS].drop_duplicates().shape[0]),
        },
        "raw_numeric_health": {
            "train": _numeric_health(train),
            "test": _numeric_health(test),
        },
        "raw_categorical_nulls": {
            "train": _categorical_nulls(train),
            "test": _categorical_nulls(test),
        },
        "target": _target_distribution(train),
        "split": {
            "primary_split": "train install_day 0-23, valid install_day 24-30",
            "time_split_possible": split_train_rows > 0 and split_valid_rows > 0,
            "train_rows_install_day_0_23": split_train_rows,
            "valid_rows_install_day_24_30": split_valid_rows,
            "train_install_day_min": int(train["install_day"].min()),
            "train_install_day_max": int(train["install_day"].max()),
            "test_install_day_min": int(test["install_day"].min()),
            "test_install_day_max": int(test["install_day"].max()),
        },
        "preprocessing_rules": {
            "categorical_sentinels": CATEGORICAL_SENTINELS,
            "numeric_sentinels": NUMERIC_SENTINELS,
            "numeric_null_or_inf": "replace +/-inf with null, then fill numeric nulls with 0",
        },
    }
    return report


def write_validation_report(path: Path, report: dict[str, Any], processed_train: pd.DataFrame, processed_test: pd.DataFrame) -> None:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    schema = report["schema"]
    grain = report["grain"]
    split = report["split"]
    target = report["target"]
    raw_numeric = report["raw_numeric_health"]
    raw_cat = report["raw_categorical_nulls"]

    processed_numeric_train = _numeric_health(processed_train)
    processed_numeric_test = _numeric_health(processed_test)
    processed_cat_train = _categorical_nulls(processed_train)
    processed_cat_test = _categorical_nulls(processed_test)

    lines = [
        "# Model Input Validation Report",
        "",
        "## Scope",
        "",
        "This step validates the feature tables and fixes model-input preprocessing rules. No model training was performed.",
        "",
        "## Schema Checks",
        "",
        f"- Train/test feature columns match after excluding target: `{schema['feature_columns_match']}`",
        f"- Target exists in train: `{schema['target_in_train']}`",
        f"- Target exists in test: `{schema['target_in_test']}`",
        f"- Target only in train: `{schema['target_only_in_train']}`",
        f"- Missing columns in test: `{schema['missing_in_test']}`",
        f"- Extra columns in test: `{schema['extra_in_test']}`",
        "",
        "## Grain Checks",
        "",
        f"- Grain: `{' + '.join(grain['grain_columns'])}`",
        f"- Train duplicate grain rows: {grain['train_duplicate_grain_rows']:,}",
        f"- Test duplicate grain rows: {grain['test_duplicate_grain_rows']:,}",
        f"- Train unique grain rows: {grain['train_unique_grain_rows']:,}",
        f"- Test unique grain rows: {grain['test_unique_grain_rows']:,}",
        "",
        "## Raw Null / Inf Checks",
        "",
        "| split | numeric null cells | numeric inf cells | categorical null cells | categorical null columns |",
        "|---|---:|---:|---:|---|",
        f"| train | {raw_numeric['train']['numeric_null_cells']:,} | {raw_numeric['train']['numeric_inf_cells']:,} | {raw_cat['train']['categorical_null_cells']:,} | {raw_cat['train']['columns_with_categorical_nulls']} |",
        f"| test | {raw_numeric['test']['numeric_null_cells']:,} | {raw_numeric['test']['numeric_inf_cells']:,} | {raw_cat['test']['categorical_null_cells']:,} | {raw_cat['test']['columns_with_categorical_nulls']} |",
        "",
        "## Preprocessing Rules",
        "",
        "- `days_to_first_iap` null -> `99`",
        "- `top_network` null -> `no_ad_network`",
        "- `top_ad_placement` null -> `no_ad_placement`",
        "- numeric null/inf -> `0` after applying specific sentinels",
        "",
        "## Post-Preprocessing Checks",
        "",
        "| split | numeric null cells | numeric inf cells | categorical null cells |",
        "|---|---:|---:|---:|",
        f"| train | {processed_numeric_train['numeric_null_cells']:,} | {processed_numeric_train['numeric_inf_cells']:,} | {processed_cat_train['categorical_null_cells']:,} |",
        f"| test | {processed_numeric_test['numeric_null_cells']:,} | {processed_numeric_test['numeric_inf_cells']:,} | {processed_cat_test['categorical_null_cells']:,} |",
        "",
        "## Target Distribution",
        "",
        f"- Rows: {target['row_count']:,}",
        f"- Target nulls: {target['null_count']:,}",
        f"- Positive LTV rate: {pct(target['positive_rate'])}",
        f"- Zero LTV rate: {pct(target['zero_rate'])}",
        f"- Mean: {target['mean']:.4f}",
        f"- P50: {target['p50']:.4f}",
        f"- P75: {target['p75']:.4f}",
        f"- P95: {target['p95']:.4f}",
        f"- P99: {target['p99']:.4f}",
        f"- Max: {target['max']:.4f}",
        "",
        "## Time-Based Split Check",
        "",
        f"- Primary split: {split['primary_split']}",
        f"- Time split possible: `{split['time_split_possible']}`",
        f"- Train rows for install_day 0-23: {split['train_rows_install_day_0_23']:,}",
        f"- Validation rows for install_day 24-30: {split['valid_rows_install_day_24_30']:,}",
        f"- Train install_day range: {split['train_install_day_min']} to {split['train_install_day_max']}",
        f"- Test install_day range: {split['test_install_day_min']} to {split['test_install_day_max']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(feature_root: Path) -> dict[str, Any]:
    processed_dir = feature_root / "data" / "processed"
    reports_dir = feature_root / "reports" / "diagnostics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train_features.parquet")
    test = pd.read_parquet(processed_dir / "test_features.parquet")
    report = validate_feature_tables(train, test)

    train_model_input = apply_model_input_preprocessing(train)
    test_model_input = apply_model_input_preprocessing(test)

    train_model_input.to_parquet(processed_dir / "train_model_input.parquet", index=False)
    test_model_input.to_parquet(processed_dir / "test_model_input.parquet", index=False)
    (processed_dir / "model_input_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_validation_report(reports_dir / "model_input_validation_report.md", report, train_model_input, test_model_input)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    args = parser.parse_args()
    run_validation(Path(args.feature_root))


if __name__ == "__main__":
    main()
