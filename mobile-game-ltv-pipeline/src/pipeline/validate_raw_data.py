from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "train.csv": {
        "user_id", "platform", "country_tier", "channel_tier", "install_day", "install_week",
        "day_since_install", "event_hour", "event_type", "event_name", "product_id", "network",
        "ad_placement", "revenue_usd", "ltv_d8_d180",
    },
    "test.csv": {
        "user_id", "platform", "country_tier", "channel_tier", "install_day", "install_week",
        "day_since_install", "event_hour", "event_type", "event_name", "product_id", "network",
        "ad_placement", "revenue_usd",
    },
}
VALID_EVENT_TYPES = {"session", "ad_impression", "iap"}


def _validate_csv(zf: zipfile.ZipFile, name: str, chunksize: int) -> dict[str, Any]:
    required = REQUIRED_COLUMNS[name]
    row_count = 0
    null_counts: dict[str, int] = {}
    invalid_day_rows = 0
    invalid_hour_rows = 0
    invalid_event_type_rows = 0
    negative_revenue_rows = 0
    session_revenue_non_null_rows = 0
    non_session_revenue_null_rows = 0
    columns: list[str] | None = None

    with zf.open(name) as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize, low_memory=False):
            if columns is None:
                columns = list(chunk.columns)
                null_counts = {col: 0 for col in columns}
            row_count += len(chunk)
            for col, value in chunk.isna().sum().items():
                null_counts[col] = null_counts.get(col, 0) + int(value)
            invalid_day_rows += int((~chunk["day_since_install"].between(0, 7)).sum())
            invalid_hour_rows += int((~chunk["event_hour"].between(0, 23)).sum())
            invalid_event_type_rows += int((~chunk["event_type"].isin(VALID_EVENT_TYPES)).sum())
            revenue = pd.to_numeric(chunk["revenue_usd"], errors="coerce")
            negative_revenue_rows += int((revenue < 0).sum())
            session_mask = chunk["event_type"].eq("session")
            session_revenue_non_null_rows += int(revenue[session_mask].notna().sum())
            non_session_revenue_null_rows += int(revenue[~session_mask].isna().sum())

    columns = columns or []
    missing = sorted(required - set(columns))
    extra = sorted(set(columns) - required)
    return {
        "file": name,
        "row_count": row_count,
        "columns": columns,
        "required_columns_present": len(missing) == 0,
        "missing_required_columns": missing,
        "extra_columns": extra,
        "invalid_day_rows": invalid_day_rows,
        "invalid_hour_rows": invalid_hour_rows,
        "invalid_event_type_rows": invalid_event_type_rows,
        "negative_revenue_rows": negative_revenue_rows,
        "session_revenue_non_null_rows": session_revenue_non_null_rows,
        "non_session_revenue_null_rows": non_session_revenue_null_rows,
        "null_counts": null_counts,
        "null_rates": {col: (count / row_count if row_count else 0.0) for col, count in null_counts.items()},
    }


def validate_raw_data(zip_path: Path, project_root: Path, chunksize: int = 500_000) -> dict[str, Any]:
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports" / "diagnostics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        files_present = set(zf.namelist())
        report = {
            "zip_path": str(zip_path),
            "files_present": sorted(files_present),
            "required_files_present": all(name in files_present for name in ["train.csv", "test.csv", "sample_submission.csv"]),
            "files": {
                "train.csv": _validate_csv(zf, "train.csv", chunksize),
                "test.csv": _validate_csv(zf, "test.csv", chunksize),
            },
        }

    (processed_dir / "raw_data_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_raw_validation_report(reports_dir / "raw_data_validation_report.md", report)
    return report


def write_raw_validation_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Raw Data Validation Report",
        "",
        "## Scope",
        "",
        "Validates raw event-log files before feature building. No feature aggregation or model training is performed here.",
        "",
        f"- Required files present: `{report['required_files_present']}`",
        f"- Files in zip: `{report['files_present']}`",
        "",
        "## Quality Checks",
        "",
        "| file | rows | required columns | invalid day | invalid hour | invalid event type | negative revenue | session revenue non-null | non-session revenue null |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, info in report["files"].items():
        lines.append(
            f"| {name} | {info['row_count']:,} | {info['required_columns_present']} | "
            f"{info['invalid_day_rows']:,} | {info['invalid_hour_rows']:,} | {info['invalid_event_type_rows']:,} | "
            f"{info['negative_revenue_rows']:,} | {info['session_revenue_non_null_rows']:,} | {info['non_session_revenue_null_rows']:,} |"
        )
    lines.extend(["", "## Notes", "", "- `session` rows are expected to have null revenue before feature aggregation.", "- IAP/ad rows are expected to have non-null revenue.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    validate_raw_data(Path(args.zip_path), Path(args.project_root), args.chunksize)


if __name__ == "__main__":
    main()
