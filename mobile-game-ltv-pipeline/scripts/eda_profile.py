from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["platform", "country_tier", "channel_tier", "install_day", "install_week"]
EVENT_COLUMNS = [
    "day_since_install",
    "event_hour",
    "event_type",
    "event_name",
    "product_id",
    "network",
    "ad_placement",
    "revenue_usd",
]
CATEGORICAL_COLUMNS = [
    "platform",
    "country_tier",
    "channel_tier",
    "event_type",
    "event_name",
    "product_id",
    "network",
    "ad_placement",
]
NUMERIC_COLUMNS = ["install_day", "install_week", "day_since_install", "event_hour", "revenue_usd"]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def summarize_series(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    quantiles = values.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "count": int(values.shape[0]),
        "mean": safe_float(values.mean()),
        "std": safe_float(values.std()),
        "min": safe_float(quantiles.loc[0]),
        "p01": safe_float(quantiles.loc[0.01]),
        "p05": safe_float(quantiles.loc[0.05]),
        "p25": safe_float(quantiles.loc[0.25]),
        "p50": safe_float(quantiles.loc[0.5]),
        "p75": safe_float(quantiles.loc[0.75]),
        "p95": safe_float(quantiles.loc[0.95]),
        "p99": safe_float(quantiles.loc[0.99]),
        "max": safe_float(quantiles.loc[1.0]),
        "zero_rate": safe_float((values == 0).mean()),
    }


def top_counter(counter: Counter, n: int = 20) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {"value": str(value), "count": int(count), "share": count / total if total else None}
        for value, count in counter.most_common(n)
    ]


def update_value_sets(store: dict[int, dict[str, set]], df: pd.DataFrame, columns: list[str]) -> None:
    slim = df[["user_id", *columns]].drop_duplicates()
    for row in slim.itertuples(index=False):
        user_id = int(row[0])
        for col, value in zip(columns, row[1:]):
            if pd.isna(value):
                value = "<NA>"
            store[user_id][col].add(value)


def profile_csv_from_zip(zip_path: Path, member: str, chunksize: int, has_target: bool) -> dict[str, Any]:
    row_count = 0
    column_order: list[str] | None = None
    null_counts: Counter = Counter()
    empty_string_counts: Counter = Counter()
    categorical_counts: dict[str, Counter] = {col: Counter() for col in CATEGORICAL_COLUMNS}
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    negative_revenue_rows = 0
    session_revenue_non_null = 0
    non_session_revenue_null = 0
    invalid_day_rows = 0
    invalid_hour_rows = 0
    invalid_event_type_rows = 0

    user_static_sets: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    user_target_sets: dict[int, set] = defaultdict(set)
    target_by_user: dict[int, float] = {}

    event_by_user: Counter = Counter()
    session_by_user: Counter = Counter()
    iap_by_user: Counter = Counter()
    ad_by_user: Counter = Counter()
    active_days_by_user: dict[int, set] = defaultdict(set)
    event_names_by_user: dict[int, set] = defaultdict(set)
    networks_by_user: dict[int, set] = defaultdict(set)
    products_by_user: dict[int, set] = defaultdict(set)
    first_day_by_user: dict[int, int] = {}
    last_day_by_user: dict[int, int] = {}
    first_iap_day_by_user: dict[int, int] = {}
    first_ad_day_by_user: dict[int, int] = {}
    first_event_hour_by_user: dict[int, int] = {}
    last_event_hour_by_user: dict[int, int] = {}
    total_revenue_by_user: Counter = Counter()
    iap_revenue_by_user: Counter = Counter()
    ad_revenue_by_user: Counter = Counter()
    revenue_by_day_bucket: dict[str, Counter] = {
        "d0": Counter(),
        "d1": Counter(),
        "d2_d3": Counter(),
        "d4_d7": Counter(),
    }
    event_type_by_day = Counter()
    rows_by_install_week = Counter()
    users_by_install_week: dict[int, set] = defaultdict(set)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            reader = pd.read_csv(fh, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                if column_order is None:
                    column_order = list(chunk.columns)
                row_count += int(chunk.shape[0])

                null_counts.update(chunk.isna().sum().to_dict())
                for col in CATEGORICAL_COLUMNS:
                    if col in chunk.columns:
                        empty_string_counts[col] += int((chunk[col].fillna("") == "").sum())
                        categorical_counts[col].update(chunk[col].fillna("<NA>").astype(str).value_counts().to_dict())

                for col in NUMERIC_COLUMNS:
                    if col in chunk.columns:
                        series = pd.to_numeric(chunk[col], errors="coerce")
                        if series.notna().any():
                            numeric_min[col] = float(min(numeric_min.get(col, math.inf), series.min()))
                            numeric_max[col] = float(max(numeric_max.get(col, -math.inf), series.max()))

                update_value_sets(user_static_sets, chunk, STATIC_COLUMNS)
                if has_target:
                    target_slim = chunk[["user_id", "ltv_d8_d180"]].drop_duplicates()
                    for row in target_slim.itertuples(index=False):
                        value = float(row.ltv_d8_d180)
                        user_target_sets[int(row.user_id)].add(value)
                        target_by_user[int(row.user_id)] = value

                valid_revenue = chunk["revenue_usd"].fillna(0)
                negative_revenue_rows += int((valid_revenue < 0).sum())
                session_revenue_non_null += int(
                    ((chunk["event_type"] == "session") & chunk["revenue_usd"].notna()).sum()
                )
                non_session_revenue_null += int(
                    ((chunk["event_type"] != "session") & chunk["revenue_usd"].isna()).sum()
                )
                invalid_day_rows += int((~chunk["day_since_install"].between(0, 7)).sum())
                invalid_hour_rows += int((~chunk["event_hour"].between(0, 23)).sum())
                invalid_event_type_rows += int((~chunk["event_type"].isin(["session", "iap", "ad_impression"])).sum())

                rows_by_install_week.update(chunk["install_week"].value_counts().to_dict())
                for week, users in chunk.groupby("install_week")["user_id"].unique().items():
                    users_by_install_week[int(week)].update(int(u) for u in users)

                user_group = chunk.groupby("user_id", sort=False)
                event_by_user.update(user_group.size().to_dict())
                session_by_user.update(chunk.loc[chunk["event_type"] == "session"].groupby("user_id").size().to_dict())
                iap_by_user.update(chunk.loc[chunk["event_type"] == "iap"].groupby("user_id").size().to_dict())
                ad_by_user.update(
                    chunk.loc[chunk["event_type"] == "ad_impression"].groupby("user_id").size().to_dict()
                )

                for user_id, days in user_group["day_since_install"].unique().items():
                    uid = int(user_id)
                    day_values = [int(day) for day in days]
                    active_days_by_user[uid].update(day_values)
                    mn = min(day_values)
                    mx = max(day_values)
                    first_day_by_user[uid] = min(first_day_by_user.get(uid, mn), mn)
                    last_day_by_user[uid] = max(last_day_by_user.get(uid, mx), mx)

                first_hour = user_group["event_hour"].min()
                last_hour = user_group["event_hour"].max()
                for user_id, hour in first_hour.items():
                    uid = int(user_id)
                    first_event_hour_by_user[uid] = min(first_event_hour_by_user.get(uid, int(hour)), int(hour))
                for user_id, hour in last_hour.items():
                    uid = int(user_id)
                    last_event_hour_by_user[uid] = max(last_event_hour_by_user.get(uid, int(hour)), int(hour))

                for user_id, values in user_group["event_name"].unique().items():
                    event_names_by_user[int(user_id)].update(str(v) for v in values if pd.notna(v))
                for user_id, values in user_group["network"].unique().items():
                    networks_by_user[int(user_id)].update(str(v) for v in values if pd.notna(v))
                for user_id, values in user_group["product_id"].unique().items():
                    products_by_user[int(user_id)].update(str(v) for v in values if pd.notna(v))

                revenue_rows = chunk.loc[chunk["revenue_usd"].notna()].copy()
                if not revenue_rows.empty:
                    total_revenue_by_user.update(revenue_rows.groupby("user_id")["revenue_usd"].sum().to_dict())
                    iap_revenue_by_user.update(
                        revenue_rows.loc[revenue_rows["event_type"] == "iap"].groupby("user_id")["revenue_usd"].sum().to_dict()
                    )
                    ad_revenue_by_user.update(
                        revenue_rows.loc[revenue_rows["event_type"] == "ad_impression"]
                        .groupby("user_id")["revenue_usd"]
                        .sum()
                        .to_dict()
                    )

                    bucket_series = pd.cut(
                        revenue_rows["day_since_install"],
                        bins=[-1, 0, 1, 3, 7],
                        labels=["d0", "d1", "d2_d3", "d4_d7"],
                    )
                    revenue_rows = revenue_rows.assign(day_bucket=bucket_series)
                    for bucket, grouped in revenue_rows.groupby("day_bucket", observed=True):
                        revenue_by_day_bucket[str(bucket)].update(grouped.groupby("user_id")["revenue_usd"].sum().to_dict())

                iap_rows = chunk.loc[chunk["event_type"] == "iap", ["user_id", "day_since_install"]]
                if not iap_rows.empty:
                    for user_id, day in iap_rows.groupby("user_id")["day_since_install"].min().items():
                        uid = int(user_id)
                        first_iap_day_by_user[uid] = min(first_iap_day_by_user.get(uid, int(day)), int(day))

                ad_rows = chunk.loc[chunk["event_type"] == "ad_impression", ["user_id", "day_since_install"]]
                if not ad_rows.empty:
                    for user_id, day in ad_rows.groupby("user_id")["day_since_install"].min().items():
                        uid = int(user_id)
                        first_ad_day_by_user[uid] = min(first_ad_day_by_user.get(uid, int(day)), int(day))

                event_type_by_day.update(
                    chunk.groupby(["day_since_install", "event_type"]).size().to_dict()
                )

    user_ids = sorted(event_by_user)
    user_features = pd.DataFrame({"user_id": user_ids})
    user_features["event_count"] = user_features["user_id"].map(event_by_user).fillna(0).astype(int)
    user_features["session_count"] = user_features["user_id"].map(session_by_user).fillna(0).astype(int)
    user_features["iap_count"] = user_features["user_id"].map(iap_by_user).fillna(0).astype(int)
    user_features["ad_impression_count"] = user_features["user_id"].map(ad_by_user).fillna(0).astype(int)
    user_features["active_days"] = user_features["user_id"].map(lambda u: len(active_days_by_user[int(u)]))
    user_features["first_event_day"] = user_features["user_id"].map(first_day_by_user)
    user_features["last_event_day"] = user_features["user_id"].map(last_day_by_user)
    user_features["unique_event_names"] = user_features["user_id"].map(lambda u: len(event_names_by_user[int(u)]))
    user_features["unique_ad_networks"] = user_features["user_id"].map(lambda u: len(networks_by_user[int(u)]))
    user_features["unique_products"] = user_features["user_id"].map(lambda u: len(products_by_user[int(u)]))
    user_features["revenue_d0_d7"] = user_features["user_id"].map(total_revenue_by_user).fillna(0.0)
    user_features["iap_revenue_d0_d7"] = user_features["user_id"].map(iap_revenue_by_user).fillna(0.0)
    user_features["ad_revenue_d0_d7"] = user_features["user_id"].map(ad_revenue_by_user).fillna(0.0)
    for bucket, counter in revenue_by_day_bucket.items():
        user_features[f"revenue_{bucket}"] = user_features["user_id"].map(counter).fillna(0.0)
    user_features["has_iap_d0_d7"] = (user_features["iap_count"] > 0).astype(int)
    user_features["has_revenue_d0_d7"] = (user_features["revenue_d0_d7"] > 0).astype(int)
    user_features["days_to_first_iap"] = user_features["user_id"].map(first_iap_day_by_user)
    user_features["days_to_first_ad"] = user_features["user_id"].map(first_ad_day_by_user)

    if has_target:
        user_features["ltv_d8_d180"] = user_features["user_id"].map(target_by_user).fillna(0.0)
        user_features["future_payer"] = (user_features["ltv_d8_d180"] > 0).astype(int)

    inconsistent_static = {
        col: sum(1 for values_by_col in user_static_sets.values() if len(values_by_col[col]) > 1)
        for col in STATIC_COLUMNS
    }
    inconsistent_target_users = (
        sum(1 for values in user_target_sets.values() if len(values) > 1) if has_target else None
    )

    user_summary_columns = [
        "event_count",
        "session_count",
        "iap_count",
        "ad_impression_count",
        "active_days",
        "last_event_day",
        "unique_event_names",
        "unique_ad_networks",
        "unique_products",
        "revenue_d0_d7",
        "iap_revenue_d0_d7",
        "ad_revenue_d0_d7",
        "revenue_d0",
        "revenue_d1",
        "revenue_d2_d3",
        "revenue_d4_d7",
        "days_to_first_iap",
        "days_to_first_ad",
    ]
    user_summaries = {col: summarize_series(user_features[col]) for col in user_summary_columns}

    target_summary = None
    if has_target:
        target_summary = summarize_series(user_features["ltv_d8_d180"])
        target_summary["positive_user_rate"] = safe_float((user_features["ltv_d8_d180"] > 0).mean())
        target_summary["positive_revenue_share_top_10pct_by_target"] = safe_float(
            user_features.nlargest(max(1, int(len(user_features) * 0.1)), "ltv_d8_d180")["ltv_d8_d180"].sum()
            / user_features["ltv_d8_d180"].sum()
            if user_features["ltv_d8_d180"].sum() > 0
            else None
        )

    segment_summary = None
    if has_target:
        static_rows = []
        for user_id, values_by_col in user_static_sets.items():
            row = {"user_id": user_id}
            for col in STATIC_COLUMNS:
                values = sorted(str(v) for v in values_by_col[col])
                row[col] = values[0] if values else None
            static_rows.append(row)
        static_df = pd.DataFrame(static_rows)
        merged = user_features[["user_id", "ltv_d8_d180", "revenue_d0_d7", "event_count"]].merge(
            static_df, on="user_id", how="left"
        )
        segment_summary = {}
        for col in ["platform", "country_tier", "channel_tier", "install_week"]:
            seg = (
                merged.groupby(col, dropna=False)
                .agg(
                    users=("user_id", "nunique"),
                    mean_ltv=("ltv_d8_d180", "mean"),
                    median_ltv=("ltv_d8_d180", "median"),
                    positive_rate=("ltv_d8_d180", lambda s: float((s > 0).mean())),
                    mean_d0_d7_revenue=("revenue_d0_d7", "mean"),
                    mean_event_count=("event_count", "mean"),
                )
                .sort_values(["users", "mean_ltv"], ascending=[False, False])
                .head(20)
                .reset_index()
            )
            segment_summary[col] = seg.to_dict(orient="records")

    event_type_by_day_rows = [
        {"day_since_install": int(day), "event_type": event_type, "rows": int(count)}
        for (day, event_type), count in sorted(event_type_by_day.items())
    ]

    return {
        "member": member,
        "row_count": row_count,
        "column_count": len(column_order or []),
        "columns": column_order,
        "user_count": len(user_ids),
        "rows_per_user": summarize_series(pd.Series(event_by_user.values())),
        "null_counts": {col: int(null_counts[col]) for col in (column_order or [])},
        "null_rates": {col: safe_float(null_counts[col] / row_count) for col in (column_order or [])},
        "empty_string_counts": {col: int(empty_string_counts[col]) for col in empty_string_counts},
        "categorical_top_values": {col: top_counter(counter) for col, counter in categorical_counts.items()},
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "validation_checks": {
            "negative_revenue_rows": negative_revenue_rows,
            "session_revenue_non_null_rows": session_revenue_non_null,
            "non_session_revenue_null_rows": non_session_revenue_null,
            "invalid_day_rows": invalid_day_rows,
            "invalid_hour_rows": invalid_hour_rows,
            "invalid_event_type_rows": invalid_event_type_rows,
            "inconsistent_static_users": inconsistent_static,
            "inconsistent_target_users": inconsistent_target_users,
        },
        "rows_by_install_week": {str(int(k)): int(v) for k, v in sorted(rows_by_install_week.items())},
        "users_by_install_week": {str(int(k)): len(v) for k, v in sorted(users_by_install_week.items())},
        "event_type_by_day": event_type_by_day_rows,
        "user_feature_summaries": user_summaries,
        "target_summary": target_summary,
        "segment_summary_top20": segment_summary,
    }


def write_markdown_report(train: dict[str, Any], test: dict[str, Any], output_path: Path) -> None:
    def pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2%}"

    def money(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"${value:,.4f}"

    lines = [
        "# Mobile Game LTV EDA",
        "",
        "## Dataset Grain",
        "",
        "- Raw grain: one row per user event during days 0-7 after install.",
        "- Modeling grain: one row per `user_id`; target is D8-D180 revenue in train only.",
        "- Prediction grain: one row per unique test `user_id`.",
        "",
        "## Shape",
        "",
        "| split | event rows | users | rows/user median | rows/user p95 | rows/user max |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| train | {train['row_count']:,} | {train['user_count']:,} | "
            f"{train['rows_per_user']['p50']:,.0f} | {train['rows_per_user']['p95']:,.0f} | "
            f"{train['rows_per_user']['max']:,.0f} |"
        ),
        (
            f"| test | {test['row_count']:,} | {test['user_count']:,} | "
            f"{test['rows_per_user']['p50']:,.0f} | {test['rows_per_user']['p95']:,.0f} | "
            f"{test['rows_per_user']['max']:,.0f} |"
        ),
        "",
        "## Target Summary",
        "",
    ]
    target = train["target_summary"] or {}
    lines.extend(
        [
            f"- Positive D8-D180 LTV user rate: {pct(target.get('positive_user_rate'))}",
            f"- Mean target: {money(target.get('mean'))}",
            f"- Median target: {money(target.get('p50'))}",
            f"- P95 target: {money(target.get('p95'))}",
            f"- P99 target: {money(target.get('p99'))}",
            f"- Max target: {money(target.get('max'))}",
            f"- Actual top 10% users by target capture: {pct(target.get('positive_revenue_share_top_10pct_by_target'))}",
            "",
            "## First-7-Day Behavior Summary",
            "",
            "| metric | train median | train p95 | test median | test p95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in [
        "event_count",
        "session_count",
        "iap_count",
        "ad_impression_count",
        "active_days",
        "revenue_d0_d7",
        "iap_revenue_d0_d7",
        "ad_revenue_d0_d7",
    ]:
        tr = train["user_feature_summaries"][metric]
        te = test["user_feature_summaries"][metric]
        lines.append(f"| {metric} | {tr.get('p50', 0):,.4f} | {tr.get('p95', 0):,.4f} | {te.get('p50', 0):,.4f} | {te.get('p95', 0):,.4f} |")

    lines.extend(
        [
            "",
            "## Quality Checks",
            "",
            "| check | train | test | interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    checks = [
        ("negative_revenue_rows", "Revenue should not be negative."),
        ("session_revenue_non_null_rows", "Session rows should not carry revenue."),
        ("non_session_revenue_null_rows", "IAP/ad rows should usually carry revenue."),
        ("invalid_day_rows", "Feature window must stay within D0-D7."),
        ("invalid_hour_rows", "Event hour must be 0-23."),
        ("invalid_event_type_rows", "Event type enum should match session/iap/ad_impression."),
    ]
    for check, interp in checks:
        lines.append(
            f"| {check} | {train['validation_checks'][check]:,} | {test['validation_checks'][check]:,} | {interp} |"
        )

    lines.extend(
        [
            "",
            "### Static Field Consistency",
            "",
            "Users should not change platform, country, channel, or install date across their event rows.",
            "",
            "| field | inconsistent train users | inconsistent test users |",
            "|---|---:|---:|",
        ]
    )
    for field in STATIC_COLUMNS:
        lines.append(
            f"| {field} | {train['validation_checks']['inconsistent_static_users'][field]:,} | "
            f"{test['validation_checks']['inconsistent_static_users'][field]:,} |"
        )
    lines.append(f"- Train users with inconsistent repeated target values: {train['validation_checks']['inconsistent_target_users']:,}")

    lines.extend(
        [
            "",
            "## Top Categories",
            "",
            "### Train Event Types",
            "",
        ]
    )
    for item in train["categorical_top_values"]["event_type"]:
        lines.append(f"- {item['value']}: {item['count']:,} rows ({item['share']:.2%})")
    lines.extend(["", "### Test Event Types", ""])
    for item in test["categorical_top_values"]["event_type"]:
        lines.append(f"- {item['value']}: {item['count']:,} rows ({item['share']:.2%})")

    lines.extend(
        [
            "",
            "## Modeling Implications",
            "",
            "- The raw data is event-level and must be aggregated to user-level before training.",
            "- Early revenue is a legitimate feature because it occurs in D0-D7 while the target is D8-D180.",
            "- The target is expected to be zero-inflated and long-tailed, so log1p regression and two-stage modeling should be evaluated.",
            "- Ranking metrics such as top-decile revenue capture matter alongside RMSE/RMSLE because UA decisions care about finding high-value users and segments.",
            "- The dataset has `channel_tier`, not full campaign cost. ROAS or budget recommendations should be framed as channel/platform/country segment simulation unless real CPI is added.",
            "",
            "## Recommended Automated Tests",
            "",
            "- Required columns exist for train/test/submission.",
            "- `day_since_install` is always between 0 and 7.",
            "- `event_hour` is always between 0 and 23.",
            "- `event_type` belongs to session, iap, or ad_impression.",
            "- Static user attributes are constant within each user.",
            "- Train target is constant within each user's repeated event rows.",
            "- Feature builder outputs exactly one row per `user_id`.",
            "- Submission contains exactly the sample submission user ids with non-negative predictions.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = profile_csv_from_zip(zip_path, "train.csv", args.chunksize, has_target=True)
    test = profile_csv_from_zip(zip_path, "test.csv", args.chunksize, has_target=False)

    summary = {"source_zip": str(zip_path), "train": train, "test": test}
    (output_dir / "eda_profile.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(train, test, output_dir / "eda_report.md")


if __name__ == "__main__":
    main()

