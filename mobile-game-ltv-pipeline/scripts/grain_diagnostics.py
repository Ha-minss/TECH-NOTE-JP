from __future__ import annotations

import argparse
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = [
    "user_id",
    "platform",
    "country_tier",
    "channel_tier",
    "install_day",
    "install_week",
    "day_since_install",
    "event_hour",
    "event_type",
    "event_name",
    "product_id",
    "network",
    "ad_placement",
    "revenue_usd",
    "ltv_d8_d180",
]
REQUIRED_TEST_COLUMNS = [col for col in REQUIRED_TRAIN_COLUMNS if col != "ltv_d8_d180"]
CONTEXT_COLUMNS = ["platform", "country_tier", "channel_tier", "install_day", "install_week"]
ALLOWED_EVENT_TYPES = {"session", "ad_impression", "iap"}
GRAIN_SPECS = {
    "A_user_id": ["user_id"],
    "B_user_id_install_day": ["user_id", "install_day"],
    "C_user_context_install_day": ["user_id", "platform", "country_tier", "channel_tier", "install_day"],
    "D_user_context_install_day_week": [
        "user_id",
        "platform",
        "country_tier",
        "channel_tier",
        "install_day",
        "install_week",
    ],
}


def make_key_frame(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = df[columns].astype("string").fillna("<NA>")
    return values.agg("\x1f".join, axis=1)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


@dataclass
class QualityStats:
    split: str
    required_columns: list[str]
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    null_counts: Counter = field(default_factory=Counter)
    invalid_day_rows: int = 0
    invalid_hour_rows: int = 0
    invalid_event_type_rows: int = 0
    negative_revenue_rows: int = 0
    session_revenue_null_rows: int = 0
    session_revenue_non_null_rows: int = 0
    non_session_revenue_null_rows: int = 0
    non_session_revenue_non_null_rows: int = 0
    event_type_counts: Counter = field(default_factory=Counter)

    def observe_columns(self, columns: list[str]) -> None:
        if not self.columns:
            self.columns = columns

    def update(self, df: pd.DataFrame) -> None:
        self.row_count += len(df)
        self.null_counts.update(df.isna().sum().to_dict())
        self.invalid_day_rows += int((~df["day_since_install"].between(0, 7)).sum())
        self.invalid_hour_rows += int((~df["event_hour"].between(0, 23)).sum())
        self.invalid_event_type_rows += int((~df["event_type"].isin(ALLOWED_EVENT_TYPES)).sum())
        self.negative_revenue_rows += int((df["revenue_usd"].fillna(0) < 0).sum())
        is_session = df["event_type"] == "session"
        is_non_session = ~is_session
        self.session_revenue_null_rows += int((is_session & df["revenue_usd"].isna()).sum())
        self.session_revenue_non_null_rows += int((is_session & df["revenue_usd"].notna()).sum())
        self.non_session_revenue_null_rows += int((is_non_session & df["revenue_usd"].isna()).sum())
        self.non_session_revenue_non_null_rows += int((is_non_session & df["revenue_usd"].notna()).sum())
        self.event_type_counts.update(df["event_type"].fillna("<NA>").astype(str).value_counts().to_dict())

    def missing_rows(self) -> list[dict[str, Any]]:
        rows = []
        for col in self.columns:
            count = int(self.null_counts[col])
            rows.append(
                {
                    "split": self.split,
                    "column": col,
                    "null_count": count,
                    "null_rate": rate(count, self.row_count),
                }
            )
        return rows

    def required_columns_result(self) -> dict[str, Any]:
        missing = [col for col in self.required_columns if col not in self.columns]
        return {
            "split": self.split,
            "missing_required_columns": missing,
            "all_required_columns_present": len(missing) == 0,
        }


@dataclass
class GrainAccumulator:
    name: str
    columns: list[str]
    split: str
    has_target: bool
    event_count: Counter = field(default_factory=Counter)
    session_count: Counter = field(default_factory=Counter)
    ad_impression_count: Counter = field(default_factory=Counter)
    iap_count: Counter = field(default_factory=Counter)
    revenue_sum: Counter = field(default_factory=Counter)
    ad_revenue_sum: Counter = field(default_factory=Counter)
    iap_revenue_sum: Counter = field(default_factory=Counter)
    revenue_d0: Counter = field(default_factory=Counter)
    revenue_d1: Counter = field(default_factory=Counter)
    revenue_d2_d3: Counter = field(default_factory=Counter)
    revenue_d4_d7: Counter = field(default_factory=Counter)
    min_day: dict[str, int] = field(default_factory=dict)
    max_day: dict[str, int] = field(default_factory=dict)
    min_hour: dict[str, int] = field(default_factory=dict)
    max_hour: dict[str, int] = field(default_factory=dict)
    active_days: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    active_hours: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    hour_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    networks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    ad_placements: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    products: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    first_iap_day: dict[str, int] = field(default_factory=dict)
    iap_amount_sum: Counter = field(default_factory=Counter)
    max_iap_amount: dict[str, float] = field(default_factory=dict)
    target_values: dict[str, set[float]] = field(default_factory=lambda: defaultdict(set))
    context_values: dict[str, dict[str, set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    def update(self, df: pd.DataFrame) -> None:
        work = df.copy()
        work["_grain_key"] = make_key_frame(work, self.columns)
        keys = work["_grain_key"]

        self.event_count.update(work.groupby("_grain_key").size().to_dict())
        self.session_count.update(work.loc[work["event_type"] == "session"].groupby("_grain_key").size().to_dict())
        self.ad_impression_count.update(
            work.loc[work["event_type"] == "ad_impression"].groupby("_grain_key").size().to_dict()
        )
        self.iap_count.update(work.loc[work["event_type"] == "iap"].groupby("_grain_key").size().to_dict())

        revenue_rows = work.loc[work["revenue_usd"].notna()].copy()
        if not revenue_rows.empty:
            self.revenue_sum.update(revenue_rows.groupby("_grain_key")["revenue_usd"].sum().to_dict())
            self.ad_revenue_sum.update(
                revenue_rows.loc[revenue_rows["event_type"] == "ad_impression"]
                .groupby("_grain_key")["revenue_usd"]
                .sum()
                .to_dict()
            )
            iap_revenue = revenue_rows.loc[revenue_rows["event_type"] == "iap"]
            self.iap_revenue_sum.update(iap_revenue.groupby("_grain_key")["revenue_usd"].sum().to_dict())
            self.iap_amount_sum.update(iap_revenue.groupby("_grain_key")["revenue_usd"].sum().to_dict())
            for key, value in iap_revenue.groupby("_grain_key")["revenue_usd"].max().items():
                self.max_iap_amount[key] = max(float(value), self.max_iap_amount.get(key, -math.inf))

            for bucket_name, mask in {
                "revenue_d0": revenue_rows["day_since_install"] == 0,
                "revenue_d1": revenue_rows["day_since_install"] == 1,
                "revenue_d2_d3": revenue_rows["day_since_install"].between(2, 3),
                "revenue_d4_d7": revenue_rows["day_since_install"].between(4, 7),
            }.items():
                counter = getattr(self, bucket_name)
                counter.update(revenue_rows.loc[mask].groupby("_grain_key")["revenue_usd"].sum().to_dict())

        min_day = work.groupby("_grain_key")["day_since_install"].min()
        max_day = work.groupby("_grain_key")["day_since_install"].max()
        min_hour = work.groupby("_grain_key")["event_hour"].min()
        max_hour = work.groupby("_grain_key")["event_hour"].max()
        for key, value in min_day.items():
            self.min_day[key] = min(int(value), self.min_day.get(key, int(value)))
        for key, value in max_day.items():
            self.max_day[key] = max(int(value), self.max_day.get(key, int(value)))
        for key, value in min_hour.items():
            self.min_hour[key] = min(int(value), self.min_hour.get(key, int(value)))
        for key, value in max_hour.items():
            self.max_hour[key] = max(int(value), self.max_hour.get(key, int(value)))

        for key, day in work[["_grain_key", "day_since_install"]].drop_duplicates().itertuples(index=False, name=None):
            self.active_days[key].add(int(day))
        for key, hour in work[["_grain_key", "event_hour"]].drop_duplicates().itertuples(index=False, name=None):
            self.active_hours[key].add(int(hour))
        hour_table = work.groupby(["_grain_key", "event_hour"]).size()
        for (key, hour), count in hour_table.items():
            self.hour_counts[key][int(hour)] += int(count)

        for key, value in work.loc[work["network"].notna(), ["_grain_key", "network"]].drop_duplicates().itertuples(index=False, name=None):
            self.networks[key].add(str(value))
        for key, value in work.loc[work["ad_placement"].notna(), ["_grain_key", "ad_placement"]].drop_duplicates().itertuples(index=False, name=None):
            self.ad_placements[key].add(str(value))
        for key, value in work.loc[work["product_id"].notna(), ["_grain_key", "product_id"]].drop_duplicates().itertuples(index=False, name=None):
            self.products[key].add(str(value))

        iap_days = work.loc[work["event_type"] == "iap"].groupby("_grain_key")["day_since_install"].min()
        for key, value in iap_days.items():
            self.first_iap_day[key] = min(int(value), self.first_iap_day.get(key, int(value)))

        if self.has_target:
            for key, value in work[["_grain_key", "ltv_d8_d180"]].drop_duplicates().itertuples(index=False, name=None):
                self.target_values[key].add(round(float(value), 10))

        for col in CONTEXT_COLUMNS:
            for key, value in work[["_grain_key", col]].drop_duplicates().itertuples(index=False, name=None):
                self.context_values[key][col].add(str(value))

    def feature_frame(self) -> pd.DataFrame:
        keys = sorted(self.event_count)
        rows = []
        for key in keys:
            event_count = int(self.event_count[key])
            sessions = int(self.session_count[key])
            ads = int(self.ad_impression_count[key])
            iaps = int(self.iap_count[key])
            ad_revenue = float(self.ad_revenue_sum[key])
            iap_revenue = float(self.iap_revenue_sum[key])
            revenue = float(self.revenue_sum[key])
            most_active_hour = None
            if self.hour_counts[key]:
                most_active_hour = self.hour_counts[key].most_common(1)[0][0]
            avg_iap_amount = float(self.iap_amount_sum[key] / iaps) if iaps else np.nan
            max_iap_amount = self.max_iap_amount.get(key, np.nan)
            row = {
                "grain_key": key,
                "event_count": event_count,
                "session_count": sessions,
                "ad_impression_count": ads,
                "iap_count": iaps,
                "active_days": len(self.active_days[key]),
                "last_event_day": self.max_day.get(key, np.nan),
                "revenue_d0_d7": revenue,
                "ad_revenue_d0_d7": ad_revenue,
                "iap_revenue_d0_d7": iap_revenue,
                "revenue_d0": float(self.revenue_d0[key]),
                "revenue_d1": float(self.revenue_d1[key]),
                "revenue_d2_d3": float(self.revenue_d2_d3[key]),
                "revenue_d4_d7": float(self.revenue_d4_d7[key]),
                "unique_network_count": len(self.networks[key]),
                "unique_ad_placement_count": len(self.ad_placements[key]),
                "ads_per_session": float(ads / sessions) if sessions else np.nan,
                "ad_revenue_per_ad": float(ad_revenue / ads) if ads else np.nan,
                "ad_revenue_per_session": float(ad_revenue / sessions) if sessions else np.nan,
                "early_payer_flag": int(iaps > 0),
                "days_to_first_iap": self.first_iap_day.get(key, np.nan),
                "unique_product_count": len(self.products[key]),
                "avg_iap_amount": avg_iap_amount,
                "max_iap_amount": max_iap_amount,
                "first_event_hour": self.min_hour.get(key, np.nan),
                "last_event_hour": self.max_hour.get(key, np.nan),
                "most_active_hour": most_active_hour,
                "active_hour_count": len(self.active_hours[key]),
            }
            for col in CONTEXT_COLUMNS:
                values = self.context_values[key][col]
                row[col] = next(iter(values)) if len(values) == 1 else np.nan
                row[f"{col}_unique_values"] = len(values)
            if self.has_target:
                targets = self.target_values[key]
                row["target_unique_values"] = len(targets)
                row["target_value"] = next(iter(targets)) if len(targets) == 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def summarize_grain(split: str, grain_name: str, grain_cols: list[str], frame: pd.DataFrame, has_target: bool) -> dict[str, Any]:
    group_count = len(frame)
    target_collision_groups = int((frame["target_unique_values"] > 1).sum()) if has_target else np.nan
    return {
        "split": split,
        "grain": grain_name,
        "grain_columns": " + ".join(grain_cols),
        "group_count": group_count,
        "target_collision_groups": target_collision_groups,
        "target_collision_rate": rate(target_collision_groups, group_count) if has_target else np.nan,
        "event_count_p25": quantile(frame["event_count"].tolist(), 0.25),
        "event_count_median": quantile(frame["event_count"].tolist(), 0.50),
        "event_count_p75": quantile(frame["event_count"].tolist(), 0.75),
        "event_count_p95": quantile(frame["event_count"].tolist(), 0.95),
        "session_count_median": quantile(frame["session_count"].tolist(), 0.50),
        "session_count_p75": quantile(frame["session_count"].tolist(), 0.75),
        "session_count_p95": quantile(frame["session_count"].tolist(), 0.95),
        "ad_impression_count_median": quantile(frame["ad_impression_count"].tolist(), 0.50),
        "ad_impression_count_p75": quantile(frame["ad_impression_count"].tolist(), 0.75),
        "ad_impression_count_p95": quantile(frame["ad_impression_count"].tolist(), 0.95),
        "iap_count_median": quantile(frame["iap_count"].tolist(), 0.50),
        "iap_count_p75": quantile(frame["iap_count"].tolist(), 0.75),
        "iap_count_p95": quantile(frame["iap_count"].tolist(), 0.95),
        "active_days_median": quantile(frame["active_days"].tolist(), 0.50),
        "active_days_p75": quantile(frame["active_days"].tolist(), 0.75),
        "active_days_p95": quantile(frame["active_days"].tolist(), 0.95),
        "revenue_d0_d7_median": quantile(frame["revenue_d0_d7"].tolist(), 0.50),
        "revenue_d0_d7_p75": quantile(frame["revenue_d0_d7"].tolist(), 0.75),
        "revenue_d0_d7_p95": quantile(frame["revenue_d0_d7"].tolist(), 0.95),
        "zero_revenue_group_rate": rate(int((frame["revenue_d0_d7"] == 0).sum()), group_count),
        "event_count_1_2_group_rate": rate(int((frame["event_count"] <= 2).sum()), group_count),
    }


def feature_availability(split: str, grain_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    feature_groups = {
        "context": CONTEXT_COLUMNS,
        "activity": ["event_count", "session_count", "ad_impression_count", "iap_count", "active_days", "last_event_day"],
        "revenue": [
            "revenue_d0_d7",
            "ad_revenue_d0_d7",
            "iap_revenue_d0_d7",
            "revenue_d0",
            "revenue_d1",
            "revenue_d2_d3",
            "revenue_d4_d7",
        ],
        "ad": [
            "unique_network_count",
            "unique_ad_placement_count",
            "ads_per_session",
            "ad_revenue_per_ad",
            "ad_revenue_per_session",
        ],
        "iap": ["early_payer_flag", "days_to_first_iap", "unique_product_count", "avg_iap_amount", "max_iap_amount"],
        "time": ["first_event_hour", "last_event_hour", "most_active_hour", "active_hour_count"],
    }
    rows = []
    n = len(frame)
    for family, features in feature_groups.items():
        for feature in features:
            series = frame[feature]
            non_null = int(series.notna().sum())
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                non_zero = int((numeric.fillna(0) != 0).sum())
                median = float(numeric.median()) if numeric.notna().any() else np.nan
                p75 = float(numeric.quantile(0.75)) if numeric.notna().any() else np.nan
                p95 = float(numeric.quantile(0.95)) if numeric.notna().any() else np.nan
            else:
                non_zero = non_null
                median = p75 = p95 = np.nan
            rows.append(
                {
                    "split": split,
                    "grain": grain_name,
                    "feature_family": family,
                    "feature": feature,
                    "non_null_rate": rate(non_null, n),
                    "non_zero_rate": rate(non_zero, n),
                    "median": median,
                    "p75": p75,
                    "p95": p95,
                }
            )
        if family == "context":
            for feature in CONTEXT_COLUMNS:
                unique_col = f"{feature}_unique_values"
                rows.append(
                    {
                        "split": split,
                        "grain": grain_name,
                        "feature_family": "context_consistency",
                        "feature": f"{feature}_constant_within_group",
                        "non_null_rate": rate(int((frame[unique_col] == 1).sum()), n),
                        "non_zero_rate": rate(int((frame[unique_col] == 1).sum()), n),
                        "median": float(frame[unique_col].median()),
                        "p75": float(frame[unique_col].quantile(0.75)),
                        "p95": float(frame[unique_col].quantile(0.95)),
                    }
                )
    return rows


def process_split(zip_path: Path, member: str, split: str, chunksize: int, has_target: bool) -> tuple[QualityStats, dict[str, pd.DataFrame]]:
    required = REQUIRED_TRAIN_COLUMNS if has_target else REQUIRED_TEST_COLUMNS
    quality = QualityStats(split=split, required_columns=required)
    accumulators = {
        name: GrainAccumulator(name=name, columns=cols, split=split, has_target=has_target)
        for name, cols in GRAIN_SPECS.items()
    }
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            reader = pd.read_csv(fh, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                quality.observe_columns(list(chunk.columns))
                quality.update(chunk)
                for acc in accumulators.values():
                    acc.update(chunk)
    frames = {name: acc.feature_frame() for name, acc in accumulators.items()}
    return quality, frames


def write_missing_quality_report(path: Path, train_quality: QualityStats, test_quality: QualityStats, missing_df: pd.DataFrame) -> None:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    lines = [
        "# Missingness and Raw Quality Report",
        "",
        "## Required Columns",
        "",
    ]
    for result in [train_quality.required_columns_result(), test_quality.required_columns_result()]:
        if result["all_required_columns_present"]:
            lines.append(f"- {result['split']}: all required columns are present.")
        else:
            lines.append(f"- {result['split']}: missing columns: {', '.join(result['missing_required_columns'])}")

    lines.extend(
        [
            "",
            "## Core Quality Checks",
            "",
            "| check | train | test | interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    rows = [
        ("invalid_day_rows", train_quality.invalid_day_rows, test_quality.invalid_day_rows, "`day_since_install` should stay within D0-D7."),
        ("invalid_hour_rows", train_quality.invalid_hour_rows, test_quality.invalid_hour_rows, "`event_hour` should stay within 0-23."),
        (
            "invalid_event_type_rows",
            train_quality.invalid_event_type_rows,
            test_quality.invalid_event_type_rows,
            "`event_type` should be session/ad_impression/iap.",
        ),
        ("negative_revenue_rows", train_quality.negative_revenue_rows, test_quality.negative_revenue_rows, "`revenue_usd` should not be negative."),
        (
            "session_revenue_non_null_rows",
            train_quality.session_revenue_non_null_rows,
            test_quality.session_revenue_non_null_rows,
            "Session rows should normally have missing revenue and can be filled with 0 for aggregation.",
        ),
        (
            "non_session_revenue_null_rows",
            train_quality.non_session_revenue_null_rows,
            test_quality.non_session_revenue_null_rows,
            "Ad/IAP rows need revenue for early monetization features.",
        ),
    ]
    for name, train_value, test_value, interpretation in rows:
        lines.append(f"| {name} | {train_value:,} | {test_value:,} | {interpretation} |")

    lines.extend(["", "## Null Rates", "", "| split | column | null count | null rate |", "|---|---|---:|---:|"])
    for row in missing_df.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['column']} | {int(row['null_count']):,} | {pct(float(row['null_rate']))} |"
        )

    def col_null(split: str, column: str) -> float:
        row = missing_df[(missing_df["split"] == split) & (missing_df["column"] == column)]
        return float(row["null_rate"].iloc[0]) if not row.empty else 0.0

    lines.extend(
        [
            "",
            "## Missingness Interpretation",
            "",
            f"- `product_id`: train null rate {pct(col_null('train', 'product_id'))}, test null rate {pct(col_null('test', 'product_id'))}. This is expected because only IAP rows carry product ids; use derived IAP features rather than raw `product_id` as a broad model feature.",
            f"- `network`: train null rate {pct(col_null('train', 'network'))}, test null rate {pct(col_null('test', 'network'))}. Missing values are mostly non-ad rows; use a missing category only if encoding row-level events, or aggregate unique network counts at the grain level.",
            f"- `ad_placement`: train null rate {pct(col_null('train', 'ad_placement'))}, test null rate {pct(col_null('test', 'ad_placement'))}. Missing values are mostly non-ad rows; aggregate placement diversity or fill missing with `no_ad_placement`.",
            f"- `revenue_usd`: train null rate {pct(col_null('train', 'revenue_usd'))}, test null rate {pct(col_null('test', 'revenue_usd'))}. Session revenue is missing by design and should be treated as 0 when aggregating D0-D7 revenue.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grain_report(
    path: Path,
    grain_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    sample_submission_rows: int,
    test_user_count: int,
) -> None:
    def pct(value: float) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.2%}"

    lines = [
        "# Modeling Grain Diagnostics",
        "",
        "## Scope",
        "",
        "This report validates modeling grain and feature aggregation feasibility only. No model training was performed.",
        "",
        "## Grain Comparison",
        "",
        "| split | grain | groups | target collision groups | collision rate | event median | event p95 | revenue median | zero revenue rate | event_count <= 2 rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grain_df.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['grain']} | {int(row['group_count']):,} | "
            f"{'' if pd.isna(row['target_collision_groups']) else f'{int(row['target_collision_groups']):,}'} | "
            f"{pct(row['target_collision_rate'])} | {row['event_count_median']:.0f} | {row['event_count_p95']:.0f} | "
            f"{row['revenue_d0_d7_median']:.4f} | {pct(row['zero_revenue_group_rate'])} | {pct(row['event_count_1_2_group_rate'])} |"
        )

    train_rows = grain_df[grain_df["split"] == "train"].set_index("grain")
    test_rows = grain_df[grain_df["split"] == "test"].set_index("grain")
    user_collision = int(train_rows.loc["A_user_id", "target_collision_groups"])
    user_collision_rate = float(train_rows.loc["A_user_id", "target_collision_rate"])
    c_collision = int(train_rows.loc["C_user_context_install_day", "target_collision_groups"])
    c_collision_rate = float(train_rows.loc["C_user_context_install_day", "target_collision_rate"])
    c_test_groups = int(test_rows.loc["C_user_context_install_day", "group_count"])
    d_test_groups = int(test_rows.loc["D_user_context_install_day_week", "group_count"])

    c_train_sparse = float(train_rows.loc["C_user_context_install_day", "event_count_1_2_group_rate"])
    c_test_sparse = float(test_rows.loc["C_user_context_install_day", "event_count_1_2_group_rate"])

    lines.extend(
        [
            "",
            "## Answers",
            "",
            "### Can `user_id` alone be used?",
            "",
            f"No. Under `user_id`, train has {user_collision:,} groups with multiple `ltv_d8_d180` values ({pct(user_collision_rate)} of groups). Static context fields also vary within many `user_id`s. A plain `groupby(user_id)` would mix different installs or contexts and create ambiguous labels.",
            "",
            "### How much does a composite key reduce target collisions?",
            "",
            f"The strongest candidate, `user_id + platform + country_tier + channel_tier + install_day`, reduces train target-collision groups to {c_collision:,} ({pct(c_collision_rate)}). That is a large reduction from `user_id` alone, though not a perfect fix.",
            "",
            "### Does the composite key make features too sparse?",
            "",
            f"No, not fatally. Under candidate C, train event_count<=2 rate is {pct(c_train_sparse)} and test event_count<=2 rate is {pct(c_test_sparse)}. The median event count is still usable, but sparse groups should be monitored and may benefit from segment priors or fallback features.",
            "",
            "### Most defensible modeling grain",
            "",
            "`user_id + platform + country_tier + channel_tier + install_day` is the most defensible grain at this stage. Candidate D adds `install_week`, but `install_week = install_day // 7`, so it is redundant with `install_day` and should be treated as a reference check rather than a necessary key component.",
            "",
            "### What to do with remaining target collisions?",
            "",
            "Recommended handling: exclude remaining collision groups from the first supervised training run and log them as data-quality exceptions. Averaging target values would hide an unresolved identity issue; it can be used only as a sensitivity analysis after a clean baseline is established.",
            "",
            "### Does Kaggle submission grain match test?",
            "",
            f"Not cleanly. `sample_submission.csv` has {sample_submission_rows:,} rows. Test has {test_user_count:,} unique `user_id`s, candidate C has {c_test_groups:,} groups, and candidate D has {d_test_groups:,} groups. Candidate C/D almost match sample row count but differ by {abs(sample_submission_rows - c_test_groups):,} rows. Before a real competition submission, the dataset version or submission key definition should be verified.",
            "",
            "## Feature Aggregation Feasibility",
            "",
            "The requested feature families are feasible at candidate C. Context features are constant by construction, activity/revenue/time features are populated for every group, ad features are mostly usable, and IAP features are intentionally sparse because IAP events are rare.",
            "",
            "Feature availability details are saved in `data/processed/feature_availability_by_grain.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    processed_dir = output_root / "data" / "processed"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_quality, train_frames = process_split(zip_path, "train.csv", "train", args.chunksize, True)
    test_quality, test_frames = process_split(zip_path, "test.csv", "test", args.chunksize, False)

    grain_rows = []
    availability_rows = []
    for split, frames, has_target in [("train", train_frames, True), ("test", test_frames, False)]:
        for grain_name, frame in frames.items():
            grain_rows.append(summarize_grain(split, grain_name, GRAIN_SPECS[grain_name], frame, has_target))
            availability_rows.extend(feature_availability(split, grain_name, frame))

    grain_df = pd.DataFrame(grain_rows)
    availability_df = pd.DataFrame(availability_rows)
    missing_df = pd.DataFrame(train_quality.missing_rows() + test_quality.missing_rows())

    grain_df.to_csv(processed_dir / "grain_comparison.csv", index=False)
    availability_df.to_csv(processed_dir / "feature_availability_by_grain.csv", index=False)

    with zipfile.ZipFile(zip_path) as zf:
        sample_submission = pd.read_csv(zf.open("sample_submission.csv"), usecols=["user_id"])
    sample_submission_rows = int(sample_submission.shape[0])
    test_user_count = int(test_frames["A_user_id"].shape[0])

    write_missing_quality_report(reports_dir / "missing_quality_report.md", train_quality, test_quality, missing_df)
    write_grain_report(
        reports_dir / "grain_diagnostics.md",
        grain_df,
        availability_df,
        sample_submission_rows,
        test_user_count,
    )


if __name__ == "__main__":
    main()
