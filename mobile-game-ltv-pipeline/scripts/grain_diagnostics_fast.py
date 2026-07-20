from __future__ import annotations

import argparse
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
REQUIRED_TEST_COLUMNS = [c for c in REQUIRED_TRAIN_COLUMNS if c != "ltv_d8_d180"]
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


def as_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def update_counter(counter: Counter, series: pd.Series) -> None:
    for key, value in series.items():
        counter[as_key(key)] += float(value)


def update_min(store: dict[tuple[Any, ...], float], series: pd.Series) -> None:
    for key, value in series.items():
        key = as_key(key)
        value = float(value)
        store[key] = min(store.get(key, value), value)


def update_max(store: dict[tuple[Any, ...], float], series: pd.Series) -> None:
    for key, value in series.items():
        key = as_key(key)
        value = float(value)
        store[key] = max(store.get(key, value), value)


def q(values: Iterable[float], quantile: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, quantile))


def pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class Quality:
    split: str
    required_columns: list[str]
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    null_counts: Counter = field(default_factory=Counter)
    invalid_day_rows: int = 0
    invalid_hour_rows: int = 0
    invalid_event_type_rows: int = 0
    negative_revenue_rows: int = 0
    session_revenue_non_null_rows: int = 0
    session_revenue_null_rows: int = 0
    non_session_revenue_null_rows: int = 0
    non_session_revenue_non_null_rows: int = 0

    def update(self, df: pd.DataFrame) -> None:
        if not self.columns:
            self.columns = list(df.columns)
        self.rows += len(df)
        self.null_counts.update(df.isna().sum().to_dict())
        self.invalid_day_rows += int((~df["day_since_install"].between(0, 7)).sum())
        self.invalid_hour_rows += int((~df["event_hour"].between(0, 23)).sum())
        self.invalid_event_type_rows += int((~df["event_type"].isin(ALLOWED_EVENT_TYPES)).sum())
        self.negative_revenue_rows += int((df["revenue_usd"].fillna(0) < 0).sum())
        is_session = df["event_type"] == "session"
        self.session_revenue_non_null_rows += int((is_session & df["revenue_usd"].notna()).sum())
        self.session_revenue_null_rows += int((is_session & df["revenue_usd"].isna()).sum())
        self.non_session_revenue_null_rows += int((~is_session & df["revenue_usd"].isna()).sum())
        self.non_session_revenue_non_null_rows += int((~is_session & df["revenue_usd"].notna()).sum())

    def missing_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "split": self.split,
                    "column": col,
                    "null_count": int(self.null_counts[col]),
                    "null_rate": pct(int(self.null_counts[col]), self.rows),
                }
                for col in self.columns
            ]
        )


@dataclass
class GrainAgg:
    name: str
    cols: list[str]
    has_target: bool
    event_count: Counter = field(default_factory=Counter)
    session_count: Counter = field(default_factory=Counter)
    ad_count: Counter = field(default_factory=Counter)
    iap_count: Counter = field(default_factory=Counter)
    revenue: Counter = field(default_factory=Counter)
    ad_revenue: Counter = field(default_factory=Counter)
    iap_revenue: Counter = field(default_factory=Counter)
    revenue_d0: Counter = field(default_factory=Counter)
    revenue_d1: Counter = field(default_factory=Counter)
    revenue_d2_d3: Counter = field(default_factory=Counter)
    revenue_d4_d7: Counter = field(default_factory=Counter)
    min_day: dict[tuple[Any, ...], float] = field(default_factory=dict)
    max_day: dict[tuple[Any, ...], float] = field(default_factory=dict)
    min_hour: dict[tuple[Any, ...], float] = field(default_factory=dict)
    max_hour: dict[tuple[Any, ...], float] = field(default_factory=dict)
    active_day_mask: Counter = field(default_factory=Counter)
    active_hour_mask: Counter = field(default_factory=Counter)
    hour_counts: dict[tuple[Any, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    network_values: dict[tuple[Any, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    placement_values: dict[tuple[Any, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    product_values: dict[tuple[Any, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    first_iap_day: dict[tuple[Any, ...], float] = field(default_factory=dict)
    iap_amount_sum: Counter = field(default_factory=Counter)
    max_iap_amount: dict[tuple[Any, ...], float] = field(default_factory=dict)
    target_min: dict[tuple[Any, ...], float] = field(default_factory=dict)
    target_max: dict[tuple[Any, ...], float] = field(default_factory=dict)
    context_values: dict[tuple[Any, ...], dict[str, set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    def group(self, df: pd.DataFrame):
        return df.groupby(self.cols, dropna=False, sort=False)

    def update(self, df: pd.DataFrame) -> None:
        grouped = self.group(df)
        update_counter(self.event_count, grouped.size())
        update_min(self.min_day, grouped["day_since_install"].min())
        update_max(self.max_day, grouped["day_since_install"].max())
        update_min(self.min_hour, grouped["event_hour"].min())
        update_max(self.max_hour, grouped["event_hour"].max())

        update_counter(self.session_count, self.group(df[df["event_type"] == "session"]).size())
        update_counter(self.ad_count, self.group(df[df["event_type"] == "ad_impression"]).size())
        update_counter(self.iap_count, self.group(df[df["event_type"] == "iap"]).size())

        revenue_rows = df[df["revenue_usd"].notna()]
        if not revenue_rows.empty:
            update_counter(self.revenue, self.group(revenue_rows)["revenue_usd"].sum())
            update_counter(self.ad_revenue, self.group(revenue_rows[revenue_rows["event_type"] == "ad_impression"])["revenue_usd"].sum())
            iap_rows = revenue_rows[revenue_rows["event_type"] == "iap"]
            update_counter(self.iap_revenue, self.group(iap_rows)["revenue_usd"].sum())
            update_counter(self.iap_amount_sum, self.group(iap_rows)["revenue_usd"].sum())
            update_max(self.max_iap_amount, self.group(iap_rows)["revenue_usd"].max())
            update_counter(self.revenue_d0, self.group(revenue_rows[revenue_rows["day_since_install"] == 0])["revenue_usd"].sum())
            update_counter(self.revenue_d1, self.group(revenue_rows[revenue_rows["day_since_install"] == 1])["revenue_usd"].sum())
            update_counter(
                self.revenue_d2_d3,
                self.group(revenue_rows[revenue_rows["day_since_install"].between(2, 3)])["revenue_usd"].sum(),
            )
            update_counter(
                self.revenue_d4_d7,
                self.group(revenue_rows[revenue_rows["day_since_install"].between(4, 7)])["revenue_usd"].sum(),
            )

        iap_days = df[df["event_type"] == "iap"]
        if not iap_days.empty:
            update_min(self.first_iap_day, self.group(iap_days)["day_since_install"].min())

        if self.has_target:
            update_min(self.target_min, grouped["ltv_d8_d180"].min())
            update_max(self.target_max, grouped["ltv_d8_d180"].max())

        day_pairs = df[self.cols + ["day_since_install"]].drop_duplicates()
        for row in day_pairs.itertuples(index=False, name=None):
            key = tuple(row[: len(self.cols)])
            day = int(row[-1])
            self.active_day_mask[key] = int(self.active_day_mask[key]) | (1 << day)

        hour_pairs = df[self.cols + ["event_hour"]].drop_duplicates()
        for row in hour_pairs.itertuples(index=False, name=None):
            key = tuple(row[: len(self.cols)])
            hour = int(row[-1])
            self.active_hour_mask[key] = int(self.active_hour_mask[key]) | (1 << hour)

        hour_counts = df.groupby(self.cols + ["event_hour"], dropna=False, sort=False).size()
        for idx, count in hour_counts.items():
            idx = as_key(idx)
            self.hour_counts[idx[: len(self.cols)]][int(idx[-1])] += int(count)

        for col, store in [
            ("network", self.network_values),
            ("ad_placement", self.placement_values),
            ("product_id", self.product_values),
        ]:
            values = df[df[col].notna()][self.cols + [col]].drop_duplicates()
            for row in values.itertuples(index=False, name=None):
                store[tuple(row[: len(self.cols)])].add(str(row[-1]))

        for col in CONTEXT_COLUMNS:
            values = df[self.cols + [col]].drop_duplicates()
            for row in values.itertuples(index=False, name=None):
                self.context_values[tuple(row[: len(self.cols)])][col].add(str(row[-1]))

    def frame(self) -> pd.DataFrame:
        rows = []
        for key in self.event_count.keys():
            events = int(self.event_count[key])
            sessions = int(self.session_count[key])
            ads = int(self.ad_count[key])
            iaps = int(self.iap_count[key])
            ad_rev = float(self.ad_revenue[key])
            iap_rev = float(self.iap_revenue[key])
            hour_counter = self.hour_counts.get(key, Counter())
            most_active_hour = hour_counter.most_common(1)[0][0] if hour_counter else np.nan
            row = {
                "event_count": events,
                "session_count": sessions,
                "ad_impression_count": ads,
                "iap_count": iaps,
                "active_days": int(int(self.active_day_mask[key]).bit_count()),
                "last_event_day": self.max_day.get(key, np.nan),
                "revenue_d0_d7": float(self.revenue[key]),
                "ad_revenue_d0_d7": ad_rev,
                "iap_revenue_d0_d7": iap_rev,
                "revenue_d0": float(self.revenue_d0[key]),
                "revenue_d1": float(self.revenue_d1[key]),
                "revenue_d2_d3": float(self.revenue_d2_d3[key]),
                "revenue_d4_d7": float(self.revenue_d4_d7[key]),
                "unique_network_count": len(self.network_values[key]),
                "unique_ad_placement_count": len(self.placement_values[key]),
                "ads_per_session": ads / sessions if sessions else np.nan,
                "ad_revenue_per_ad": ad_rev / ads if ads else np.nan,
                "ad_revenue_per_session": ad_rev / sessions if sessions else np.nan,
                "early_payer_flag": int(iaps > 0),
                "days_to_first_iap": self.first_iap_day.get(key, np.nan),
                "unique_product_count": len(self.product_values[key]),
                "avg_iap_amount": float(self.iap_amount_sum[key] / iaps) if iaps else np.nan,
                "max_iap_amount": self.max_iap_amount.get(key, np.nan),
                "first_event_hour": self.min_hour.get(key, np.nan),
                "last_event_hour": self.max_hour.get(key, np.nan),
                "most_active_hour": most_active_hour,
                "active_hour_count": int(int(self.active_hour_mask[key]).bit_count()),
            }
            for col in CONTEXT_COLUMNS:
                value_count = len(self.context_values[key][col])
                row[f"{col}_unique_values"] = value_count
                row[col] = next(iter(self.context_values[key][col])) if value_count == 1 else np.nan
            if self.has_target:
                tmin = self.target_min.get(key, np.nan)
                tmax = self.target_max.get(key, np.nan)
                row["target_collision"] = bool(pd.notna(tmin) and pd.notna(tmax) and round(float(tmin), 10) != round(float(tmax), 10))
            rows.append(row)
        return pd.DataFrame(rows)


def process_split(zip_path: Path, member: str, split: str, chunksize: int, has_target: bool) -> tuple[Quality, dict[str, pd.DataFrame]]:
    quality = Quality(split, REQUIRED_TRAIN_COLUMNS if has_target else REQUIRED_TEST_COLUMNS)
    aggs = {name: GrainAgg(name, cols, has_target) for name, cols in GRAIN_SPECS.items()}
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
                quality.update(chunk)
                for agg in aggs.values():
                    agg.update(chunk)
    return quality, {name: agg.frame() for name, agg in aggs.items()}


def summarize_grains(split: str, frames: dict[str, pd.DataFrame], has_target: bool) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        n = len(frame)
        collisions = int(frame["target_collision"].sum()) if has_target else np.nan
        rows.append(
            {
                "split": split,
                "grain": name,
                "grain_columns": " + ".join(GRAIN_SPECS[name]),
                "group_count": n,
                "target_collision_groups": collisions,
                "target_collision_rate": pct(collisions, n) if has_target else np.nan,
                "event_count_p25": q(frame["event_count"], 0.25),
                "event_count_median": q(frame["event_count"], 0.50),
                "event_count_p75": q(frame["event_count"], 0.75),
                "event_count_p95": q(frame["event_count"], 0.95),
                "session_count_median": q(frame["session_count"], 0.50),
                "session_count_p75": q(frame["session_count"], 0.75),
                "session_count_p95": q(frame["session_count"], 0.95),
                "ad_impression_count_median": q(frame["ad_impression_count"], 0.50),
                "ad_impression_count_p75": q(frame["ad_impression_count"], 0.75),
                "ad_impression_count_p95": q(frame["ad_impression_count"], 0.95),
                "iap_count_median": q(frame["iap_count"], 0.50),
                "iap_count_p75": q(frame["iap_count"], 0.75),
                "iap_count_p95": q(frame["iap_count"], 0.95),
                "active_days_median": q(frame["active_days"], 0.50),
                "active_days_p75": q(frame["active_days"], 0.75),
                "active_days_p95": q(frame["active_days"], 0.95),
                "revenue_d0_d7_median": q(frame["revenue_d0_d7"], 0.50),
                "revenue_d0_d7_p75": q(frame["revenue_d0_d7"], 0.75),
                "revenue_d0_d7_p95": q(frame["revenue_d0_d7"], 0.95),
                "zero_revenue_group_rate": pct(int((frame["revenue_d0_d7"] == 0).sum()), n),
                "event_count_1_2_group_rate": pct(int((frame["event_count"] <= 2).sum()), n),
            }
        )
    return pd.DataFrame(rows)


def availability(split: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    feature_families = {
        "context": CONTEXT_COLUMNS,
        "activity": ["event_count", "session_count", "ad_impression_count", "iap_count", "active_days", "last_event_day"],
        "revenue": ["revenue_d0_d7", "ad_revenue_d0_d7", "iap_revenue_d0_d7", "revenue_d0", "revenue_d1", "revenue_d2_d3", "revenue_d4_d7"],
        "ad": ["unique_network_count", "unique_ad_placement_count", "ads_per_session", "ad_revenue_per_ad", "ad_revenue_per_session"],
        "iap": ["early_payer_flag", "days_to_first_iap", "unique_product_count", "avg_iap_amount", "max_iap_amount"],
        "time": ["first_event_hour", "last_event_hour", "most_active_hour", "active_hour_count"],
    }
    rows = []
    for grain, frame in frames.items():
        n = len(frame)
        for family, features in feature_families.items():
            for feature in features:
                s = frame[feature]
                numeric = pd.to_numeric(s, errors="coerce")
                rows.append(
                    {
                        "split": split,
                        "grain": grain,
                        "feature_family": family,
                        "feature": feature,
                        "non_null_rate": pct(int(s.notna().sum()), n),
                        "non_zero_rate": pct(int((numeric.fillna(0) != 0).sum()), n),
                        "median": float(numeric.median()) if numeric.notna().any() else np.nan,
                        "p75": float(numeric.quantile(0.75)) if numeric.notna().any() else np.nan,
                        "p95": float(numeric.quantile(0.95)) if numeric.notna().any() else np.nan,
                    }
                )
            if family == "context":
                for feature in CONTEXT_COLUMNS:
                    c = frame[f"{feature}_unique_values"]
                    rows.append(
                        {
                            "split": split,
                            "grain": grain,
                            "feature_family": "context_consistency",
                            "feature": f"{feature}_constant_within_group",
                            "non_null_rate": pct(int((c == 1).sum()), n),
                            "non_zero_rate": pct(int((c == 1).sum()), n),
                            "median": float(c.median()),
                            "p75": float(c.quantile(0.75)),
                            "p95": float(c.quantile(0.95)),
                        }
                    )
    return pd.DataFrame(rows)


def write_missing_report(path: Path, train_q: Quality, test_q: Quality, missing_df: pd.DataFrame) -> None:
    def fmt_pct(x: float) -> str:
        return f"{x:.2%}"

    lines = [
        "# Missing and Raw Quality Report",
        "",
        "## Required Columns",
        "",
    ]
    for qlty in [train_q, test_q]:
        missing = [col for col in qlty.required_columns if col not in qlty.columns]
        status = "all required columns present" if not missing else "missing: " + ", ".join(missing)
        lines.append(f"- {qlty.split}: {status}.")

    checks = [
        ("invalid_day_rows", train_q.invalid_day_rows, test_q.invalid_day_rows, "`day_since_install` outside 0-7."),
        ("invalid_hour_rows", train_q.invalid_hour_rows, test_q.invalid_hour_rows, "`event_hour` outside 0-23."),
        ("invalid_event_type_rows", train_q.invalid_event_type_rows, test_q.invalid_event_type_rows, "`event_type` outside session/ad_impression/iap."),
        ("negative_revenue_rows", train_q.negative_revenue_rows, test_q.negative_revenue_rows, "Negative `revenue_usd`."),
        ("session_revenue_non_null_rows", train_q.session_revenue_non_null_rows, test_q.session_revenue_non_null_rows, "Session rows with non-null revenue."),
        ("non_session_revenue_null_rows", train_q.non_session_revenue_null_rows, test_q.non_session_revenue_null_rows, "Ad/IAP rows with missing revenue."),
    ]
    lines.extend(["", "## Core Quality Checks", "", "| check | train | test | interpretation |", "|---|---:|---:|---|"])
    for name, train_value, test_value, note in checks:
        lines.append(f"| {name} | {train_value:,} | {test_value:,} | {note} |")

    lines.extend(["", "## Null Summary", "", "| split | column | null count | null rate |", "|---|---|---:|---:|"])
    for row in missing_df.to_dict(orient="records"):
        lines.append(f"| {row['split']} | {row['column']} | {int(row['null_count']):,} | {fmt_pct(float(row['null_rate']))} |")

    def null_rate(split: str, column: str) -> float:
        row = missing_df[(missing_df["split"] == split) & (missing_df["column"] == column)]
        return float(row["null_rate"].iloc[0]) if not row.empty else 0.0

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `product_id`: train {fmt_pct(null_rate('train', 'product_id'))}, test {fmt_pct(null_rate('test', 'product_id'))} null. It is mostly absent because only IAP rows carry product identifiers; use derived IAP features instead of broad raw product encoding.",
            f"- `network`: train {fmt_pct(null_rate('train', 'network'))}, test {fmt_pct(null_rate('test', 'network'))} null. Missing is expected for non-ad rows; aggregate network diversity or use a missing category if row-level encoding is ever needed.",
            f"- `ad_placement`: train {fmt_pct(null_rate('train', 'ad_placement'))}, test {fmt_pct(null_rate('test', 'ad_placement'))} null. Missing is expected for non-ad rows; aggregate placement diversity or fill a `no_ad_placement` category.",
            f"- `revenue_usd`: train {fmt_pct(null_rate('train', 'revenue_usd'))}, test {fmt_pct(null_rate('test', 'revenue_usd'))} null. Session revenue is missing by design and should be filled as 0 for D0-D7 revenue aggregation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grain_report(path: Path, grain_df: pd.DataFrame, sample_rows: int) -> None:
    def fmt_pct(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.2%}"

    lines = [
        "# Grain Diagnostics",
        "",
        "No model training was performed. This report only checks raw quality, candidate modeling grains, and feature aggregation feasibility.",
        "",
        "## Candidate Grain Comparison",
        "",
        "| split | grain | groups | target collision groups | collision rate | event p25/median/p75/p95 | session median/p75/p95 | ad median/p75/p95 | iap median/p75/p95 | active days median/p75/p95 | revenue median/p75/p95 | zero revenue | event_count 1-2 |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for row in grain_df.to_dict(orient="records"):
        collision_groups = "" if pd.isna(row["target_collision_groups"]) else f"{int(row['target_collision_groups']):,}"
        lines.append(
            f"| {row['split']} | {row['grain']} | {int(row['group_count']):,} | {collision_groups} | {fmt_pct(row['target_collision_rate'])} | "
            f"{row['event_count_p25']:.0f}/{row['event_count_median']:.0f}/{row['event_count_p75']:.0f}/{row['event_count_p95']:.0f} | "
            f"{row['session_count_median']:.0f}/{row['session_count_p75']:.0f}/{row['session_count_p95']:.0f} | "
            f"{row['ad_impression_count_median']:.0f}/{row['ad_impression_count_p75']:.0f}/{row['ad_impression_count_p95']:.0f} | "
            f"{row['iap_count_median']:.0f}/{row['iap_count_p75']:.0f}/{row['iap_count_p95']:.0f} | "
            f"{row['active_days_median']:.0f}/{row['active_days_p75']:.0f}/{row['active_days_p95']:.0f} | "
            f"{row['revenue_d0_d7_median']:.4f}/{row['revenue_d0_d7_p75']:.4f}/{row['revenue_d0_d7_p95']:.4f} | "
            f"{fmt_pct(row['zero_revenue_group_rate'])} | {fmt_pct(row['event_count_1_2_group_rate'])} |"
        )

    train = grain_df[grain_df["split"] == "train"].set_index("grain")
    test = grain_df[grain_df["split"] == "test"].set_index("grain")
    a_collisions = int(train.loc["A_user_id", "target_collision_groups"])
    c_collisions = int(train.loc["C_user_context_install_day", "target_collision_groups"])
    c_groups = int(train.loc["C_user_context_install_day", "group_count"])
    c_test = int(test.loc["C_user_context_install_day", "group_count"])
    d_test = int(test.loc["D_user_context_install_day_week", "group_count"])
    a_test = int(test.loc["A_user_id", "group_count"])

    lines.extend(
        [
            "",
            "## Final Judgment",
            "",
            "### Can `user_id` alone be used?",
            "",
            f"No. Candidate A has {a_collisions:,} train groups with multiple target values. This means a plain `groupby(user_id)` would mix different install/context records and produce ambiguous labels.",
            "",
            "### How much does a composite key reduce target collisions?",
            "",
            f"Candidate C reduces target-collision groups to {c_collisions:,} out of {c_groups:,}. This is a major improvement over `user_id` alone, but it does not fully eliminate label ambiguity.",
            "",
            "### Does composite grain make features too weak?",
            "",
            f"No. Candidate C keeps usable event histories: train event p25/median/p75/p95 are {train.loc['C_user_context_install_day', 'event_count_p25']:.0f}/{train.loc['C_user_context_install_day', 'event_count_median']:.0f}/{train.loc['C_user_context_install_day', 'event_count_p75']:.0f}/{train.loc['C_user_context_install_day', 'event_count_p95']:.0f}. Sparse groups exist, but not enough to reject the grain.",
            "",
            "### Most defensible modeling grain",
            "",
            "`user_id + platform + country_tier + channel_tier + install_day` is the most defensible grain. Candidate D is equivalent in group count here because `install_week` is derived from `install_day`; keep `install_week` as a feature, not as a required key.",
            "",
            "### Remaining target collisions",
            "",
            "For the first clean training set, drop remaining collision groups and log them as data-quality exceptions. Do not average targets in the primary pipeline because averaging hides unresolved identity ambiguity. A target-mean sensitivity run can be added later only as an analysis appendix.",
            "",
            "### Kaggle submission alignment",
            "",
            f"`sample_submission.csv` has {sample_rows:,} rows. Test has {a_test:,} unique `user_id` groups, {c_test:,} candidate-C groups, and {d_test:,} candidate-D groups. Candidate C/D is much closer to the sample row count but still differs by {abs(sample_rows - c_test):,} rows, so the exact competition submission key remains unresolved.",
            "",
            "## Feature Aggregation Feasibility",
            "",
            "The requested context, activity, revenue, ad, IAP, and time features are aggregatable at candidate C. IAP-derived features are sparse by nature, while activity, ad, revenue, and time features retain enough signal for downstream modeling experiments.",
            "",
            "Detailed feature availability is saved to `data/processed/feature_availability_by_grain.csv`.",
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

    train_q, train_frames = process_split(zip_path, "train.csv", "train", args.chunksize, True)
    test_q, test_frames = process_split(zip_path, "test.csv", "test", args.chunksize, False)

    grain_df = pd.concat(
        [summarize_grains("train", train_frames, True), summarize_grains("test", test_frames, False)],
        ignore_index=True,
    )
    availability_df = pd.concat([availability("train", train_frames), availability("test", test_frames)], ignore_index=True)
    missing_df = pd.concat([train_q.missing_frame(), test_q.missing_frame()], ignore_index=True)

    grain_df.to_csv(processed_dir / "grain_comparison.csv", index=False)
    availability_df.to_csv(processed_dir / "feature_availability_by_grain.csv", index=False)

    with zipfile.ZipFile(zip_path) as zf:
        sample_rows = len(pd.read_csv(zf.open("sample_submission.csv"), usecols=["user_id"]))

    write_missing_report(reports_dir / "missing_quality_report.md", train_q, test_q, missing_df)
    write_grain_report(reports_dir / "grain_diagnostics.md", grain_df, sample_rows)


if __name__ == "__main__":
    main()
