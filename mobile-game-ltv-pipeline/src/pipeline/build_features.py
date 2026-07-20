from __future__ import annotations

import argparse
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GRAIN_COLUMNS = ["user_id", "platform", "country_tier", "channel_tier", "install_day"]
CONTEXT_COLUMNS = ["platform", "country_tier", "channel_tier", "install_day", "install_week"]


def _key(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _update_counter(counter: Counter, series: pd.Series) -> None:
    for key, value in series.items():
        counter[_key(key)] += float(value)


def _update_min(store: dict[tuple[Any, ...], float], series: pd.Series) -> None:
    for key, value in series.items():
        key = _key(key)
        value = float(value)
        store[key] = min(store.get(key, value), value)


def _update_max(store: dict[tuple[Any, ...], float], series: pd.Series) -> None:
    for key, value in series.items():
        key = _key(key)
        value = float(value)
        store[key] = max(store.get(key, value), value)


def _mode(counter: Counter) -> Any:
    if not counter:
        return np.nan
    return counter.most_common(1)[0][0]


@dataclass
class FeatureAccumulator:
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
    network_counts: dict[tuple[Any, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    placement_counts: dict[tuple[Any, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    network_revenue: dict[tuple[Any, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    placement_revenue: dict[tuple[Any, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))
    product_values: dict[tuple[Any, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    first_iap_day: dict[tuple[Any, ...], float] = field(default_factory=dict)
    iap_amount_sum: Counter = field(default_factory=Counter)
    max_iap_amount: dict[tuple[Any, ...], float] = field(default_factory=dict)
    target_min: dict[tuple[Any, ...], float] = field(default_factory=dict)
    target_max: dict[tuple[Any, ...], float] = field(default_factory=dict)
    target_value: dict[tuple[Any, ...], float] = field(default_factory=dict)

    def _group(self, df: pd.DataFrame):
        return df.groupby(GRAIN_COLUMNS, dropna=False, sort=False)

    def update(self, df: pd.DataFrame) -> None:
        work = df.copy()
        work["revenue_usd"] = work["revenue_usd"].fillna(0.0)
        grouped = self._group(work)

        _update_counter(self.event_count, grouped.size())
        _update_min(self.min_day, grouped["day_since_install"].min())
        _update_max(self.max_day, grouped["day_since_install"].max())
        _update_min(self.min_hour, grouped["event_hour"].min())
        _update_max(self.max_hour, grouped["event_hour"].max())

        _update_counter(self.session_count, self._group(work[work["event_type"] == "session"]).size())
        _update_counter(self.ad_count, self._group(work[work["event_type"] == "ad_impression"]).size())
        _update_counter(self.iap_count, self._group(work[work["event_type"] == "iap"]).size())

        _update_counter(self.revenue, grouped["revenue_usd"].sum())
        ad_rows = work[work["event_type"] == "ad_impression"]
        iap_rows = work[work["event_type"] == "iap"]
        _update_counter(self.ad_revenue, self._group(ad_rows)["revenue_usd"].sum())
        _update_counter(self.iap_revenue, self._group(iap_rows)["revenue_usd"].sum())
        _update_counter(self.iap_amount_sum, self._group(iap_rows)["revenue_usd"].sum())
        _update_max(self.max_iap_amount, self._group(iap_rows)["revenue_usd"].max())

        _update_counter(self.revenue_d0, self._group(work[work["day_since_install"] == 0])["revenue_usd"].sum())
        _update_counter(self.revenue_d1, self._group(work[work["day_since_install"] == 1])["revenue_usd"].sum())
        _update_counter(self.revenue_d2_d3, self._group(work[work["day_since_install"].between(2, 3)])["revenue_usd"].sum())
        _update_counter(self.revenue_d4_d7, self._group(work[work["day_since_install"].between(4, 7)])["revenue_usd"].sum())

        if not iap_rows.empty:
            _update_min(self.first_iap_day, self._group(iap_rows)["day_since_install"].min())

        if self.has_target:
            _update_min(self.target_min, grouped["ltv_d8_d180"].min())
            _update_max(self.target_max, grouped["ltv_d8_d180"].max())
            for key, value in grouped["ltv_d8_d180"].min().items():
                self.target_value[_key(key)] = float(value)

        for row in work[GRAIN_COLUMNS + ["day_since_install"]].drop_duplicates().itertuples(index=False, name=None):
            key = tuple(row[: len(GRAIN_COLUMNS)])
            self.active_day_mask[key] = int(self.active_day_mask[key]) | (1 << int(row[-1]))

        for row in work[GRAIN_COLUMNS + ["event_hour"]].drop_duplicates().itertuples(index=False, name=None):
            key = tuple(row[: len(GRAIN_COLUMNS)])
            self.active_hour_mask[key] = int(self.active_hour_mask[key]) | (1 << int(row[-1]))

        for idx, count in work.groupby(GRAIN_COLUMNS + ["event_hour"], dropna=False, sort=False).size().items():
            idx = _key(idx)
            self.hour_counts[idx[: len(GRAIN_COLUMNS)]][int(idx[-1])] += int(count)

        for col, counter_store in [("network", self.network_counts), ("ad_placement", self.placement_counts)]:
            values = work[work[col].notna()].groupby(GRAIN_COLUMNS + [col], dropna=False, sort=False).size()
            for idx, count in values.items():
                idx = _key(idx)
                counter_store[idx[: len(GRAIN_COLUMNS)]][str(idx[-1])] += int(count)

        for col, revenue_store in [("network", self.network_revenue), ("ad_placement", self.placement_revenue)]:
            values = ad_rows[ad_rows[col].notna()].groupby(GRAIN_COLUMNS + [col], dropna=False, sort=False)["revenue_usd"].sum()
            for idx, value in values.items():
                idx = _key(idx)
                revenue_store[idx[: len(GRAIN_COLUMNS)]][str(idx[-1])] += float(value)

        for row in iap_rows[iap_rows["product_id"].notna()][GRAIN_COLUMNS + ["product_id"]].drop_duplicates().itertuples(index=False, name=None):
            self.product_values[tuple(row[: len(GRAIN_COLUMNS)])].add(str(row[-1]))

    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_rows: list[dict[str, Any]] = []
        dropped_rows: list[dict[str, Any]] = []
        for key in sorted(self.event_count):
            base = dict(zip(GRAIN_COLUMNS, key))
            tmin = self.target_min.get(key)
            tmax = self.target_max.get(key)
            collision = self.has_target and tmin is not None and tmax is not None and round(tmin, 10) != round(tmax, 10)
            if collision:
                dropped_rows.append(
                    {
                        **base,
                        "target_min": tmin,
                        "target_max": tmax,
                        "event_count": int(self.event_count[key]),
                    }
                )
                continue

            sessions = int(self.session_count[key])
            ads = int(self.ad_count[key])
            iaps = int(self.iap_count[key])
            ad_rev = float(self.ad_revenue[key])
            network_top = _mode(self.network_counts[key])
            placement_top = _mode(self.placement_counts[key])
            row = {
                **base,
                "install_week": int(int(base["install_day"]) // 7),
                "event_count": int(self.event_count[key]),
                "session_count": sessions,
                "ad_impression_count": ads,
                "iap_count": iaps,
                "active_days": int(int(self.active_day_mask[key]).bit_count()),
                "last_event_day": self.max_day.get(key, np.nan),
                "revenue_d0_d7": float(self.revenue[key]),
                "ad_revenue_d0_d7": ad_rev,
                "iap_revenue_d0_d7": float(self.iap_revenue[key]),
                "revenue_d0": float(self.revenue_d0[key]),
                "revenue_d1": float(self.revenue_d1[key]),
                "revenue_d2_d3": float(self.revenue_d2_d3[key]),
                "revenue_d4_d7": float(self.revenue_d4_d7[key]),
                "unique_network_count": len(self.network_counts[key]),
                "unique_ad_placement_count": len(self.placement_counts[key]),
                "top_network": network_top,
                "top_ad_placement": placement_top,
                "top_network_event_share": float(self.network_counts[key][network_top] / ads) if ads and pd.notna(network_top) else 0.0,
                "top_ad_placement_event_share": float(self.placement_counts[key][placement_top] / ads) if ads and pd.notna(placement_top) else 0.0,
                "top_network_revenue_share": float(self.network_revenue[key][network_top] / ad_rev) if ad_rev and pd.notna(network_top) else 0.0,
                "top_ad_placement_revenue_share": float(self.placement_revenue[key][placement_top] / ad_rev) if ad_rev and pd.notna(placement_top) else 0.0,
                "ads_per_session": float(ads / sessions) if sessions else 0.0,
                "ad_revenue_per_ad": float(ad_rev / ads) if ads else 0.0,
                "ad_revenue_per_session": float(ad_rev / sessions) if sessions else 0.0,
                "early_payer_flag": int(iaps > 0),
                "days_to_first_iap": self.first_iap_day.get(key, np.nan),
                "unique_product_count": len(self.product_values[key]),
                "avg_iap_amount": float(self.iap_amount_sum[key] / iaps) if iaps else 0.0,
                "max_iap_amount": float(self.max_iap_amount.get(key, 0.0)),
                "first_event_hour": self.min_hour.get(key, np.nan),
                "last_event_hour": self.max_hour.get(key, np.nan),
                "most_active_hour": _mode(self.hour_counts[key]),
                "active_hour_count": int(int(self.active_hour_mask[key]).bit_count()),
            }
            if self.has_target:
                row["ltv_d8_d180"] = self.target_value[key]
            feature_rows.append(row)

        return pd.DataFrame(feature_rows), pd.DataFrame(dropped_rows)


def build_feature_frame(df: pd.DataFrame, has_target: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    accumulator = FeatureAccumulator(has_target=has_target)
    accumulator.update(df)
    return accumulator.to_frames()


def build_features_from_csv_chunks(csv_file, has_target: bool, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    accumulator = FeatureAccumulator(has_target=has_target)
    for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
        accumulator.update(chunk)
    return accumulator.to_frames()


def build_features_from_zip(zip_path: Path, output_root: Path, chunksize: int = 500_000) -> dict[str, Any]:
    processed_dir = output_root / "data" / "processed"
    reports_dir = output_root / "reports" / "diagnostics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        train_features, dropped = build_features_from_csv_chunks(zf.open("train.csv"), has_target=True, chunksize=chunksize)
        test_features, _ = build_features_from_csv_chunks(zf.open("test.csv"), has_target=False, chunksize=chunksize)

    train_path = processed_dir / "train_features.parquet"
    test_path = processed_dir / "test_features.parquet"
    dropped_path = processed_dir / "dropped_collision_groups.csv"
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)
    dropped.to_csv(dropped_path, index=False)

    summary = {
        "train_features_rows": int(len(train_features)),
        "test_features_rows": int(len(test_features)),
        "dropped_collision_groups": int(len(dropped)),
        "train_columns": int(train_features.shape[1]),
        "test_columns": int(test_features.shape[1]),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "dropped_path": str(dropped_path),
    }
    write_summary(reports_dir / "feature_builder_summary.md", summary, train_features, test_features)
    return summary


def write_summary(path: Path, summary: dict[str, Any], train_features: pd.DataFrame, test_features: pd.DataFrame) -> None:
    lines = [
        "# Feature Builder Summary",
        "",
        "## Scope",
        "",
        "The modeling grain is fixed as `user_id + platform + country_tier + channel_tier + install_day`. No model training was performed.",
        "",
        "## Outputs",
        "",
        f"- Train feature rows: {summary['train_features_rows']:,}",
        f"- Test feature rows: {summary['test_features_rows']:,}",
        f"- Dropped train target-collision groups: {summary['dropped_collision_groups']:,}",
        f"- Train feature columns: {summary['train_columns']:,}",
        f"- Test feature columns: {summary['test_columns']:,}",
        f"- Train parquet: `{summary['train_path']}`",
        f"- Test parquet: `{summary['test_path']}`",
        f"- Collision log: `{summary['dropped_path']}`",
        "",
        "## Feature Rules",
        "",
        "- `revenue_usd` nulls are filled as 0 before aggregation.",
        "- `product_id` is not emitted as a raw categorical feature; it only contributes to IAP aggregate features.",
        "- `network` and `ad_placement` are not emitted as raw row-level features; they contribute to unique counts, top categories, event shares, and revenue shares.",
        "- Remaining target-collision groups are removed from train features and saved to the collision log.",
        "",
        "## Feature Columns",
        "",
        "```text",
        "\n".join(train_features.columns),
        "```",
        "",
        "## Quick Distribution Check",
        "",
        "| split | rows | median events | p95 events | zero D0-D7 revenue rate | early payer rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, frame in [("train", train_features), ("test", test_features)]:
        lines.append(
            f"| {split} | {len(frame):,} | {frame['event_count'].median():.0f} | "
            f"{frame['event_count'].quantile(0.95):.0f} | {(frame['revenue_d0_d7'] == 0).mean():.2%} | "
            f"{frame['early_payer_flag'].mean():.2%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    build_features_from_zip(Path(args.zip_path), Path(args.output_root), args.chunksize)


if __name__ == "__main__":
    main()
