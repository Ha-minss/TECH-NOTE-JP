import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_feature_ablation import (
    FEATURE_SETS,
    add_frequency_interaction_features,
    add_target_encoding_features,
    add_velocity_ratio_features,
    build_time_bucket_features,
    get_feature_set_frame,
    make_time_split,
    restore_ltv_scale,
)


class FeatureEngineeringExperimentTests(unittest.TestCase):
    def _base_frame(self):
        return pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 0,
                    "install_week": 0,
                    "top_network": "net_a",
                    "top_ad_placement": "reward",
                    "event_count": 4,
                    "session_count": 2,
                    "ad_impression_count": 1,
                    "iap_count": 1,
                    "revenue_d0_d7": 10.0,
                    "revenue_d0": 1.0,
                    "revenue_d1": 2.0,
                    "revenue_d2_d3": 3.0,
                    "revenue_d4_d7": 4.0,
                    "ltv_d8_d180": 20.0,
                },
                {
                    "user_id": 2,
                    "platform": "ios",
                    "country_tier": "KR",
                    "channel_tier": "organic",
                    "install_day": 24,
                    "install_week": 3,
                    "top_network": "new_net",
                    "top_ad_placement": "banner",
                    "event_count": 2,
                    "session_count": 1,
                    "ad_impression_count": 1,
                    "iap_count": 0,
                    "revenue_d0_d7": 0.0,
                    "revenue_d0": 0.0,
                    "revenue_d1": 0.0,
                    "revenue_d2_d3": 0.0,
                    "revenue_d4_d7": 0.0,
                    "ltv_d8_d180": 0.0,
                },
            ]
        )

    def _raw_events(self):
        return pd.DataFrame(
            [
                {"user_id": 1, "platform": "android", "country_tier": "US", "channel_tier": "paid", "install_day": 0, "day_since_install": 0, "event_type": "session", "revenue_usd": np.nan},
                {"user_id": 1, "platform": "android", "country_tier": "US", "channel_tier": "paid", "install_day": 0, "day_since_install": 1, "event_type": "ad_impression", "revenue_usd": 0.5},
                {"user_id": 1, "platform": "android", "country_tier": "US", "channel_tier": "paid", "install_day": 0, "day_since_install": 4, "event_type": "iap", "revenue_usd": 3.0},
                {"user_id": 2, "platform": "ios", "country_tier": "KR", "channel_tier": "organic", "install_day": 24, "day_since_install": 7, "event_type": "session", "revenue_usd": np.nan},
            ]
        )

    def test_time_bucket_features_are_aggregated_by_modeling_grain(self):
        buckets = build_time_bucket_features(self._raw_events())

        self.assertIn("event_count_d4_d7", buckets.columns)
        first = buckets[buckets["user_id"] == 1].iloc[0]
        self.assertEqual(first["event_count_d0"], 1)
        self.assertEqual(first["ad_impression_count_d1"], 1)
        self.assertEqual(first["iap_count_d4_d7"], 1)

    def test_velocity_ratio_features_are_safe_when_denominator_is_zero(self):
        frame = add_velocity_ratio_features(
            self._base_frame().assign(
                event_count_d0=0,
                event_count_d1=0,
                event_count_d4_d7=[2, 0],
                session_count_d0=0,
                session_count_d1=0,
                session_count_d4_d7=[1, 0],
                ad_impression_count_d0=0,
                ad_impression_count_d1=0,
                ad_impression_count_d4_d7=[1, 0],
            )
        )

        self.assertFalse(np.isinf(frame["event_growth_d4_d7_vs_d0_d1"]).any())
        self.assertFalse(frame["late_revenue_share"].isna().any())

    def test_frequency_features_use_train_split_counts_for_valid_rows(self):
        train_split = self._base_frame().iloc[[0]].copy()
        valid_split = self._base_frame().iloc[[1]].copy()

        train_out, valid_out, metadata = add_frequency_interaction_features(train_split, valid_split)

        self.assertIn("platform_country_freq", train_out.columns)
        self.assertEqual(train_out["platform_country_freq"].iloc[0], 1)
        self.assertEqual(valid_out["platform_country_freq"].iloc[0], 0)
        self.assertIn("platform_country", metadata["interaction_columns"])

    def test_target_encoding_uses_train_only_and_global_fallback(self):
        train_split = self._base_frame().iloc[[0]].copy()
        valid_split = self._base_frame().iloc[[1]].copy()

        train_out, valid_out, metadata = add_target_encoding_features(train_split, valid_split)

        global_log_mean = float(np.log1p(train_split["ltv_d8_d180"]).mean())
        self.assertAlmostEqual(valid_out["te_channel_ltv_log_mean"].iloc[0], global_log_mean)
        self.assertAlmostEqual(valid_out["te_channel_positive_rate"].iloc[0], 1.0)
        self.assertEqual(metadata["fallback_log_mean"], global_log_mean)

    def test_feature_set_frames_are_cumulative(self):
        base = self._base_frame()
        bucket = build_time_bucket_features(self._raw_events())
        train_split, valid_split = make_time_split(base)

        current_train, current_valid, _ = get_feature_set_frame("xgb_current_full", train_split, valid_split, bucket)
        ratio_train, ratio_valid, _ = get_feature_set_frame("xgb_velocity_ratio_features", train_split, valid_split, bucket)

        self.assertEqual(FEATURE_SETS[0], "xgb_current_full")
        self.assertNotIn("late_event_share", current_train.columns)
        self.assertIn("late_event_share", ratio_train.columns)
        self.assertEqual(len(ratio_valid), len(valid_split))

    def test_restore_ltv_scale_clips_negative_predictions(self):
        pred = restore_ltv_scale(np.array([-3.0, np.log1p(4.0)]))

        self.assertTrue((pred >= 0).all())
        self.assertAlmostEqual(pred[-1], 4.0)


if __name__ == "__main__":
    unittest.main()
