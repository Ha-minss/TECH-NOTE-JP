import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.rolling_time_validation import (
    FOLDS,
    MODEL_SPECS,
    build_summary,
    make_expanding_time_splits,
    prepare_fold_features,
    validate_fold_predictions,
)
from experiments.run_feature_ablation import build_time_bucket_features


class RollingValidationTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for i, day in enumerate(range(0, 31)):
            rows.append(
                {
                    "user_id": i,
                    "platform": "android" if i % 2 == 0 else "ios",
                    "country_tier": "US" if i % 3 == 0 else "KR",
                    "channel_tier": "paid" if i % 2 == 0 else "organic",
                    "install_day": day,
                    "install_week": day // 7,
                    "top_network": "net_a" if i < 20 else "new_net",
                    "top_ad_placement": "reward" if i < 20 else "banner",
                    "event_count": 5 + i,
                    "session_count": 2 + i,
                    "ad_impression_count": 1 + i,
                    "iap_count": i % 3,
                    "revenue_d0_d7": float(i),
                    "revenue_d0": float(i % 2),
                    "revenue_d1": float(i % 3),
                    "revenue_d2_d3": float(i % 4),
                    "revenue_d4_d7": float(i % 5),
                    "ltv_d8_d180": 0.0 if i % 4 == 0 else float(i + 1),
                }
            )
        return pd.DataFrame(rows)

    def _raw_events(self):
        rows = []
        for row in self._frame().itertuples(index=False):
            for day in [0, 1, 4]:
                rows.append(
                    {
                        "user_id": row.user_id,
                        "platform": row.platform,
                        "country_tier": row.country_tier,
                        "channel_tier": row.channel_tier,
                        "install_day": row.install_day,
                        "day_since_install": day,
                        "event_type": "session" if day == 0 else ("ad_impression" if day == 1 else "iap"),
                        "revenue_usd": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_expanding_time_splits_follow_requested_windows(self):
        splits = make_expanding_time_splits(self._frame())

        self.assertEqual(len(splits), 4)
        self.assertEqual(FOLDS[0]["train_max"], 13)
        self.assertEqual(FOLDS[-1]["valid_min"], 24)
        self.assertEqual(splits[0][2]["install_day"].min(), 14)
        self.assertEqual(splits[-1][2]["install_day"].max(), 30)

    def test_model_specs_are_limited_to_requested_candidates(self):
        self.assertEqual(
            [spec["model_name"] for spec in MODEL_SPECS],
            [
                "single_stage_xgb_target_encoding_features",
                "single_stage_xgb_velocity_ratio_features",
                "two_stage_xgb_target_encoding_features",
                "two_stage_xgb_velocity_ratio_features",
            ],
        )

    def test_target_encoding_is_fit_from_train_fold_only(self):
        frame = self._frame()
        bucket = build_time_bucket_features(self._raw_events())
        train_split = frame[frame["install_day"].between(0, 13)].copy()
        valid_split = frame[frame["install_day"].between(14, 16)].copy()
        valid_split.loc[:, "ltv_d8_d180"] = 999999.0
        valid_split.loc[:, "channel_tier"] = "unseen_channel"

        train_out, valid_out, metadata = prepare_fold_features(
            "xgb_target_encoding_features", train_split, valid_split, bucket
        )

        expected = float(np.log1p(train_split["ltv_d8_d180"].clip(lower=0)).mean())
        self.assertAlmostEqual(valid_out["te_channel_ltv_log_mean"].iloc[0], expected)
        self.assertIn("target_encoding_columns", metadata)

    def test_prediction_validation_flags_bad_values(self):
        checks = validate_fold_predictions(pd.Series([1.0, np.nan, -1.0]), valid_rows=3)

        self.assertTrue(checks["rows_match_valid"])
        self.assertEqual(checks["null_predictions"], 1)
        self.assertEqual(checks["negative_predictions"], 1)

    def test_summary_aggregates_mean_std_and_winner_flags(self):
        metrics = pd.DataFrame(
            [
                {"model": "a", "model_family": "single_stage", "feature_set": "fa", "fold": 1, "mae": 1, "rmse": 1, "rmsle": 0.5, "spearman_corr": 0.1, "positive_ltv_rate_pred_top_decile": 1, "top_10pct_revenue_capture": 0.7, "top_decile_lift": 7},
                {"model": "a", "model_family": "single_stage", "feature_set": "fa", "fold": 2, "mae": 1, "rmse": 1, "rmsle": 0.7, "spearman_corr": 0.1, "positive_ltv_rate_pred_top_decile": 1, "top_10pct_revenue_capture": 0.6, "top_decile_lift": 6},
                {"model": "b", "model_family": "two_stage", "feature_set": "fb", "fold": 1, "mae": 1, "rmse": 1, "rmsle": 0.4, "spearman_corr": 0.1, "positive_ltv_rate_pred_top_decile": 1, "top_10pct_revenue_capture": 0.5, "top_decile_lift": 5},
                {"model": "b", "model_family": "two_stage", "feature_set": "fb", "fold": 2, "mae": 1, "rmse": 1, "rmsle": 0.6, "spearman_corr": 0.1, "positive_ltv_rate_pred_top_decile": 1, "top_10pct_revenue_capture": 0.8, "top_decile_lift": 8},
            ]
        )

        summary = build_summary(metrics)

        self.assertIn("rmsle_mean", summary.columns)
        self.assertIn("top_10pct_revenue_capture_mean", summary.columns)
        self.assertEqual(summary.loc[summary["model"] == "b", "best_rmsle_mean"].iloc[0], True)
        self.assertEqual(summary.loc[summary["model"] == "a", "best_top_capture_mean"].iloc[0], False)


if __name__ == "__main__":
    unittest.main()

