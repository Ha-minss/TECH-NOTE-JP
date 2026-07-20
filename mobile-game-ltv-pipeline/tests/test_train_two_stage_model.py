import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.train_two_stage_model import (
    TWO_STAGE_FEATURE_SETS,
    build_final_prediction,
    build_stage2_training_frame,
    compute_stage1_diagnostics,
    compute_stage2_diagnostics,
    make_time_split,
    prepare_two_stage_feature_frames,
    restore_ltv_scale,
    validate_prediction_frame,
)
from experiments.run_feature_ablation import build_time_bucket_features


class TwoStageModelTests(unittest.TestCase):
    def _base_frame(self):
        rows = []
        for i in range(12):
            rows.append(
                {
                    "user_id": i,
                    "platform": "android" if i % 2 == 0 else "ios",
                    "country_tier": "US" if i % 3 == 0 else "KR",
                    "channel_tier": "paid" if i % 2 == 0 else "organic",
                    "install_day": i if i < 8 else 24 + (i - 8),
                    "install_week": 0 if i < 8 else 3,
                    "top_network": "net_a" if i < 10 else "new_net",
                    "top_ad_placement": "reward" if i < 10 else "banner",
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
        base = self._base_frame()
        rows = []
        for row in base.itertuples(index=False):
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
                        "revenue_usd": 0.0 if day == 0 else float(day),
                    }
                )
        return pd.DataFrame(rows)

    def test_time_split_uses_install_day_windows(self):
        train_split, valid_split = make_time_split(self._base_frame())

        self.assertEqual(len(train_split), 8)
        self.assertEqual(len(valid_split), 4)
        self.assertTrue((train_split["install_day"] <= 23).all())
        self.assertTrue((valid_split["install_day"] >= 24).all())

    def test_stage2_training_uses_only_positive_train_rows(self):
        train_split, _ = make_time_split(self._base_frame())
        stage2_train = build_stage2_training_frame(train_split)

        self.assertTrue((stage2_train["ltv_d8_d180"] > 0).all())
        self.assertLess(len(stage2_train), len(train_split))

    def test_final_prediction_is_probability_times_positive_prediction(self):
        p_positive = pd.Series([0.2, 0.8])
        positive_pred = pd.Series([10.0, 5.0])

        final_pred = build_final_prediction(p_positive, positive_pred)

        self.assertAlmostEqual(final_pred.iloc[0], 2.0)
        self.assertAlmostEqual(final_pred.iloc[1], 4.0)

    def test_restore_ltv_scale_clips_negative_predictions(self):
        pred = restore_ltv_scale(np.array([-10.0, np.log1p(2.5)]))

        self.assertTrue((pred >= 0).all())
        self.assertAlmostEqual(pred[-1], 2.5)

    def test_prediction_validation_checks_probability_and_final_values(self):
        frame = pd.DataFrame(
            {
                "p_positive": [0.0, 1.0],
                "positive_ltv_pred": [3.0, 4.0],
                "pred_two_stage_x": [0.0, 4.0],
            }
        )

        checks = validate_prediction_frame(frame, "two_stage_x", valid_rows=2)

        self.assertTrue(checks["rows_match_valid"])
        self.assertEqual(checks["probability_out_of_range"], 0)
        self.assertEqual(checks["negative_predictions"], 0)

    def test_stage_diagnostics_are_finite(self):
        y_true = pd.Series([0, 1, 1, 0])
        p_positive = pd.Series([0.1, 0.8, 0.7, 0.2])
        stage1 = compute_stage1_diagnostics("two_stage_test", y_true, p_positive)
        stage2 = compute_stage2_diagnostics(
            "two_stage_test",
            pd.Series([1.0, 4.0]),
            pd.Series([1.5, 3.0]),
        )

        self.assertTrue(np.isfinite(stage1["roc_auc"]))
        self.assertTrue(np.isfinite(stage1["pr_auc"]))
        self.assertTrue(np.isfinite(stage2["positive_only_rmsle"]))

    def test_feature_generation_keeps_target_encoding_train_only(self):
        base = self._base_frame()
        bucket = build_time_bucket_features(self._raw_events())
        train_split, valid_split = make_time_split(base)
        valid_split = valid_split.copy()
        valid_split.loc[:, "ltv_d8_d180"] = 999999.0
        valid_split.loc[:, "channel_tier"] = "unseen_channel"

        train_out, valid_out, metadata = prepare_two_stage_feature_frames(
            "xgb_target_encoding_features", train_split, valid_split, bucket
        )

        expected_fallback = float(np.log1p(train_split["ltv_d8_d180"].clip(lower=0)).mean())
        self.assertAlmostEqual(valid_out["te_channel_ltv_log_mean"].iloc[0], expected_fallback)
        self.assertIn("target_encoding_columns", metadata)

    def test_two_stage_feature_sets_are_limited_to_requested_candidates(self):
        self.assertEqual(
            TWO_STAGE_FEATURE_SETS,
            ["xgb_velocity_ratio_features", "xgb_target_encoding_features"],
        )


if __name__ == "__main__":
    unittest.main()

