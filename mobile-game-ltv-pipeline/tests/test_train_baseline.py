import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.train_baseline import (
    SEGMENT_COLUMNS,
    evaluate_predictions,
    make_time_split,
    predict_early_revenue_multiplier,
    predict_global_mean,
    predict_segment_mean,
)


class BaselineTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 1,
                    "install_week": 0,
                    "revenue_d0_d7": 1.0,
                    "ltv_d8_d180": 2.0,
                },
                {
                    "user_id": 2,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 2,
                    "install_week": 0,
                    "revenue_d0_d7": 3.0,
                    "ltv_d8_d180": 6.0,
                },
                {
                    "user_id": 3,
                    "platform": "ios",
                    "country_tier": "KR",
                    "channel_tier": "organic",
                    "install_day": 24,
                    "install_week": 3,
                    "revenue_d0_d7": 4.0,
                    "ltv_d8_d180": 8.0,
                },
                {
                    "user_id": 4,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 25,
                    "install_week": 0,
                    "revenue_d0_d7": 5.0,
                    "ltv_d8_d180": 10.0,
                },
            ]
        )

    def test_time_split_uses_install_day_windows(self):
        train_split, valid_split = make_time_split(self._frame())

        self.assertEqual(train_split["user_id"].tolist(), [1, 2])
        self.assertEqual(valid_split["user_id"].tolist(), [3, 4])

    def test_global_mean_predicts_train_split_mean(self):
        train_split, valid_split = make_time_split(self._frame())

        pred = predict_global_mean(train_split, valid_split)

        self.assertEqual(len(pred), len(valid_split))
        self.assertTrue(np.allclose(pred, [4.0, 4.0]))

    def test_segment_mean_falls_back_to_global_mean_for_unseen_segments(self):
        train_split, valid_split = make_time_split(self._frame())

        pred = predict_segment_mean(train_split, valid_split)

        self.assertEqual(SEGMENT_COLUMNS, ["platform", "country_tier", "channel_tier", "install_week"])
        self.assertTrue(np.allclose(pred, [4.0, 4.0]))

    def test_early_revenue_multiplier_uses_multiplier_and_clips_non_negative(self):
        train_split, valid_split = make_time_split(self._frame())

        pred = predict_early_revenue_multiplier(train_split, valid_split)

        self.assertTrue(np.allclose(pred, [8.0, 10.0]))
        self.assertTrue((pred >= 0).all())

    def test_early_revenue_multiplier_falls_back_when_revenue_mean_is_zero(self):
        frame = self._frame()
        frame.loc[frame["install_day"] <= 23, "revenue_d0_d7"] = 0.0
        train_split, valid_split = make_time_split(frame)

        pred = predict_early_revenue_multiplier(train_split, valid_split)

        self.assertTrue(np.allclose(pred, [4.0, 4.0]))

    def test_evaluate_predictions_returns_business_metrics(self):
        y_true = pd.Series([0.0, 1.0, 5.0, 10.0])
        y_pred = pd.Series([0.0, 2.0, 4.0, 8.0])

        metrics = evaluate_predictions(y_true, y_pred)

        for key in [
            "mae",
            "rmse",
            "rmsle",
            "spearman_corr",
            "positive_ltv_rate_pred_top_decile",
            "top_10pct_revenue_capture",
            "top_decile_lift",
        ]:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_top_decile_uses_only_top_ten_percent_when_predictions_are_tied(self):
        y_true = pd.Series([1.0] * 9 + [100.0])
        y_pred = pd.Series([5.0] * 10)

        metrics = evaluate_predictions(y_true, y_pred)

        self.assertLess(metrics["top_10pct_revenue_capture"], 1.0)
        self.assertAlmostEqual(metrics["top_decile_lift"], metrics["top_10pct_revenue_capture"] / 0.1)


if __name__ == "__main__":
    unittest.main()

