import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.train_linear_model import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    build_feature_lists,
    fit_linear_models,
    make_time_split,
    restore_ltv_scale,
)


class LinearModelTests(unittest.TestCase):
    def _frame(self):
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
                    "top_network": "net_a" if i < 10 else "unseen_net",
                    "top_ad_placement": "reward" if i < 10 else "unseen_place",
                    "event_count": 2 + i,
                    "session_count": 1 + i,
                    "revenue_d0_d7": float(i),
                    "ads_per_session": float(i) / 2,
                    "ltv_d8_d180": float(i * 2),
                }
            )
        return pd.DataFrame(rows)

    def test_time_split_uses_only_install_day_windows(self):
        train_split, valid_split = make_time_split(self._frame())

        self.assertTrue((train_split["install_day"] <= 23).all())
        self.assertTrue((valid_split["install_day"] >= 24).all())
        self.assertEqual(len(train_split), 8)
        self.assertEqual(len(valid_split), 4)

    def test_feature_lists_exclude_ids_split_columns_and_target(self):
        frame = self._frame()
        numeric_features, categorical_features = build_feature_lists(frame)

        for col in EXCLUDED_FEATURES:
            self.assertNotIn(col, numeric_features)
            self.assertNotIn(col, categorical_features)
        self.assertEqual(categorical_features, CATEGORICAL_FEATURES)
        self.assertIn("revenue_d0_d7", numeric_features)

    def test_restore_ltv_scale_clips_negative_predictions(self):
        restored = restore_ltv_scale(np.array([-10.0, 0.0, np.log1p(4.0)]))

        self.assertTrue((restored >= 0).all())
        self.assertAlmostEqual(restored[-1], 4.0)

    def test_models_predict_with_unseen_categorical_values(self):
        frame = self._frame()
        train_split, valid_split = make_time_split(frame)

        result = fit_linear_models(train_split, valid_split)

        self.assertEqual(set(result.predictions), {"ridge_log_linear", "elasticnet_log_linear"})
        for pred in result.predictions.values():
            self.assertEqual(len(pred), len(valid_split))
            self.assertFalse(pd.Series(pred).isna().any())
            self.assertFalse(np.isinf(np.asarray(pred, dtype=float)).any())
            self.assertTrue((np.asarray(pred) >= 0).all())


if __name__ == "__main__":
    unittest.main()
