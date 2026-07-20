import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.train_xgboost_model import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    build_feature_importance,
    build_feature_lists,
    fit_xgboost_model,
    make_time_split,
    restore_ltv_scale,
    write_feature_importance,
)


class XGBoostModelTests(unittest.TestCase):
    def _frame(self):
        rows = []
        for i in range(18):
            rows.append(
                {
                    "user_id": i,
                    "platform": "android" if i % 2 == 0 else "ios",
                    "country_tier": "US" if i % 3 == 0 else "KR",
                    "channel_tier": "paid" if i % 2 == 0 else "organic",
                    "install_day": i if i < 12 else 24 + (i - 12),
                    "install_week": 0 if i < 12 else 3,
                    "top_network": "net_a" if i < 15 else "new_net",
                    "top_ad_placement": "reward" if i < 15 else "new_place",
                    "event_count": 2 + i,
                    "session_count": 1 + i,
                    "revenue_d0_d7": float(i),
                    "ads_per_session": float(i) / 3,
                    "ltv_d8_d180": float(i * 1.5),
                }
            )
        return pd.DataFrame(rows)

    def test_time_split_uses_install_day_windows(self):
        train_split, valid_split = make_time_split(self._frame())

        self.assertEqual(len(train_split), 12)
        self.assertEqual(len(valid_split), 6)
        self.assertTrue((train_split["install_day"] <= 23).all())
        self.assertTrue((valid_split["install_day"] >= 24).all())

    def test_feature_lists_exclude_leakage_columns(self):
        numeric_features, categorical_features = build_feature_lists(self._frame())

        for col in EXCLUDED_FEATURES:
            self.assertNotIn(col, numeric_features)
            self.assertNotIn(col, categorical_features)
        self.assertEqual(categorical_features, CATEGORICAL_FEATURES)
        self.assertIn("revenue_d0_d7", numeric_features)

    def test_restore_ltv_scale_clips_negative_predictions(self):
        pred = restore_ltv_scale(np.array([-5.0, 0.0, np.log1p(3.0)]))

        self.assertTrue((pred >= 0).all())
        self.assertAlmostEqual(pred[-1], 3.0)

    def test_xgboost_predicts_with_unseen_categorical_values(self):
        train_split, valid_split = make_time_split(self._frame())

        result = fit_xgboost_model(train_split, valid_split, n_estimators=20, early_stopping_rounds=5)

        self.assertEqual(len(result.prediction), len(valid_split))
        self.assertFalse(pd.Series(result.prediction).isna().any())
        self.assertFalse(np.isinf(np.asarray(result.prediction, dtype=float)).any())
        self.assertTrue((np.asarray(result.prediction) >= 0).all())

    def test_feature_importance_file_is_created(self):
        train_split, valid_split = make_time_split(self._frame())
        result = fit_xgboost_model(train_split, valid_split, n_estimators=20, early_stopping_rounds=5)
        importance = build_feature_importance(result.pipeline)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "importance.csv"
            write_feature_importance(path, importance)
            self.assertTrue(path.exists())
            saved = pd.read_csv(path)
            self.assertIn("feature", saved.columns)
            self.assertIn("importance", saved.columns)


if __name__ == "__main__":
    unittest.main()
