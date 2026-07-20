import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.validate_model_input import (
    GRAIN_COLUMNS,
    apply_model_input_preprocessing,
    validate_feature_tables,
)


class ModelInputValidationTests(unittest.TestCase):
    def _base_frames(self):
        train = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 1,
                    "install_week": 0,
                    "days_to_first_iap": np.nan,
                    "top_network": None,
                    "top_ad_placement": None,
                    "event_count": 3,
                    "ads_per_session": np.inf,
                    "ltv_d8_d180": 0.0,
                },
                {
                    "user_id": 2,
                    "platform": "ios",
                    "country_tier": "KR",
                    "channel_tier": "organic",
                    "install_day": 25,
                    "install_week": 3,
                    "days_to_first_iap": 2.0,
                    "top_network": "net_a",
                    "top_ad_placement": "reward",
                    "event_count": 10,
                    "ads_per_session": 1.5,
                    "ltv_d8_d180": 5.0,
                },
            ]
        )
        test = train.drop(columns=["ltv_d8_d180"]).copy()
        test["user_id"] = [3, 4]
        return train, test

    def test_preprocessing_applies_fixed_missing_rules(self):
        train, _ = self._base_frames()

        processed = apply_model_input_preprocessing(train)

        self.assertEqual(processed.loc[0, "days_to_first_iap"], 99)
        self.assertEqual(processed.loc[0, "top_network"], "no_ad_network")
        self.assertEqual(processed.loc[0, "top_ad_placement"], "no_ad_placement")
        self.assertEqual(processed.loc[0, "ads_per_session"], 0)
        self.assertFalse(np.isinf(processed.select_dtypes(include=[np.number]).to_numpy()).any())
        self.assertFalse(processed.select_dtypes(include=[np.number]).isna().any().any())

    def test_validate_feature_tables_checks_schema_target_grain_and_split(self):
        train, test = self._base_frames()

        report = validate_feature_tables(train, test)

        self.assertTrue(report["schema"]["feature_columns_match"])
        self.assertTrue(report["schema"]["target_only_in_train"])
        self.assertEqual(report["grain"]["train_duplicate_grain_rows"], 0)
        self.assertEqual(report["grain"]["test_duplicate_grain_rows"], 0)
        self.assertTrue(report["split"]["time_split_possible"])
        self.assertEqual(report["split"]["train_rows_install_day_0_23"], 1)
        self.assertEqual(report["split"]["valid_rows_install_day_24_30"], 1)

    def test_validate_feature_tables_flags_duplicate_grain(self):
        train, test = self._base_frames()
        duplicated = pd.concat([train, train.iloc[[0]]], ignore_index=True)

        report = validate_feature_tables(duplicated, test)

        self.assertEqual(report["grain"]["train_duplicate_grain_rows"], 1)
        self.assertEqual(duplicated.duplicated(GRAIN_COLUMNS).sum(), 1)


if __name__ == "__main__":
    unittest.main()
