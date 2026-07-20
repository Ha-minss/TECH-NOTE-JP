import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.predict_submission import (
    FINAL_MODEL,
    aggregate_test_predictions,
    build_fallback_predictions,
    build_submission_frame,
    validate_submission_outputs,
)


class CreateSubmissionTests(unittest.TestCase):
    def _train(self):
        return pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "platform": ["ios", "ios", "android", "android"],
                "country_tier": ["US", "US", "KR", "KR"],
                "channel_tier": ["paid", "organic", "paid", "paid"],
                "ltv_d8_d180": [10.0, 20.0, 30.0, 50.0],
            }
        )

    def test_context_predictions_are_aggregated_to_one_row_per_user(self):
        row_predictions = pd.DataFrame(
            {
                "user_id": [2, 2, 7],
                "platform": ["android", "android", "ios"],
                "country_tier": ["US", "US", "KR"],
                "channel_tier": ["paid", "paid", "organic"],
                "pred_ltv": [1.5, 2.5, 4.0],
                "p_positive": [0.2, 0.8, 0.5],
                "positive_ltv_pred": [3.0, 5.0, 8.0],
            }
        )

        out = aggregate_test_predictions(row_predictions)

        self.assertEqual(out["user_id"].tolist(), [2, 7])
        self.assertAlmostEqual(out.loc[out["user_id"] == 2, "pred_ltv"].iloc[0], 2.0)
        self.assertAlmostEqual(out.loc[out["user_id"] == 2, "p_positive"].iloc[0], 0.5)
        self.assertEqual(int(out.loc[out["user_id"] == 2, "model_row_count"].iloc[0]), 2)

    def test_fallback_predictions_use_segment_then_country_channel_then_global(self):
        train = self._train()
        missing = pd.DataFrame(
            {
                "user_id": [101, 102, 103],
                "platform": ["ios", "android", "web"],
                "country_tier": ["US", "US", "ZZ"],
                "channel_tier": ["paid", "organic", "new"],
            }
        )

        out = build_fallback_predictions(missing, train)

        self.assertEqual(out.loc[out["user_id"] == 101, "fallback_method"].iloc[0], "platform_country_channel_mean")
        self.assertAlmostEqual(out.loc[out["user_id"] == 101, "ltv_d8_d180"].iloc[0], 10.0)
        self.assertEqual(out.loc[out["user_id"] == 102, "fallback_method"].iloc[0], "country_channel_mean")
        self.assertAlmostEqual(out.loc[out["user_id"] == 102, "ltv_d8_d180"].iloc[0], 20.0)
        self.assertEqual(out.loc[out["user_id"] == 103, "fallback_method"].iloc[0], "global_mean")
        self.assertAlmostEqual(out.loc[out["user_id"] == 103, "ltv_d8_d180"].iloc[0], 27.5)

    def test_submission_uses_expected_unique_test_users_not_sample_row_count(self):
        expected_user_ids = pd.Series([2, 7, 9], name="user_id")
        predictions = pd.DataFrame(
            {
                "user_id": [2, 7],
                "platform": ["android", "ios"],
                "country_tier": ["KR", "US"],
                "channel_tier": ["paid", "paid"],
                "pred_ltv": [3.5, 1.5],
                "p_positive": [0.7, 0.5],
                "positive_ltv_pred": [5.0, 3.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "user_id": [2, 2, 7, 9],
                "platform": ["android", "android", "ios", "ios"],
                "country_tier": ["KR", "KR", "US", "US"],
                "channel_tier": ["paid", "paid", "paid", "paid"],
            }
        )

        submission, fallback_rows = build_submission_frame(expected_user_ids, predictions, test_features, self._train())

        self.assertEqual(submission["user_id"].tolist(), [2, 7, 9])
        self.assertEqual(submission["ltv_d8_d180"].tolist()[:2], [3.5, 1.5])
        self.assertEqual(len(fallback_rows), 1)
        self.assertEqual(int(fallback_rows["user_id"].iloc[0]), 9)
        self.assertTrue((submission["ltv_d8_d180"] >= 0).all())

    def test_validate_submission_outputs_checks_core_contracts(self):
        expected_ids = pd.Series([1, 2], name="user_id")
        submission = pd.DataFrame({"user_id": [1, 2], "ltv_d8_d180": [1.0, 2.0]})
        fallback = pd.DataFrame({"user_id": [2], "fallback_method": ["global_mean"]})
        artifacts = {"stage1": True, "stage2": True, "preprocessor": True}

        checks = validate_submission_outputs(expected_ids, submission, fallback, expected_fallback_count=1, artifacts=artifacts)

        self.assertTrue(checks["row_count_matches_expected_users"])
        self.assertTrue(checks["id_order_matches_expected_users"])
        self.assertTrue(checks["prediction_not_null"])
        self.assertTrue(checks["prediction_not_inf"])
        self.assertTrue(checks["prediction_non_negative"])
        self.assertTrue(checks["fallback_count_matches_expected"])
        self.assertTrue(checks["model_artifacts_exist"])
        self.assertEqual(FINAL_MODEL, "optuna_two_stage_top_capture")


if __name__ == "__main__":
    unittest.main()

