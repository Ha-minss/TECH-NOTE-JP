import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.build_business_report import (
    ALLOWED_UA_DECISIONS,
    FINAL_STAGE1_ARTIFACT,
    FINAL_STAGE2_ARTIFACT,
    MIN_UA_SEGMENT_USERS,
    add_top_decile_flag,
    build_low_sample_segments,
    build_segment_ltv_summary,
    build_synthetic_cpi_table,
    build_top_decile_numeric_analysis,
    build_ua_decision_simulation,
    validate_business_outputs,
    validate_final_artifacts,
)


class FinalBusinessAnalysisTests(unittest.TestCase):
    def _frame(self, n_rows=20):
        return pd.DataFrame(
            {
                "user_id": range(n_rows),
                "platform": (["android", "ios"] * ((n_rows // 2) + 1))[:n_rows],
                "country_tier": (["US", "KR", "BR", "OTHER"] * ((n_rows // 4) + 1))[:n_rows],
                "channel_tier": (["paid", "organic", "social", "paid"] * ((n_rows // 4) + 1))[:n_rows],
                "install_day": list(range(n_rows)),
                "install_week": [i // 7 for i in range(n_rows)],
                "ltv_d8_d180": np.linspace(0, 19, n_rows),
                "pred_ltv": np.linspace(19, 0, n_rows),
                "revenue_d0_d7": np.linspace(0, 10, n_rows),
                "ad_revenue_d0_d7": np.linspace(0, 2, n_rows),
                "iap_revenue_d0_d7": np.linspace(0, 8, n_rows),
                "active_days": ([1, 2, 3, 4] * ((n_rows // 4) + 1))[:n_rows],
                "last_event_day": ([0, 1, 4, 7] * ((n_rows // 4) + 1))[:n_rows],
                "event_count": np.arange(n_rows) + 1,
                "session_count": np.arange(n_rows) + 2,
                "ad_impression_count": np.arange(n_rows) + 3,
                "iap_count": ([0, 1] * ((n_rows // 2) + 1))[:n_rows],
                "ads_per_session": np.linspace(0.1, 2.0, n_rows),
                "revenue_per_active_day": np.linspace(0, 5, n_rows),
                "early_payer_flag": ([0, 1] * ((n_rows // 2) + 1))[:n_rows],
                "top_network": (["net_a", "net_b"] * ((n_rows // 2) + 1))[:n_rows],
                "top_ad_placement": (["reward", "banner"] * ((n_rows // 2) + 1))[:n_rows],
            }
        )

    def test_top_decile_is_exact_ceiling_ten_percent(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")

        self.assertEqual(int(frame["is_pred_top_decile"].sum()), 2)
        self.assertTrue(frame.sort_values("pred_ltv", ascending=False).head(2)["is_pred_top_decile"].all())

    def test_segment_summary_has_rows_and_required_metrics(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        summary = build_segment_ltv_summary(frame)

        self.assertGreater(len(summary), 0)
        for col in ["user_count", "actual_ltv_mean", "predicted_ltv_mean", "top_decile_user_share"]:
            self.assertIn(col, summary.columns)

    def test_synthetic_cpi_is_deterministic(self):
        frame = self._frame()
        first = build_synthetic_cpi_table(frame)
        second = build_synthetic_cpi_table(frame)

        pd.testing.assert_frame_equal(first.sort_index(axis=1), second.sort_index(axis=1))
        self.assertTrue((first["synthetic_cpi"] > 0).all())

    def test_ua_decision_outputs_valid_roas_and_decisions(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        ua = build_ua_decision_simulation(frame)

        self.assertGreater(len(ua), 0)
        self.assertFalse(ua["predicted_roas"].isna().any())
        self.assertFalse(np.isinf(ua["predicted_roas"].to_numpy(dtype=float)).any())
        self.assertTrue((ua["predicted_roas"] >= 0).all())
        self.assertTrue(set(ua["decision"]).issubset(ALLOWED_UA_DECISIONS | {"insufficient_sample"}))

    def test_low_sample_segments_are_insufficient_sample(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        ua = build_ua_decision_simulation(frame)

        low_sample = ua[ua["user_count"] < MIN_UA_SEGMENT_USERS]
        high_sample = ua[ua["user_count"] >= MIN_UA_SEGMENT_USERS]
        self.assertTrue((low_sample["decision"] == "insufficient_sample").all())
        self.assertTrue(set(high_sample["decision"]).issubset(ALLOWED_UA_DECISIONS))

    def test_high_sample_segments_can_receive_business_decisions(self):
        frame = self._frame(n_rows=120)
        frame["country_tier"] = "US"
        frame["channel_tier"] = "paid"
        frame = add_top_decile_flag(frame, pred_col="pred_ltv")
        ua = build_ua_decision_simulation(frame)

        self.assertTrue((ua["user_count"] >= MIN_UA_SEGMENT_USERS).all())
        self.assertTrue(set(ua["decision"]).issubset(ALLOWED_UA_DECISIONS))

    def test_low_sample_segments_output_is_separate_warning_table(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        ua = build_ua_decision_simulation(frame)
        low_sample = build_low_sample_segments(ua)

        self.assertGreater(len(low_sample), 0)
        self.assertTrue((low_sample["user_count"] < MIN_UA_SEGMENT_USERS).all())
        self.assertTrue((low_sample["decision"] == "insufficient_sample").all())
        self.assertTrue((low_sample["warning"] == "low_sample_warning").all())

    def test_numeric_top_decile_analysis_compares_two_groups(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        analysis = build_top_decile_numeric_analysis(frame)

        self.assertIn("top_decile_mean", analysis.columns)
        self.assertIn("non_top_decile_mean", analysis.columns)
        self.assertGreater(len(analysis), 0)

    def test_validate_final_artifacts_checks_model_and_importance_files(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            processed = Path(tmp)
            (processed / FINAL_STAGE1_ARTIFACT).write_bytes(b"stage1")
            (processed / FINAL_STAGE2_ARTIFACT).write_bytes(b"stage2")
            pd.DataFrame({"feature": ["a"], "importance_gain": [1.0]}).to_csv(
                processed / "final_feature_importance_stage1.csv", index=False
            )
            pd.DataFrame({"feature": ["b"], "importance_gain": [2.0]}).to_csv(
                processed / "final_feature_importance_stage2.csv", index=False
            )

            checks = validate_final_artifacts(processed)

        self.assertTrue(checks["final_stage1_artifact_exists"])
        self.assertTrue(checks["final_stage2_artifact_exists"])
        self.assertTrue(checks["stage1_importance_non_empty"])
        self.assertTrue(checks["stage2_importance_non_empty"])

    def test_validate_business_outputs_checks_core_contracts(self):
        frame = add_top_decile_flag(self._frame(), pred_col="pred_ltv")
        checks = validate_business_outputs(frame, build_segment_ltv_summary(frame), build_ua_decision_simulation(frame))

        self.assertTrue(checks["top_decile_count_matches"])
        self.assertTrue(checks["segment_summary_non_empty"])
        self.assertTrue(checks["ua_decision_allowed_values"])
        self.assertTrue(checks["low_sample_segments_insufficient"])


if __name__ == "__main__":
    unittest.main()
