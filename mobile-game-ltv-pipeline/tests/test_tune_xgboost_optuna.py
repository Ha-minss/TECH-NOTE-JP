import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.tune_xgboost_optuna import (
    BASELINE_SINGLE_STAGE_RMSLE,
    BASELINE_TWO_STAGE_TOP_CAPTURE,
    FINAL_HOLDOUT,
    TARGET_FEATURE_SET,
    TUNING_FOLDS,
    build_best_params_payload,
    build_param_dict,
    build_valid_predictions_frame,
    objective_value_from_metrics,
    prepare_target_encoding_splits,
    summarize_best_metrics,
)
from experiments.run_feature_ablation import build_time_bucket_features


class FakeTrial:
    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


class OptunaTuningTests(unittest.TestCase):
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

    def test_tuning_folds_exclude_final_holdout(self):
        self.assertEqual([f["fold"] for f in TUNING_FOLDS], [1, 2, 3])
        self.assertEqual(FINAL_HOLDOUT["fold"], 4)
        self.assertEqual(FINAL_HOLDOUT["valid_min"], 24)

    def test_param_dict_uses_requested_search_space_and_fixed_values(self):
        params = build_param_dict(FakeTrial())

        self.assertEqual(params["n_estimators"], 300)
        self.assertEqual(params["learning_rate"], 0.01)
        self.assertEqual(params["max_depth"], 3)
        self.assertEqual(params["min_child_weight"], 1)
        self.assertEqual(params["tree_method"], "hist")
        self.assertEqual(params["random_state"], 42)
        self.assertIn("reg_alpha", params)
        self.assertIn("gamma", params)

    def test_objective_value_mapping(self):
        metrics = {"rmsle": 0.5, "top_10pct_revenue_capture": 0.8}

        self.assertEqual(objective_value_from_metrics(metrics, "rmsle"), 0.5)
        self.assertEqual(objective_value_from_metrics(metrics, "top_capture"), -0.8)

    def test_target_encoding_splits_use_train_only_fallback(self):
        base = self._frame()
        bucket = build_time_bucket_features(self._raw_events())
        train_split = base[base["install_day"].between(0, 13)].copy()
        valid_split = base[base["install_day"].between(14, 16)].copy()
        valid_split.loc[:, "ltv_d8_d180"] = 999999.0
        valid_split.loc[:, "channel_tier"] = "unseen_channel"

        train_out, valid_out, metadata = prepare_target_encoding_splits(train_split, valid_split, bucket)

        expected = float(np.log1p(train_split["ltv_d8_d180"].clip(lower=0)).mean())
        self.assertEqual(metadata["feature_set"], TARGET_FEATURE_SET)
        self.assertAlmostEqual(valid_out["te_channel_ltv_log_mean"].iloc[0], expected)

    def test_prediction_frame_contains_both_tuned_models(self):
        valid = self._frame().tail(3).copy()
        predictions = {
            "optuna_single_stage_rmsle": pd.Series([1.0, 2.0, 3.0]),
            "optuna_two_stage_top_capture": pd.Series([0.5, 1.5, 2.5]),
        }

        out = build_valid_predictions_frame(valid, predictions)

        self.assertEqual(len(out), 3)
        self.assertIn("pred_optuna_single_stage_rmsle", out.columns)
        self.assertIn("pred_optuna_two_stage_top_capture", out.columns)
        self.assertFalse(out.filter(like="pred_").isna().any().any())

    def test_best_metrics_summary_compares_against_baselines(self):
        metrics = pd.DataFrame(
            [
                {"model": "optuna_single_stage_rmsle", "objective": "rmsle", "rmsle": 0.53, "top_10pct_revenue_capture": 0.75},
                {"model": "optuna_two_stage_top_capture", "objective": "top_capture", "rmsle": 0.55, "top_10pct_revenue_capture": 0.78},
            ]
        )

        summary = summarize_best_metrics(metrics)

        self.assertTrue(summary["single_stage_improved_rmsle"])
        self.assertTrue(summary["two_stage_improved_top_capture"])
        self.assertEqual(BASELINE_SINGLE_STAGE_RMSLE, 0.5404)
        self.assertEqual(BASELINE_TWO_STAGE_TOP_CAPTURE, 0.7766)

    def test_best_params_payload_is_json_serializable(self):
        payload = build_best_params_payload(
            {
                "optuna_single_stage_rmsle": {"learning_rate": 0.01},
                "optuna_two_stage_top_capture": {"learning_rate": 0.02},
            },
            {"n_trials": 30},
        )

        self.assertEqual(payload["settings"]["n_trials"], 30)
        self.assertIn("optuna_single_stage_rmsle", payload["best_params"])


if __name__ == "__main__":
    unittest.main()
