import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.build_features import GRAIN_COLUMNS, build_feature_frame


class FeatureBuilderTests(unittest.TestCase):
    def test_builds_one_row_per_user_context_and_drops_target_collisions(self):
        raw = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 3,
                    "install_week": 0,
                    "day_since_install": 0,
                    "event_hour": 9,
                    "event_type": "session",
                    "event_name": "session_start",
                    "product_id": None,
                    "network": None,
                    "ad_placement": None,
                    "revenue_usd": None,
                    "ltv_d8_d180": 10.0,
                },
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 3,
                    "install_week": 0,
                    "day_since_install": 1,
                    "event_hour": 10,
                    "event_type": "ad_impression",
                    "event_name": "ad_reward",
                    "product_id": None,
                    "network": "net_a",
                    "ad_placement": "reward_home",
                    "revenue_usd": 0.5,
                    "ltv_d8_d180": 10.0,
                },
                {
                    "user_id": 1,
                    "platform": "android",
                    "country_tier": "US",
                    "channel_tier": "paid",
                    "install_day": 3,
                    "install_week": 0,
                    "day_since_install": 2,
                    "event_hour": 10,
                    "event_type": "iap",
                    "event_name": "af_purchase",
                    "product_id": "coin_pack",
                    "network": None,
                    "ad_placement": None,
                    "revenue_usd": 4.0,
                    "ltv_d8_d180": 10.0,
                },
                {
                    "user_id": 2,
                    "platform": "ios",
                    "country_tier": "KR",
                    "channel_tier": "organic",
                    "install_day": 4,
                    "install_week": 0,
                    "day_since_install": 0,
                    "event_hour": 4,
                    "event_type": "session",
                    "event_name": "session_start",
                    "product_id": None,
                    "network": None,
                    "ad_placement": None,
                    "revenue_usd": None,
                    "ltv_d8_d180": 1.0,
                },
                {
                    "user_id": 2,
                    "platform": "ios",
                    "country_tier": "KR",
                    "channel_tier": "organic",
                    "install_day": 4,
                    "install_week": 0,
                    "day_since_install": 1,
                    "event_hour": 5,
                    "event_type": "session",
                    "event_name": "session_start",
                    "product_id": None,
                    "network": None,
                    "ad_placement": None,
                    "revenue_usd": None,
                    "ltv_d8_d180": 2.0,
                },
            ]
        )

        features, dropped = build_feature_frame(raw, has_target=True)

        self.assertEqual(len(features), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(features.loc[0, GRAIN_COLUMNS].to_dict()["user_id"], 1)
        self.assertAlmostEqual(features.loc[0, "revenue_d0_d7"], 4.5)
        self.assertAlmostEqual(features.loc[0, "ad_revenue_d0_d7"], 0.5)
        self.assertAlmostEqual(features.loc[0, "iap_revenue_d0_d7"], 4.0)
        self.assertEqual(features.loc[0, "early_payer_flag"], 1)
        self.assertEqual(features.loc[0, "days_to_first_iap"], 2)
        self.assertEqual(features.loc[0, "unique_product_count"], 1)
        self.assertEqual(features.loc[0, "top_network"], "net_a")
        self.assertEqual(features.loc[0, "top_ad_placement"], "reward_home")
        self.assertAlmostEqual(features.loc[0, "top_network_revenue_share"], 1.0)
        self.assertAlmostEqual(features.loc[0, "top_ad_placement_revenue_share"], 1.0)
        self.assertNotIn("product_id", features.columns)
        self.assertNotIn("network", features.columns)
        self.assertNotIn("ad_placement", features.columns)

    def test_builds_test_features_without_target_or_collision_drop(self):
        raw = pd.DataFrame(
            [
                {
                    "user_id": 9,
                    "platform": "android",
                    "country_tier": "OTHER",
                    "channel_tier": "paid",
                    "install_day": 8,
                    "install_week": 1,
                    "day_since_install": 0,
                    "event_hour": 12,
                    "event_type": "session",
                    "event_name": "session_start",
                    "product_id": None,
                    "network": None,
                    "ad_placement": None,
                    "revenue_usd": None,
                }
            ]
        )

        features, dropped = build_feature_frame(raw, has_target=False)

        self.assertEqual(len(features), 1)
        self.assertTrue(dropped.empty)
        self.assertNotIn("ltv_d8_d180", features.columns)
        self.assertEqual(features.loc[0, "revenue_d0_d7"], 0.0)
        self.assertEqual(features.loc[0, "event_count"], 1)


if __name__ == "__main__":
    unittest.main()
