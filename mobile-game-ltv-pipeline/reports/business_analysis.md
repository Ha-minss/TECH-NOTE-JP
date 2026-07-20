# Business Analysis

## Executive Summary

- Final model: `optuna_two_stage_top_capture`.
- Predicted top decile size: 3,208 users (10.00% of validation rows).
- Actual D8-D180 revenue captured by predicted top decile: 78.77%.
- UA decision simulation uses `min_users=100`; smaller segments are marked `insufficient_sample`.
- Synthetic CPI in the UA simulation is not real ad spend. It is a deterministic example table for decision workflow demonstration.

## Feature Importance Interpretation

- Importance source: `final_tuned_optuna_two_stage_model_artifact_gain_importance`.
- Stage 1 classifier importance is extracted from the final tuned classifier artifact and should be read as signals for whether a user becomes positive-LTV.
- Stage 2 regressor importance is extracted from the final tuned regressor artifact and should be read as signals for the expected amount among users modeled as future positive-LTV.

### Stage 1 Top Features

| rank | feature | gain importance |
|---:|---|---:|
| 1 | late_revenue_share | 1690.292236 |
| 2 | revenue_d4_d7 | 615.789795 |
| 3 | session_count | 194.198135 |
| 4 | last_event_day | 88.620789 |
| 5 | event_count_d4_d7 | 58.043423 |
| 6 | ad_impression_count_d4_d7 | 44.025410 |
| 7 | ad_growth_d4_d7_vs_d0_d1 | 35.762970 |
| 8 | ads_per_session | 28.005032 |
| 9 | session_count_d4_d7 | 26.748680 |
| 10 | platform_channel_freq | 25.994280 |

### Stage 2 Top Features

| rank | feature | gain importance |
|---:|---|---:|
| 1 | iap_revenue_d0_d7 | 813.587708 |
| 2 | iap_count | 138.402176 |
| 3 | avg_iap_amount | 110.165436 |
| 4 | revenue_d4_d7 | 105.507866 |
| 5 | revenue_d0_d7 | 102.309258 |
| 6 | ads_per_session | 70.591835 |
| 7 | unique_product_count | 46.990078 |
| 8 | session_count | 43.985687 |
| 9 | iap_count_d4_d7 | 36.016125 |
| 10 | te_platform_country_channel_ltv_log_mean | 34.292149 |

## Predicted Top-Decile Behavior

| feature | top decile mean | non-top mean | difference | ratio |
|---|---:|---:|---:|---:|
| ad_impression_count | 574.6328 | 73.4238 | 501.2090 | 7.83 |
| event_count | 586.4177 | 90.5853 | 495.8324 | 6.47 |
| revenue_d0_d7 | 17.5543 | 0.5012 | 17.0531 | 35.02 |
| iap_revenue_d0_d7 | 15.3444 | 0.2520 | 15.0925 | 60.90 |
| revenue_per_active_day | 4.3845 | 0.1728 | 4.2117 | 25.37 |
| last_event_day | 6.5695 | 2.7875 | 3.7820 | 2.36 |
| active_days | 6.8974 | 3.1880 | 3.7095 | 2.16 |
| ad_revenue_d0_d7 | 2.2098 | 0.2492 | 1.9606 | 8.87 |

Top-decile users are the users ranked highest by predicted LTV, not users known to be high-value in advance. Their profile is therefore useful for UA targeting hypotheses and lifecycle prioritization.

## UA Decision Simulation

Main table includes only segments with at least 100 users, sorted by predicted ROAS. Segments below the threshold are not assigned scale/keep/reduce decisions.

| country_tier | channel_tier | users | predicted LTV | actual LTV | synthetic CPI | predicted ROAS | actual ROAS | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NL | bb16a88d | 141 | 38.9756 | 61.6187 | 0.8075 | 48.27 | 76.31 | scale_up |
| ES | 92247aa9 | 108 | 22.7146 | 77.0867 | 0.8075 | 28.13 | 95.46 | scale_up |
| TR | bb16a88d | 115 | 18.5414 | 53.4072 | 0.8075 | 22.96 | 66.14 | scale_up |
| IT | bb16a88d | 128 | 12.4819 | 18.1852 | 0.8075 | 15.46 | 22.52 | scale_up |
| PL | 92247aa9 | 301 | 11.2661 | 12.3424 | 0.8075 | 13.95 | 15.28 | scale_up |
| IT | 92247aa9 | 178 | 9.2909 | 26.0930 | 0.8075 | 11.51 | 32.31 | scale_up |
| UK | 92247aa9 | 169 | 7.4533 | 30.5400 | 0.8075 | 9.23 | 37.82 | scale_up |
| UA | 92247aa9 | 1268 | 6.9185 | 7.4858 | 0.8075 | 8.57 | 9.27 | scale_up |
| PL | bb16a88d | 142 | 6.0868 | 8.2477 | 0.8075 | 7.54 | 10.21 | scale_up |
| OTHER | 92247aa9 | 2378 | 5.9434 | 10.4629 | 0.8075 | 7.36 | 12.96 | scale_up |

### Low-Sample Warning Segments

These segments may show high predicted ROAS, but they are marked `insufficient_sample` because their validation sample size is below the minimum threshold.

| country_tier | channel_tier | users | predicted LTV | synthetic CPI | predicted ROAS | decision | warning |
|---|---|---:|---:|---:|---:|---|---|
| OTHER | 0a0ae9c4 | 71 | 65.4857 | 0.8075 | 81.10 | insufficient_sample | low_sample_warning |
| NL | other | 1 | 39.7695 | 0.6800 | 58.48 | insufficient_sample | low_sample_warning |
| NL | 0a0ae9c4 | 4 | 15.9887 | 0.8075 | 19.80 | insufficient_sample | low_sample_warning |
| JP | 92247aa9 | 96 | 24.5455 | 1.8050 | 13.60 | insufficient_sample | low_sample_warning |
| ES | bb16a88d | 89 | 9.3040 | 0.8075 | 11.52 | insufficient_sample | low_sample_warning |
| KZ | 92247aa9 | 78 | 7.6355 | 0.8075 | 9.46 | insufficient_sample | low_sample_warning |
| BY | other | 2 | 5.5484 | 0.6800 | 8.16 | insufficient_sample | low_sample_warning |
| IT | 0a0ae9c4 | 4 | 5.0451 | 0.8075 | 6.25 | insufficient_sample | low_sample_warning |
| KR | 92247aa9 | 78 | 8.3727 | 1.4725 | 5.69 | insufficient_sample | low_sample_warning |
| IT | other | 1 | 3.5405 | 0.6800 | 5.21 | insufficient_sample | low_sample_warning |

## Validation Checks

| check | passed |
|---|---|
| top_decile_count_matches | True |
| segment_summary_non_empty | True |
| predicted_roas_valid | True |
| ua_decision_allowed_values | True |
| low_sample_segments_insufficient | True |
| high_sample_segments_have_business_decisions | True |
| synthetic_cpi_positive | True |
| final_stage1_artifact_exists | True |
| final_stage2_artifact_exists | True |
| stage1_importance_non_empty | True |
| stage2_importance_non_empty | True |
