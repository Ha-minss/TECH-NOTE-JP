# EDA Findings: Mobile Game LTV Forecasting

## Executive Summary

This dataset is strong for an LTV portfolio project because the raw grain is event-level and the target is true future revenue: D8-D180 USD revenue after observing D0-D7 behavior.

The main modeling opportunity is not "run XGBoost." It is converting noisy, high-volume event logs into user/context-level behavioral features, then handling a zero-inflated, long-tailed LTV target.

The main data risk is identity grain. The files do not behave exactly as the dataset description says: `user_id` alone is not stable across the full train/test files.

## Shape

| split | event rows | observed user_id count | rows/user median | rows/user p95 | rows/user max |
|---|---:|---:|---:|---:|---:|
| train | 21,006,238 | 75,464 | 32 | 1,166 | 178,821 |
| test | 5,192,340 | 31,399 | 9 | 860 | 96,075 |

The test event file has 31,399 unique `user_id`s, but `sample_submission.csv` has 40,112 unique `user_id`s. That is a material mismatch for Kaggle-style submission if predictions are expected once per `user_id`.

## Identity Grain Issue

`user_id` alone is not a clean entity key in the actual files.

Full-file key consistency checks:

| key candidate | train groups | train groups with multiple target values | test groups |
|---|---:|---:|---:|
| `user_id` | 75,464 | 34,191 | 31,399 |
| `user_id + install_day` | 146,129 | 6,953 | 38,649 |
| `user_id + install_week` | 109,560 | 23,309 | 35,085 |
| `user_id + platform + country_tier + channel_tier + install_day` | 159,926 | 405 | 40,105 |

Implication: for modeling and EDA, `user_id` alone is unsafe. The most defensible analytical grain is closer to:

```text
user_id + platform + country_tier + channel_tier + install_day
```

This almost matches the sample submission row count, but not exactly: test has 40,105 such context groups while sample submission has 40,112 rows. The source file/version should be checked before treating this as a real competition submission pipeline.

For a portfolio project, this is useful: it shows why a validation step must verify grain before feature engineering.

## Target Distribution

Train target is highly sparse and long-tailed.

| metric | value |
|---|---:|
| positive D8-D180 LTV rate | 40.94% |
| mean target | $16.3491 |
| median target | $0.0000 |
| p95 target | $20.4390 |
| p99 target | $200.2790 |
| max target | $24,234.8997 |
| actual revenue captured by top 10% target users | 95.14% |

Modeling implication: RMSE alone will be misleading. Use log-target modeling, two-stage modeling, and ranking metrics such as top-decile revenue capture.

## Event Mix

| split | ad impression | session | IAP |
|---|---:|---:|---:|
| train | 88.50% | 11.23% | 0.27% |
| test | 88.95% | 10.76% | 0.29% |

`event_name` is very simple:

- `ad_reward`: about 88.5-88.9% of rows
- `session_start`: about 10.8-11.2% of rows
- `af_purchase`: about 0.27-0.29% of rows

Modeling implication: ad behavior dominates row volume. Feature engineering should normalize ad volume by sessions and active days so heavy users do not dominate purely because of event count.

## First-7-Day Behavior

| metric | train median | train p95 | test median | test p95 |
|---|---:|---:|---:|---:|
| event_count | 32 | 1,166 | 9 | 860 |
| session_count | 6 | 103 | 3 | 63 |
| iap_count | 0 | 1 | 0 | 0 |
| ad_impression_count | 8 | 1,125 | 1 | 830 |
| active_days | 5 | 8 | 2 | 8 |
| revenue_d0_d7 | $0.1667 | $12.8481 | $0.0178 | $4.9404 |
| iap_revenue_d0_d7 | $0.0000 | $7.3734 | $0.0000 | $0.0000 |
| ad_revenue_d0_d7 | $0.1284 | $3.3984 | $0.0086 | $2.0471 |

Train users look more engaged and monetized than test users. This may be expected from the split, but it should be treated as distribution shift and monitored.

## Data Quality Checks

Good checks:

- No negative revenue.
- Session rows do not carry revenue.
- IAP/ad rows have revenue populated.
- `day_since_install` stays within 0-7.
- `event_hour` stays within 0-23.
- `event_type` is limited to `session`, `iap`, and `ad_impression`.

Risk checks:

- `user_id` is not a stable entity key across the full file.
- Sample submission and test event ids do not align by `user_id`.
- Repeated static fields are not constant under `user_id`; this is likely because the true analytical grain includes install/context columns.

## Feature Engineering Direction

The feature set should emphasize behavior over raw event count:

- engagement: sessions, active days, last active day, session density
- monetization: D0-D7 total revenue, ad revenue, IAP revenue, revenue by day bucket
- ad behavior: rewarded ad count, ads per session, ad revenue per ad impression
- IAP behavior: early payer flag, purchase count, days to first IAP
- timing: first event hour, event-hour entropy, activity recency
- concentration: D0 revenue share, D4-D7 revenue share, revenue growth from early to late window
- context: platform, country tier, channel tier, install week

## Modeling Direction

Recommended experiment ladder:

1. Segment mean baseline using platform/country/channel/install week.
2. Basic aggregate features with Ridge or ElasticNet on `log1p(ltv)`.
3. Advanced behavioral features with gradient boosting.
4. Optuna-tuned gradient boosting optimizing RMSLE while monitoring top-decile capture.
5. Two-stage model:
   - classifier predicts `ltv_d8_d180 > 0`
   - regressor predicts `log1p(ltv_d8_d180)` among positive users
   - final prediction is probability times conditional value

Recommended metrics:

- MAE and RMSE on raw USD
- RMSLE on long-tailed target
- top 10% revenue capture
- top-decile lift
- segment-level predicted vs actual LTV error

## Portfolio Positioning

The strongest story is:

> I found that `user_id` was not a stable modeling grain, validated the dataset before modeling, built user/context-level behavioral features from 26M event rows, and designed an LTV pipeline for a zero-inflated long-tail revenue target with ranking metrics tied to UA decisions.

