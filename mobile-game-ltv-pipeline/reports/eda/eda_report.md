# Mobile Game LTV EDA

## Dataset Grain

- Raw grain: one row per user event during days 0-7 after install.
- Modeling grain: one row per `user_id`; target is D8-D180 revenue in train only.
- Prediction grain: one row per unique test `user_id`.

## Shape

| split | event rows | users | rows/user median | rows/user p95 | rows/user max |
|---|---:|---:|---:|---:|---:|
| train | 21,006,238 | 75,464 | 32 | 1,166 | 178,821 |
| test | 5,192,340 | 31,399 | 9 | 860 | 96,075 |

## Target Summary

- Positive D8-D180 LTV user rate: 40.94%
- Mean target: $16.3491
- Median target: $0.0000
- P95 target: $20.4390
- P99 target: $200.2790
- Max target: $24,234.8997
- Actual top 10% users by target capture: 95.14%

## First-7-Day Behavior Summary

| metric | train median | train p95 | test median | test p95 |
|---|---:|---:|---:|---:|
| event_count | 32.0000 | 1,166.0000 | 9.0000 | 860.0000 |
| session_count | 6.0000 | 103.0000 | 3.0000 | 63.0000 |
| iap_count | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| ad_impression_count | 8.0000 | 1,125.0000 | 1.0000 | 830.1000 |
| active_days | 5.0000 | 8.0000 | 2.0000 | 8.0000 |
| revenue_d0_d7 | 0.1667 | 12.8481 | 0.0178 | 4.9404 |
| iap_revenue_d0_d7 | 0.0000 | 7.3734 | 0.0000 | 0.0000 |
| ad_revenue_d0_d7 | 0.1284 | 3.3984 | 0.0086 | 2.0471 |

## Quality Checks

| check | train | test | interpretation |
|---|---:|---:|---|
| negative_revenue_rows | 0 | 0 | Revenue should not be negative. |
| session_revenue_non_null_rows | 0 | 0 | Session rows should not carry revenue. |
| non_session_revenue_null_rows | 0 | 0 | IAP/ad rows should usually carry revenue. |
| invalid_day_rows | 0 | 0 | Feature window must stay within D0-D7. |
| invalid_hour_rows | 0 | 0 | Event hour must be 0-23. |
| invalid_event_type_rows | 0 | 0 | Event type enum should match session/iap/ad_impression. |

### Static Field Consistency

Users should not change platform, country, channel, or install date across their event rows.

| field | inconsistent train users | inconsistent test users |
|---|---:|---:|
| platform | 16,216 | 1,578 |
| country_tier | 45,669 | 5,957 |
| channel_tier | 35,758 | 3,655 |
| install_day | 46,462 | 5,904 |
| install_week | 29,425 | 3,395 |
- Train users with inconsistent repeated target values: 34,191

## Top Categories

### Train Event Types

- ad_impression: 18,590,666 rows (88.50%)
- session: 2,359,512 rows (11.23%)
- iap: 56,060 rows (0.27%)

### Test Event Types

- ad_impression: 4,618,468 rows (88.95%)
- session: 558,644 rows (10.76%)
- iap: 15,228 rows (0.29%)

## Modeling Implications

- The raw data is event-level and must be aggregated to user-level before training.
- Early revenue is a legitimate feature because it occurs in D0-D7 while the target is D8-D180.
- The target is expected to be zero-inflated and long-tailed, so log1p regression and two-stage modeling should be evaluated.
- Ranking metrics such as top-decile revenue capture matter alongside RMSE/RMSLE because UA decisions care about finding high-value users and segments.
- The dataset has `channel_tier`, not full campaign cost. ROAS or budget recommendations should be framed as channel/platform/country segment simulation unless real CPI is added.

## Recommended Automated Tests

- Required columns exist for train/test/submission.
- `day_since_install` is always between 0 and 7.
- `event_hour` is always between 0 and 23.
- `event_type` belongs to session, iap, or ad_impression.
- Static user attributes are constant within each user.
- Train target is constant within each user's repeated event rows.
- Feature builder outputs exactly one row per `user_id`.
- Submission contains exactly the sample submission user ids with non-negative predictions.
