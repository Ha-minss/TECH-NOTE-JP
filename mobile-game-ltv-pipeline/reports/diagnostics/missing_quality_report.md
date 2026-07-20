# Missingness and Raw Quality Report

## Required Columns

- train: all required columns are present.
- test: all required columns are present.

## Core Quality Checks

| check | train | test | interpretation |
|---|---:|---:|---|
| invalid_day_rows | 0 | 0 | `day_since_install` should stay within D0-D7. |
| invalid_hour_rows | 0 | 0 | `event_hour` should stay within 0-23. |
| invalid_event_type_rows | 0 | 0 | `event_type` should be session/ad_impression/iap. |
| negative_revenue_rows | 0 | 0 | `revenue_usd` should not be negative. |
| session_revenue_non_null_rows | 0 | 0 | Session rows should normally have missing revenue and can be filled with 0 for aggregation. |
| non_session_revenue_null_rows | 0 | 0 | Ad/IAP rows need revenue for early monetization features. |

## Null Rates

| split | column | null count | null rate |
|---|---|---:|---:|
| train | user_id | 0 | 0.00% |
| train | platform | 0 | 0.00% |
| train | country_tier | 0 | 0.00% |
| train | channel_tier | 0 | 0.00% |
| train | install_day | 0 | 0.00% |
| train | install_week | 0 | 0.00% |
| train | day_since_install | 0 | 0.00% |
| train | event_hour | 0 | 0.00% |
| train | event_type | 0 | 0.00% |
| train | event_name | 0 | 0.00% |
| train | product_id | 20,950,193 | 99.73% |
| train | network | 2,415,572 | 11.50% |
| train | ad_placement | 2,415,572 | 11.50% |
| train | revenue_usd | 2,359,512 | 11.23% |
| train | ltv_d8_d180 | 0 | 0.00% |
| test | user_id | 0 | 0.00% |
| test | platform | 0 | 0.00% |
| test | country_tier | 0 | 0.00% |
| test | channel_tier | 0 | 0.00% |
| test | install_day | 0 | 0.00% |
| test | install_week | 0 | 0.00% |
| test | day_since_install | 0 | 0.00% |
| test | event_hour | 0 | 0.00% |
| test | event_type | 0 | 0.00% |
| test | event_name | 0 | 0.00% |
| test | product_id | 5,177,112 | 99.71% |
| test | network | 573,872 | 11.05% |
| test | ad_placement | 573,872 | 11.05% |
| test | revenue_usd | 558,644 | 10.76% |

## Missingness Interpretation

- `product_id`: train null rate 99.73%, test null rate 99.71%. This is expected because only IAP rows carry product ids; use derived IAP features rather than raw `product_id` as a broad model feature.
- `network`: train null rate 11.50%, test null rate 11.05%. Missing values are mostly non-ad rows; use a missing category only if encoding row-level events, or aggregate unique network counts at the grain level.
- `ad_placement`: train null rate 11.50%, test null rate 11.05%. Missing values are mostly non-ad rows; aggregate placement diversity or fill missing with `no_ad_placement`.
- `revenue_usd`: train null rate 11.23%, test null rate 10.76%. Session revenue is missing by design and should be treated as 0 when aggregating D0-D7 revenue.
