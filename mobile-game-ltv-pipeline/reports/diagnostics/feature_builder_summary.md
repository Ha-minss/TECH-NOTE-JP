# Feature Builder Summary

## Scope

The modeling grain is fixed as `user_id + platform + country_tier + channel_tier + install_day`. No model training was performed.

## Outputs

- Train feature rows: 159,521
- Test feature rows: 40,105
- Dropped train target-collision groups: 405
- Train feature columns: 40
- Test feature columns: 39
- Train parquet: `C:\dev\Codex\mobile-game-ltv-pipeline\data\processed\train_features.parquet`
- Test parquet: `C:\dev\Codex\mobile-game-ltv-pipeline\data\processed\test_features.parquet`
- Collision log: `C:\dev\Codex\mobile-game-ltv-pipeline\data\processed\dropped_collision_groups.csv`

## Feature Rules

- `revenue_usd` nulls are filled as 0 before aggregation.
- `product_id` is not emitted as a raw categorical feature; it only contributes to IAP aggregate features.
- `network` and `ad_placement` are not emitted as raw row-level features; they contribute to unique counts, top categories, event shares, and revenue shares.
- Remaining target-collision groups are removed from train features and saved to the collision log.

## Feature Columns

```text
user_id
platform
country_tier
channel_tier
install_day
install_week
event_count
session_count
ad_impression_count
iap_count
active_days
last_event_day
revenue_d0_d7
ad_revenue_d0_d7
iap_revenue_d0_d7
revenue_d0
revenue_d1
revenue_d2_d3
revenue_d4_d7
unique_network_count
unique_ad_placement_count
top_network
top_ad_placement
top_network_event_share
top_ad_placement_event_share
top_network_revenue_share
top_ad_placement_revenue_share
ads_per_session
ad_revenue_per_ad
ad_revenue_per_session
early_payer_flag
days_to_first_iap
unique_product_count
avg_iap_amount
max_iap_amount
first_event_hour
last_event_hour
most_active_hour
active_hour_count
ltv_d8_d180
```

## Quick Distribution Check

| split | rows | median events | p95 events | zero D0-D7 revenue rate | early payer rate |
|---|---:|---:|---:|---:|---:|
| train | 159,521 | 7 | 764 | 48.68% | 4.62% |
| test | 40,105 | 7 | 764 | 48.78% | 4.69% |
