# Model Input Validation Report

## Scope

This step validates the feature tables and fixes model-input preprocessing rules. No model training was performed.

## Schema Checks

- Train/test feature columns match after excluding target: `True`
- Target exists in train: `True`
- Target exists in test: `False`
- Target only in train: `True`
- Missing columns in test: `[]`
- Extra columns in test: `[]`

## Grain Checks

- Grain: `user_id + platform + country_tier + channel_tier + install_day`
- Train duplicate grain rows: 0
- Test duplicate grain rows: 0
- Train unique grain rows: 159,521
- Test unique grain rows: 40,105

## Raw Null / Inf Checks

| split | numeric null cells | numeric inf cells | categorical null cells | categorical null columns |
|---|---:|---:|---:|---|
| train | 152,159 | 0 | 162,796 | ['top_network', 'top_ad_placement'] |
| test | 38,224 | 0 | 40,940 | ['top_network', 'top_ad_placement'] |

## Preprocessing Rules

- `days_to_first_iap` null -> `99`
- `top_network` null -> `no_ad_network`
- `top_ad_placement` null -> `no_ad_placement`
- numeric null/inf -> `0` after applying specific sentinels

## Post-Preprocessing Checks

| split | numeric null cells | numeric inf cells | categorical null cells |
|---|---:|---:|---:|
| train | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

## Target Distribution

- Rows: 159,521
- Target nulls: 0
- Positive LTV rate: 40.45%
- Zero LTV rate: 59.55%
- Mean: 18.9218
- P50: 0.0000
- P75: 1.2754
- P95: 20.2234
- P99: 279.9575
- Max: 62046.4734

## Time-Based Split Check

- Primary split: train install_day 0-23, valid install_day 24-30
- Time split possible: `True`
- Train rows for install_day 0-23: 127,450
- Validation rows for install_day 24-30: 32,071
- Train install_day range: 0 to 30
- Test install_day range: 0 to 30
