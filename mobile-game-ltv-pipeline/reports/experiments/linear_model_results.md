# Linear Log-Target Model Results

## Scope

This is the first supervised ML baseline after non-ML baselines. It uses only time-based validation and does not use OOF, random KFold, LightGBM, XGBoost, Optuna, or two-stage modeling.

## Validation Split

- Train split: install_day 0 to 23 (127,450 rows)
- Valid split: install_day 24 to 30 (32,071 rows)

## Feature Setup

- Categorical: `['platform', 'country_tier', 'channel_tier', 'top_network', 'top_ad_placement']`
- Numeric feature count: 32
- Excluded from model features: `['user_id', 'install_day', 'ltv_d8_d180']`
- Numeric preprocessing: p1/p99 clipping on train split, then StandardScaler
- Categorical preprocessing: OneHotEncoder(handle_unknown='ignore')
- Target transform: log1p; predictions use expm1_then_clip_at_zero

## Metrics

| model | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_log_linear | 6.2176 | 97.1298 | 0.6645 | 0.7686 | 95.45% | 74.08% | 7.41 |
| elasticnet_log_linear | 6.2380 | 96.3264 | 0.6695 | 0.7713 | 95.11% | 74.22% | 7.42 |

## Baseline Reference

| baseline | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|
| global_mean | 24.1797 | 104.0161 | 2.7500 | 0.0000 | 6.51% | 0.65 |
| segment_mean | 16.6481 | 105.0316 | 2.0948 | 0.1139 | 17.15% | 1.71 |
| early_revenue_multiplier | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 72.06% | 7.20 |

## Prediction Checks

| model | rows match valid | null predictions | inf predictions | negative predictions |
|---|---|---:|---:|---:|
| ridge_log_linear | True | 0 | 0 | 0 |
| elasticnet_log_linear | True | 0 | 0 | 0 |
