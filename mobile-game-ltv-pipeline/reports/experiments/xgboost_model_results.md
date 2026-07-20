# XGBoost Log-Target Baseline Results

## Scope

This is the first nonlinear tree-based supervised ML baseline. It uses only time-based validation and does not use LightGBM, Optuna, OOF, KFold, or two-stage modeling.

## Validation Split

- Train split: install_day 0 to 23 (127,450 rows)
- Valid split: install_day 24 to 30 (32,071 rows)

## Feature Setup

- Categorical: `['platform', 'country_tier', 'channel_tier', 'top_network', 'top_ad_placement']`
- Numeric feature count: 32
- Excluded from model features: `['user_id', 'install_day', 'ltv_d8_d180']`
- Numeric preprocessing: p1/p99 clipping on train split; no StandardScaler
- Categorical preprocessing: OneHotEncoder(handle_unknown='ignore')
- Model: XGBRegressor objective=`reg:squarederror`, tree_method=`hist`, n_estimators=500, learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, early stopping on valid split.
- Best iteration: 499
- Best validation log-RMSE: 0.5477659953682207

## Metrics

| model | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost_log_target | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 98.66% | 76.32% | 7.63 |

## Linear Model Reference

| model | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture |
|---|---:|---:|---:|---:|---:|
| ridge_log_linear | 6.2176 | 97.1298 | 0.6645 | 0.7686 | 74.08% |
| elasticnet_log_linear | 6.2380 | 96.3264 | 0.6695 | 0.7713 | 74.22% |

## Baseline Reference

| baseline | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture |
|---|---:|---:|---:|---:|---:|
| global_mean | 24.1797 | 104.0161 | 2.7500 | 0.0000 | 6.51% |
| segment_mean | 16.6481 | 105.0316 | 2.0948 | 0.1139 | 17.15% |
| early_revenue_multiplier | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 72.06% |

## Prediction Checks

| model | rows match valid | null predictions | inf predictions | negative predictions |
|---|---|---:|---:|---:|
| xgboost_log_target | True | 0 | 0 | 0 |
