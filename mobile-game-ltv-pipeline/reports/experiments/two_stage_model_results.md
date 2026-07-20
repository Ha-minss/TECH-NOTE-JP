# Two-Stage LTV Model Results

## Scope

This compares a two-stage XGBoost model against existing baselines on the same time-based validation split. It does not use Optuna, OOF, random KFold, or LightGBM.

## Validation Split

- Train split: install_day 0 to 23 (127,450 rows)
- Valid split: install_day 24 to 30 (32,071 rows)

## Final Metrics

| model | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_stage_xgb_velocity_ratio_features | 5.0717 | 80.0574 | 0.5477 | 0.8053 | 97.94% | 77.51% | 7.75 |
| two_stage_xgb_target_encoding_features | 5.0390 | 80.2077 | 0.5447 | 0.8051 | 98.07% | 77.66% | 7.76 |

## Comparison Reference

| group | model | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture | top-decile lift |
|---|---|---:|---:|---:|---:|---:|---:|
| single_stage_xgboost | xgb_current_full | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 76.32% | 7.63 |
| single_stage_xgboost | xgb_velocity_ratio_features | 5.1225 | 82.5995 | 0.5456 | 0.8032 | 76.60% | 7.66 |
| single_stage_xgboost | xgb_target_encoding_features | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 76.41% | 7.64 |
| linear | ridge_log_linear | 6.2176 | 97.1298 | 0.6645 | 0.7686 | 74.08% | 7.41 |
| baseline | early_revenue_multiplier | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 72.06% | 7.20 |
| two_stage | two_stage_xgb_velocity_ratio_features | 5.0717 | 80.0574 | 0.5477 | 0.8053 | 77.51% | 7.75 |
| two_stage | two_stage_xgb_target_encoding_features | 5.0390 | 80.2077 | 0.5447 | 0.8051 | 77.66% | 7.76 |

## Stage 1 Diagnostics

| model | ROC-AUC | PR-AUC | LogLoss | Brier | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_stage_xgb_velocity_ratio_features | 0.9715 | 0.9662 | 0.1953 | 0.0556 | 0.9316 | 0.8784 | 0.9042 |
| two_stage_xgb_target_encoding_features | 0.9714 | 0.9661 | 0.1952 | 0.0556 | 0.9302 | 0.8798 | 0.9043 |

## Stage 2 Diagnostics

| model | positive valid rows | positive-only MAE | positive-only RMSE | positive-only RMSLE |
|---|---:|---:|---:|---:|
| two_stage_xgb_velocity_ratio_features | 12,604 | 12.5590 | 126.6939 | 0.7863 |
| two_stage_xgb_target_encoding_features | 12,604 | 12.4758 | 126.7050 | 0.7830 |

## Questions

- Is two-stage better than best single-stage RMSLE 0.5404? No; best two-stage RMSLE is 0.544686 from `two_stage_xgb_target_encoding_features`.
- Is two-stage better than best single-stage top 10% revenue capture 76.60%? Yes; best two-stage capture is 77.66% from `two_stage_xgb_target_encoding_features`.
- Zero-heavy target impact: compare RMSLE, Spearman, and top-decile capture together; final prediction multiplies propensity by conditional positive value, so it may improve calibration while sometimes softening high-value ranking.
- Stage 1 quality: ROC-AUC/PR-AUC show whether positive LTV users are separable before amount prediction.
- Stage 2 stability: positive-only MAE/RMSE/RMSLE describe amount prediction only among positive valid users, separated from the zero classification task.

## Prediction Checks

| model | rows match valid | null predictions | inf predictions | negative predictions | probability out of range |
|---|---|---:|---:|---:|---:|
| two_stage_xgb_velocity_ratio_features | True | 0 | 0 | 0 | 0 |
| two_stage_xgb_target_encoding_features | True | 0 | 0 | 0 | 0 |
