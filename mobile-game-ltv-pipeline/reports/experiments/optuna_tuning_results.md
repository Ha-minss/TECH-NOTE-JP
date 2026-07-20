# Optuna XGBoost Tuning Results

## Scope

This tunes only the two rolling-validated candidate models. It does not use AutoML, random KFold, OOF, or LightGBM.

## Tuning Setup

- Trials per study: 30
- Early stopping rounds: 50
- Tuning folds: install_day 0-13 -> 14-16, 0-16 -> 17-19, 0-19 -> 20-23.
- Final holdout: install_day 0-23 -> 24-30, not used in Optuna objective.
- Feature set: target_encoding_features; target encodings are fit within each train fold only.

## Final Holdout Metrics

| model | objective | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| optuna_single_stage_rmsle | rmsle | 4.9102 | 80.5268 | 0.5307 | 0.8004 | 98.91% | 76.93% | 7.69 |
| optuna_two_stage_top_capture | top_capture | 4.7472 | 76.5951 | 0.5272 | 0.7970 | 98.32% | 78.77% | 7.88 |

## Comparison Reference

| group | model | MAE | RMSE | RMSLE | Spearman | top 10% revenue capture | top-decile lift |
|---|---|---:|---:|---:|---:|---:|---:|
| untuned_xgboost | xgb_current_full | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 76.32% | 7.63 |
| untuned_xgboost | xgb_target_encoding_features | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 76.41% | 7.64 |
| untuned_two_stage | two_stage_xgb_target_encoding_features | 5.0390 | 80.2077 | 0.5447 | 0.8051 | 77.66% | 7.76 |
| linear | ridge_log_linear | 6.2176 | 97.1298 | 0.6645 | 0.7686 | 74.08% | 7.41 |
| linear | elasticnet_log_linear | 6.2380 | 96.3264 | 0.6695 | 0.7713 | 74.22% | 7.42 |
| baseline | early_revenue_multiplier | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 72.06% | 7.20 |
| optuna_tuned | optuna_single_stage_rmsle | 4.9102 | 80.5268 | 0.5307 | 0.8004 | 76.93% | 7.69 |
| optuna_tuned | optuna_two_stage_top_capture | 4.7472 | 76.5951 | 0.5272 | 0.7970 | 78.77% | 7.88 |

## Questions

- Did tuned single-stage improve RMSLE 0.5404? Yes; delta=0.009706.
- Did tuned two-stage improve top 10% capture 77.66%? Yes; delta=0.011134.
- Rolling objective vs holdout consistency: best CV single-stage mean RMSLE=0.521878; final holdout RMSLE=0.530694. Best CV two-stage mean capture=82.28%; final holdout capture=78.77%.
- RMSLE and top-capture objectives select different model families by design: single-stage for RMSLE, two-stage for business ranking.
- If improvements are small, the bottleneck is more likely feature/model structure and target formulation than basic hyperparameter settings.

## Best Params

```json
{
  "optuna_single_stage_rmsle": {
    "n_estimators": 1220,
    "learning_rate": 0.04297694391923703,
    "max_depth": 5,
    "min_child_weight": 14,
    "subsample": 0.8995530403016572,
    "colsample_bytree": 0.8122387609410768,
    "reg_lambda": 3.621129089013974,
    "reg_alpha": 3.5544483564378737,
    "gamma": 0.2641481533202849,
    "tree_method": "hist",
    "random_state": 42
  },
  "optuna_two_stage_top_capture": {
    "n_estimators": 1155,
    "learning_rate": 0.058272498861928146,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.9129013486744039,
    "colsample_bytree": 0.994854413722088,
    "reg_lambda": 4.542141433075535,
    "reg_alpha": 0.8103509281058586,
    "gamma": 0.16634071567742276,
    "tree_method": "hist",
    "random_state": 42
  }
}
```
