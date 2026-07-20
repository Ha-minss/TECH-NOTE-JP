# Final Model Card

## Selected Model

- Model: `optuna_two_stage_top_capture`
- Architecture: two-stage XGBoost. Stage 1 estimates positive LTV probability; Stage 2 estimates conditional positive LTV; final prediction is `p_positive * predicted_ltv_if_positive`.
- Validation split: train install_day 0-23, valid install_day 24-30.
- Selection reason: best overall final holdout result after rolling validation and Optuna tuning, with stronger RMSLE, MAE/RMSE, and top-decile revenue capture than previous candidates.
- Persisted artifacts: `data/processed/final_optuna_two_stage_stage1_classifier.pkl`, `data/processed/final_optuna_two_stage_stage2_regressor.pkl`.

## Holdout Metrics

- MAE: 4.7472
- RMSE: 76.5951
- RMSLE: 0.5272
- Spearman correlation: 0.7970
- Top 10% revenue capture: 78.77%
- Positive LTV rate in predicted top decile: 98.32%

The positive-rate metric is not an accuracy score. It means the share of users inside the model's predicted top decile whose actual D8-D180 LTV is greater than zero.

## Parameters

```json
{
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
```

## Feature Importance Note

Feature importance is extracted from the saved final tuned model artifacts using XGBoost gain importance: `final_tuned_optuna_two_stage_model_artifact_gain_importance`.
This step does not rerun Optuna or reselect models; it reconstructs the already selected final model artifact from saved best parameters for persistence and interpretation.
