# Model Selection Summary

The final `make all` pipeline intentionally excludes model-selection experiments. The reports below are retained as evidence for why the final model was selected, and can be regenerated with `make experiments` when needed.

## Experiment Reports

- [Baseline results](experiments/baseline_results.md)
- [Linear model results](experiments/linear_model_results.md)
- [XGBoost baseline results](experiments/xgboost_model_results.md)
- [Feature engineering results](experiments/feature_engineering_results.md)
- [Two-stage model results](experiments/two_stage_model_results.md)
- [Rolling time validation results](experiments/rolling_validation_results.md)
- [Optuna tuning results](experiments/optuna_tuning_results.md)

## Diagnostic Reports

- [Raw data validation](diagnostics/raw_data_validation_report.md)
- [Feature builder summary](diagnostics/feature_builder_summary.md)
- [Model input validation](diagnostics/model_input_validation_report.md)
- [Grain diagnostics](diagnostics/grain_diagnostics.md)
- [Missing and quality report](diagnostics/missing_quality_report.md)

## Selected Final Model

- Model: `optuna_two_stage_top_capture`
- Purpose: maximize top-decile revenue capture while preserving competitive RMSLE.
- Final pipeline parameters: `data/processed/final_model_params.json`
- Final holdout metrics: `data/processed/final_model_metrics.csv`
- Final holdout predictions: `data/processed/final_holdout_predictions.parquet`
