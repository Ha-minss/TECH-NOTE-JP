# XGBoost Feature Engineering Experiment Results

## Scope

This experiment keeps the XGBoost log-target model and time-based validation split fixed, then changes only the feature family. It does not use Optuna, OOF, random KFold, LightGBM, or two-stage modeling.

## Validation Split

- Train split: install_day 0 to 23 (127,450 rows)
- Valid split: install_day 24 to 30 (32,071 rows)

## Metrics

| feature set | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb_current_full | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 98.66% | 76.32% | 7.63 |
| xgb_time_bucket_features | 5.1355 | 84.1380 | 0.5447 | 0.8025 | 98.63% | 76.57% | 7.66 |
| xgb_velocity_ratio_features | 5.1225 | 82.5995 | 0.5456 | 0.8032 | 98.75% | 76.60% | 7.66 |
| xgb_frequency_interaction_features | 5.0886 | 83.7111 | 0.5430 | 0.8031 | 98.91% | 76.49% | 7.65 |
| xgb_target_encoding_features | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 98.88% | 76.41% | 7.64 |
| xgb_all_derived_features | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 98.88% | 76.41% | 7.64 |

## Questions

- Best RMSLE improvement versus current_full: xgb_all_derived_features (rmsle=0.540359, current_full delta=0.007231).
- Best top 10% revenue capture improvement versus current_full: xgb_velocity_ratio_features (top_10pct_revenue_capture=0.766040, current_full delta=0.002849).
- Target encoding result: target encoding RMSLE=0.540359, top 10% capture=76.41%. Current full RMSLE=0.547590. Because target encodings are train-split fitted and not OOF, validation mapping is leakage-safe but train-side overfit risk remains; this should be revisited with rolling/OOF encoding later.
- Final feature set candidate: `xgb_all_derived_features` for the next modeling stage, unless the business goal prioritizes top-decile capture over RMSLE.
- `xgb_all_derived_features` is an explicit full derived-feature set: time buckets + velocity ratios + frequency/count interactions + leakage-safe target encodings.

## Prediction Checks

| feature set | rows match valid | null predictions | inf predictions | negative predictions |
|---|---|---:|---:|---:|
| xgb_current_full | True | 0 | 0 | 0 |
| xgb_time_bucket_features | True | 0 | 0 | 0 |
| xgb_velocity_ratio_features | True | 0 | 0 | 0 |
| xgb_frequency_interaction_features | True | 0 | 0 | 0 |
| xgb_target_encoding_features | True | 0 | 0 | 0 |
| xgb_all_derived_features | True | 0 | 0 | 0 |
