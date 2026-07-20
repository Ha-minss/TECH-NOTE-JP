# Rolling Time Validation Results

## Scope

This report evaluates model stability with expanding install_day time folds. It does not use random KFold, OOF, Optuna, or LightGBM.

## Folds

| fold | train days | valid days | train rows | valid rows | train positive rate | valid positive rate |
|---:|---|---|---:|---:|---:|---:|
| 1 | 0-13 | 14-16 | 81,185 | 14,186 | 42.28% | 36.10% |
| 2 | 0-16 | 17-19 | 95,371 | 13,825 | 41.36% | 38.34% |
| 3 | 0-19 | 20-23 | 109,196 | 18,254 | 40.98% | 39.30% |
| 4 | 0-23 | 24-30 | 127,450 | 32,071 | 40.74% | 39.30% |

## Rolling Summary

| model | RMSLE mean | RMSLE std | top 10% capture mean | top 10% capture std | Spearman mean | fold count |
|---|---:|---:|---:|---:|---:|---:|
| single_stage_xgb_target_encoding_features | 0.5349 | 0.0157 | 78.76% | 2.32% | 0.7961 | 4 |
| single_stage_xgb_velocity_ratio_features | 0.5400 | 0.0169 | 78.52% | 2.10% | 0.7955 | 4 |
| two_stage_xgb_target_encoding_features | 0.5409 | 0.0120 | 79.77% | 2.62% | 0.7979 | 4 |
| two_stage_xgb_velocity_ratio_features | 0.5438 | 0.0132 | 79.16% | 2.06% | 0.7971 | 4 |

## Fold Metrics

| fold | model | RMSLE | RMSE | Spearman | top 10% revenue capture | top-decile lift |
|---:|---|---:|---:|---:|---:|---:|
| 1 | single_stage_xgb_target_encoding_features | 0.5153 | 54.0589 | 0.7870 | 80.40% | 8.04 |
| 1 | single_stage_xgb_velocity_ratio_features | 0.5192 | 51.3234 | 0.7867 | 80.34% | 8.03 |
| 1 | two_stage_xgb_target_encoding_features | 0.5265 | 48.7735 | 0.7885 | 82.22% | 8.22 |
| 1 | two_stage_xgb_velocity_ratio_features | 0.5281 | 47.7425 | 0.7865 | 80.51% | 8.05 |
| 2 | single_stage_xgb_target_encoding_features | 0.5311 | 104.7475 | 0.7936 | 77.14% | 7.71 |
| 2 | single_stage_xgb_velocity_ratio_features | 0.5357 | 104.0903 | 0.7908 | 76.80% | 7.68 |
| 2 | two_stage_xgb_target_encoding_features | 0.5373 | 98.7809 | 0.7952 | 77.35% | 7.73 |
| 2 | two_stage_xgb_velocity_ratio_features | 0.5397 | 100.8794 | 0.7948 | 77.29% | 7.73 |
| 3 | single_stage_xgb_target_encoding_features | 0.5527 | 112.0781 | 0.8016 | 81.08% | 8.11 |
| 3 | single_stage_xgb_velocity_ratio_features | 0.5593 | 113.3209 | 0.8014 | 80.33% | 8.03 |
| 3 | two_stage_xgb_target_encoding_features | 0.5549 | 110.6037 | 0.8028 | 81.84% | 8.18 |
| 3 | two_stage_xgb_velocity_ratio_features | 0.5596 | 112.1314 | 0.8016 | 81.32% | 8.13 |
| 4 | single_stage_xgb_target_encoding_features | 0.5404 | 83.7934 | 0.8023 | 76.41% | 7.64 |
| 4 | single_stage_xgb_velocity_ratio_features | 0.5456 | 82.5995 | 0.8032 | 76.60% | 7.66 |
| 4 | two_stage_xgb_target_encoding_features | 0.5447 | 80.2077 | 0.8051 | 77.66% | 7.76 |
| 4 | two_stage_xgb_velocity_ratio_features | 0.5477 | 80.0574 | 0.8053 | 77.51% | 7.75 |

## Questions

- Single holdout consistency: fold 4 RMSLE winner is `single_stage_xgb_target_encoding_features`; rolling mean RMSLE winner is `single_stage_xgb_target_encoding_features`. Fold 4 top-capture winner is `two_stage_xgb_target_encoding_features`; rolling mean top-capture winner is `two_stage_xgb_target_encoding_features`.
- Most stable by RMSLE mean: `single_stage_xgb_target_encoding_features` with mean RMSLE 0.534866 and std 0.015745.
- Most stable by top 10% capture mean: `two_stage_xgb_target_encoding_features` with mean capture 79.77% and std 2.62%.
- Fold-specific RMSLE winners: fold 1 `single_stage_xgb_target_encoding_features` (0.5153); fold 2 `single_stage_xgb_target_encoding_features` (0.5311); fold 3 `single_stage_xgb_target_encoding_features` (0.5527); fold 4 `single_stage_xgb_target_encoding_features` (0.5404).
- Fold-specific top-capture winners: fold 1 `two_stage_xgb_target_encoding_features` (82.22%); fold 2 `two_stage_xgb_target_encoding_features` (77.35%); fold 3 `two_stage_xgb_target_encoding_features` (81.84%); fold 4 `two_stage_xgb_target_encoding_features` (77.66%).
- Optuna recommendation: tune `single_stage_xgb_target_encoding_features` for RMSLE and `two_stage_xgb_target_encoding_features` for business ranking. If they are the same model, use two objectives on that model first before broadening the search.

## Leakage Controls

- Each fold rebuilds target encoding using only that fold's train rows, then maps encodings to the valid rows.
- Each model fit constructs preprocessing and p1/p99 clipping from the fold train matrix only.
- Validation target is used only for early stopping and metric calculation, not for target encoding fit.
