# Test Prediction Report

## Scope

- Final model: `optuna_two_stage_top_capture` refit on all labeled train_model_input rows.
- Submission grain: one row per unique `user_id` in `test.csv`.
- Test labels are unavailable, so test MAE/RMSE/RMSLE are not calculated.
- Kaggle hidden test score is available only after submitting `submission.csv` to Kaggle.

## Counts

- Train row count: 159,521
- Test feature row count: 40,105
- Test unique user count / submission row count: 31,399
- sample_submission row count in provided zip: 40,112
- Matched prediction row count: 31,399
- Fallback prediction row count: 0

## Fallback Method Summary

| fallback method | row count |
|---|---:|
| none | 0 |

## Prediction Checks

| check | passed |
|---|---|
| row_count_matches_expected_users | True |
| id_order_matches_expected_users | True |
| prediction_not_null | True |
| prediction_not_inf | True |
| prediction_non_negative | True |
| fallback_count_matches_expected | True |
| model_artifacts_exist | True |

## Test Prediction Summary

| metric | value |
|---|---:|
| mean | 6.400414 |
| median | 0.076051 |
| p75 | 2.011875 |
| p90 | 7.190641 |
| p95 | 13.001448 |
| p99 | 78.487470 |
| max | 3506.552311 |

## p_positive Summary

| metric | value |
|---|---:|
| mean | 0.390395 |
| median | 0.146956 |
| p90 | 0.997776 |
| p99 | 0.999843 |

## Submission Notes

- Kaggle rejected the provided sample_submission row count; the accepted contract is the competition description: one row per unique test user_id.
- Context-level predictions are averaged to user-level LTV before writing `submission.csv`.
- Test targets are not available in the public files; hidden test score can only be checked after Kaggle submission.
