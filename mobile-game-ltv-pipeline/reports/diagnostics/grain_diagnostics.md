# Modeling Grain Diagnostics

## Scope

This report validates modeling grain and feature aggregation feasibility only. No model training was performed.

## Grain Comparison

| split | grain | groups | target collision groups | collision rate | event median | event p95 | revenue median | zero revenue rate | event_count <= 2 rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | A_user_id | 75,464 | 34,191 | 45.31% | 32 | 1166 | 0.1667 | 30.42% | 6.59% |
| train | B_user_id_install_day | 146,129 | 6,953 | 4.76% | 8 | 810 | 0.0093 | 47.40% | 15.10% |
| train | C_user_context_install_day | 159,926 | 405 | 0.25% | 7 | 768 | 0.0050 | 48.57% | 15.94% |
| train | D_user_context_install_day_week | 159,926 | 405 | 0.25% | 7 | 768 | 0.0050 | 48.57% | 15.94% |
| test | A_user_id | 31,399 |  | n/a | 9 | 860 | 0.0178 | 45.49% | 13.84% |
| test | B_user_id_install_day | 38,649 |  | n/a | 7 | 765 | 0.0051 | 48.55% | 15.81% |
| test | C_user_context_install_day | 40,105 |  | n/a | 7 | 764 | 0.0044 | 48.78% | 15.94% |
| test | D_user_context_install_day_week | 40,105 |  | n/a | 7 | 764 | 0.0044 | 48.78% | 15.94% |

## Answers

### Can `user_id` alone be used?

No. Under `user_id`, train has 34,191 groups with multiple `ltv_d8_d180` values (45.31% of groups). Static context fields also vary within many `user_id`s. A plain `groupby(user_id)` would mix different installs or contexts and create ambiguous labels.

### How much does a composite key reduce target collisions?

The strongest candidate, `user_id + platform + country_tier + channel_tier + install_day`, reduces train target-collision groups to 405 (0.25%). That is a large reduction from `user_id` alone, though not a perfect fix.

### Does the composite key make features too sparse?

No, not fatally. Under candidate C, train event_count<=2 rate is 15.94% and test event_count<=2 rate is 15.94%. The median event count is still usable, but sparse groups should be monitored and may benefit from segment priors or fallback features.

### Most defensible modeling grain

`user_id + platform + country_tier + channel_tier + install_day` is the most defensible grain at this stage. Candidate D adds `install_week`, but `install_week = install_day // 7`, so it is redundant with `install_day` and should be treated as a reference check rather than a necessary key component.

### What to do with remaining target collisions?

Recommended handling: exclude remaining collision groups from the first supervised training run and log them as data-quality exceptions. Averaging target values would hide an unresolved identity issue; it can be used only as a sensitivity analysis after a clean baseline is established.

### Does Kaggle submission grain match test?

Not cleanly. `sample_submission.csv` has 40,112 rows. Test has 31,399 unique `user_id`s, candidate C has 40,105 groups, and candidate D has 40,105 groups. Candidate C/D almost match sample row count but differ by 7 rows. Before a real competition submission, the dataset version or submission key definition should be verified.

## Feature Aggregation Feasibility

The requested feature families are feasible at candidate C. Context features are constant by construction, activity/revenue/time features are populated for every group, ad features are mostly usable, and IAP features are intentionally sparse because IAP events are rare.

Feature availability details are saved in `data/processed/feature_availability_by_grain.csv`.
