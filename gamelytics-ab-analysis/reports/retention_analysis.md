# Retention Analysis

Retention is treated as a separate supporting analysis. It is not joined to the A/B result because no documented shared identity contract ties `ab_test.user_id` to `reg_data.uid`.

D0 is excluded from the decision interpretation because the registration and first auth event appear to be recorded together.

## Exact-Day Retention
     eligible_users  retained_users  retention_rate
day
1            998952           20071        0.020092
2            997311           40997        0.041108
3            995673           46338        0.046539
7            989145           58140        0.058778
14           977825           44726        0.045740
30           952434           26971        0.028318

## Monthly Cohorts
- Rows in cohort heatmap table: 499
- Latest immature cohorts are excluded per retention day denominator.
- Cohort size is retained in `reports/retention_monthly_heatmap_data.csv` so early sparse cohorts can be interpreted cautiously.
