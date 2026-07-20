# Results Provenance

`Untitled41.ipynb` is the source of truth for AMEX experiment results. This repository does not invent model results, feature counts, or policy metrics that are not visible in that notebook.

## Public Tables

| Public file | Source in original notebook | Notes |
|---|---|---|
| `outputs/tables/model_cv_summary.csv` | cells 21, 30, 33, 34, 35, 36, 37, 38, 40 | Model and blend metrics transcribed from visible output tables. |
| `outputs/tables/blend_comparison.csv` | cells 21, 39, 40 | Blend and stacking metrics only. |
| `outputs/tables/topk_policy_tradeoff.csv` | cell 45 Top-K Policy Simulation | Observed precision from the modeling sample. |
| `outputs/tables/weighted_policy_tradeoff.csv` | cell 45 Weighted Top-K Simulation | AMEX competition sampling-adjusted scenario; not population calibration. |
| `outputs/tables/risk_decile_summary.csv` | public aggregate output and validation notebook output | Decile counts and default rates from final OOF ranking. |
| `outputs/tables/top17_base_cost_scenario.csv` | cell 45 Best Threshold by Scenario | Base cost assumption Top 17% row. |
| `outputs/tables/best_threshold_by_scenario_weighted.csv` | cell 45 Best Threshold by Scenario | Conservative, Base, Aggressive scenario best rows. |
| `outputs/tables/top10_to_top100_cumulative_summary.csv` | public aggregate output | Cumulative capture and observed default rate by review scope. |
| `outputs/tables/monitoring_bucket_summary.csv` | cell 45 Monitoring Bucket Summary | Risk-band monitoring aggregate. |

## Key Numbers

| Claim | Value | Source |
|---|---:|---|
| Total modeling rows | `458,913` | cells 8, 14, 45 |
| Total defaults | `118,828` | cells 14, 45 |
| Overall default rate | `25.8934%` | cells 14, 45 |
| Equal blend 8 models AMEX metric | `0.797631` | cell 21 |
| Equal blend 8 models ROC AUC | `0.962782` | cell 21 |
| Ridge stacking AMEX metric | `0.797538` | cell 40 |
| Top 1% observed precision | `99.98%` | cells 14, 45 |
| Top 5% observed precision | `99.11%` | cells 14, 45 |
| Top 5% 20x weighted scenario precision | `84.79%` | cell 45 |
| D1 default rate | `96.59%` | public validation output |
| D10 default rate | `0.04%` | public validation output |
| Base scenario Top 17% simulated net benefit | `4,365.15` | cell 45 |

## Removed Or Rejected Values

- The former public modeling notebook generated fake model columns by applying small formulas to `risk_score`. Those values are excluded here.
- The previous `public_sample_scores.csv` contained only the top 1,000 scored rows and all targets were `1`. It is excluded because it is not a valid performance sample.
- No unverified model family, feature set, or leaderboard-style score is added.

