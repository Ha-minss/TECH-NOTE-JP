# Promotion A/B Test Report

## Decision
Recommendation: **run an additional experiment**. B showed a higher observed ARPU, but the ARPU confidence interval includes zero and payer conversion is lower.

## Metric Definitions
- Primary metric: ARPU = total revenue / all assigned users.
- Secondary metrics: conversion rate, ARPPU, payer count, total revenue, and revenue concentration.
- Identity: ARPU = conversion rate x ARPPU. ARPPU is conditional on becoming a payer and is not interpreted as an independent treatment effect.

## Group Summary
            users  payers  total_revenue       arpu  median_revenue  max_revenue  conversion_rate        arppu  arpu_identity
testgroup
a          202103    1928        5136189  25.413720             0.0        37433         0.009540  2663.998444      25.413720
b          202667    1805        5421603  26.751287             0.0         4000         0.008906  3003.658172      26.751287

## Main Effects
- B minus A ARPU: 1.34 (5.26% lift).
- B minus A conversion: -0.0633 percentage points (-6.64% relative).
- B minus A ARPPU: 339.66.
- Total revenue difference: 285,414.00.

## Statistical Checks
- SRM chi-square: 0.785869, p-value: 0.375352.
- Conversion z-test p-value: 0.035029; 95% CI for B-A conversion: -0.001222 to -0.000044.
- ARPU bootstrap B-A: 1.34, 95% CI -2.87 to 5.46.
- ARPU relative lift bootstrap: 5.26%, 95% CI -9.74% to 25.17%.
- Permutation test p-value for ARPU difference: 0.527200.
- ARPPU bootstrap B-A among payers only: 339.66, 95% CI -69.43 to 732.98.

## Revenue Concentration
            users  payers  zero_revenue_share  payer_revenue_mean  payer_revenue_median  payer_revenue_p90  payer_revenue_p95  payer_revenue_p99  max_revenue  top_1pct_payer_revenue_share  top_5pct_payer_revenue_share  top_10pct_payer_revenue_share
testgroup
a          202103    1928            0.990460         2663.998444                 311.0              393.3           37299.65           37340.73      37433.0                      0.138230                      0.697647                       0.899042
b          202667    1805            0.991094         3003.658172                3022.0             3795.8            3891.80            3981.92       4000.0                      0.013253                      0.065567                       0.129383

## Whale Sensitivity
                    scenario  cutoff    arpu_a    arpu_b  b_minus_a    ci_low   ci_high              direction
               raw_all_users     NaN 25.413720 26.751287   1.337567 -2.873315  5.464838 B observed ARPU higher
common_top_0.1pct_winsorized  3675.0  4.937888 26.523573  21.585685 20.282970 22.936279 B observed ARPU higher
common_top_0.5pct_winsorized   390.0  2.935949  3.473432   0.537483  0.325591  0.752032 B observed ARPU higher

## Business-Scale Readout
 users  gross_incremental_revenue  contribution_margin_assumption  promo_cost_per_user_assumption  illustrative_net_contribution
100000              133756.685631                             0.2                             0.0                   26751.337126
100000              133756.685631                             0.2                             0.1                   16751.337126
100000              133756.685631                             0.2                             0.5                  -23248.662874
100000              133756.685631                             0.2                             1.0                  -73248.662874
100000              133756.685631                             0.4                             0.0                   53502.674253
100000              133756.685631                             0.4                             0.1                   43502.674253
100000              133756.685631                             0.4                             0.5                    3502.674253
100000              133756.685631                             0.4                             1.0                  -46497.325747
100000              133756.685631                             0.6                             0.0                   80254.011379
100000              133756.685631                             0.6                             0.1                   70254.011379
100000              133756.685631                             0.6                             0.5                   30254.011379
100000              133756.685631                             0.6                             1.0                  -19745.988621

## Interpretation
B showed a higher observed ARPU while payer conversion was lower. This means the offer may trade breadth of paying users for higher revenue per payer, but the current data does not establish a stable B advantage.
Statistical non-significance does not mean there is no effect; it means the current data is not strong enough to confirm the effect. Whale sensitivity is used to describe revenue concentration and mean uncertainty, not to declare whale payments invalid.

No cost, margin, or discount data is available. Business impact is reported as incremental gross revenue scenarios, not ROI.
