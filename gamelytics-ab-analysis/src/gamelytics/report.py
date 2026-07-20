from __future__ import annotations

from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"{value:,.2f}"


def write_data_audit(path: Path, audit: dict, ab_reg_overlap: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Audit",
        "",
        "## ab_test.csv",
        f"- Rows: {audit['rows']:,}",
        f"- Unique users: {audit['unique_users']:,}",
        f"- Duplicate rows: {audit['duplicate_rows']:,}",
        f"- Duplicate user_id values: {audit['duplicate_user_ids']:,}",
        f"- Users assigned to multiple groups: {audit['users_in_multiple_groups']:,}",
        f"- Missing values: {audit['missing_values']}",
        f"- Negative revenue rows: {audit['negative_revenue_rows']:,}",
        f"- Overall zero-revenue share: {pct(audit['zero_revenue_share'])}",
        f"- Group counts: {audit['group_counts']}",
        f"- Payers by group: {audit['payers_by_group']}",
        "",
        "CSV files use semicolon delimiters (`;`). The A/B analysis uses only `ab_test.csv` because no documented user-level join key ties this experiment to registration/auth logs.",
    ]
    if ab_reg_overlap:
        lines.extend(
            [
                "",
                "## Cross-file Check",
                f"- A/B users present in `reg_data.uid`: {pct(ab_reg_overlap['ab_users_in_reg_share'])}",
                f"- A/B user count: {ab_reg_overlap['ab_users']:,}",
                f"- Matching registration users: {ab_reg_overlap['matching_reg_users']:,}",
                "This overlap is descriptive only. It is not used to combine A/B and retention results because the prompt does not establish that both files use the same user identity namespace.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ab_report(
    path: Path,
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    comparison: dict,
    srm: dict,
    conversion_test: dict,
    arpu_bootstrap,
    lift_bootstrap,
    permutation,
    arppu_bootstrap,
    sensitivity: pd.DataFrame,
    business: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Promotion A/B Test Report",
        "",
        "## Decision",
        "Recommendation: **run an additional experiment**. B showed a higher observed ARPU, but the ARPU confidence interval includes zero and payer conversion is lower.",
        "",
        "## Metric Definitions",
        "- Primary metric: ARPU = total revenue / all assigned users.",
        "- Secondary metrics: conversion rate, ARPPU, payer count, total revenue, and revenue concentration.",
        "- Identity: ARPU = conversion rate x ARPPU. ARPPU is conditional on becoming a payer and is not interpreted as an independent treatment effect.",
        "",
        "## Group Summary",
        summary.to_string(),
        "",
        "## Main Effects",
        f"- B minus A ARPU: {money(comparison['arpu_diff'])} ({pct(comparison['arpu_lift'])} lift).",
        f"- B minus A conversion: {comparison['conversion_diff_pp']:.4f} percentage points ({pct(comparison['conversion_relative_diff'])} relative).",
        f"- B minus A ARPPU: {money(comparison['arppu_diff'])}.",
        f"- Total revenue difference: {money(comparison['total_revenue_diff'])}.",
        "",
        "## Statistical Checks",
        f"- SRM chi-square: {srm['chi_square']:.6f}, p-value: {srm['p_value']:.6f}.",
        f"- Conversion z-test p-value: {conversion_test['p_value']:.6f}; 95% CI for B-A conversion: {conversion_test['ci_low']:.6f} to {conversion_test['ci_high']:.6f}.",
        f"- ARPU bootstrap B-A: {money(arpu_bootstrap.observed_diff)}, 95% CI {money(arpu_bootstrap.ci_low)} to {money(arpu_bootstrap.ci_high)}.",
        f"- ARPU relative lift bootstrap: {pct(lift_bootstrap.observed_diff)}, 95% CI {pct(lift_bootstrap.ci_low)} to {pct(lift_bootstrap.ci_high)}.",
        f"- Permutation test p-value for ARPU difference: {permutation.p_value:.6f}.",
        f"- ARPPU bootstrap B-A among payers only: {money(arppu_bootstrap.observed_diff)}, 95% CI {money(arppu_bootstrap.ci_low)} to {money(arppu_bootstrap.ci_high)}.",
        "",
        "## Revenue Concentration",
        distribution.to_string(),
        "",
        "## Whale Sensitivity",
        sensitivity.to_string(index=False),
        "",
        "## Business-Scale Readout",
        business.to_string(index=False),
        "",
        "## Interpretation",
        "B showed a higher observed ARPU while payer conversion was lower. This means the offer may trade breadth of paying users for higher revenue per payer, but the current data does not establish a stable B advantage.",
        "Statistical non-significance does not mean there is no effect; it means the current data is not strong enough to confirm the effect. Whale sensitivity is used to describe revenue concentration and mean uncertainty, not to declare whale payments invalid.",
        "",
        "No cost, margin, or discount data is available. Business impact is reported as incremental gross revenue scenarios, not ROI.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retention_report(path: Path, summary: pd.DataFrame, heatmap_data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retention Analysis",
        "",
        "Retention is treated as a separate supporting analysis. It is not joined to the A/B result because no documented shared identity contract ties `ab_test.user_id` to `reg_data.uid`.",
        "",
        "D0 is excluded from the decision interpretation because the registration and first auth event appear to be recorded together.",
        "",
        "## Exact-Day Retention",
        summary.to_string(),
        "",
        "## Monthly Cohorts",
        f"- Rows in cohort heatmap table: {len(heatmap_data):,}",
        "- Latest immature cohorts are excluded per retention day denominator.",
        "- Cohort size is retained in `reports/retention_monthly_heatmap_data.csv` so early sparse cohorts can be interpreted cautiously.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def business_scenarios(arpu_diff: float) -> pd.DataFrame:
    rows = []
    for margin in (0.2, 0.4, 0.6):
        for promo_cost in (0, 0.1, 0.5, 1.0):
            incremental_revenue = arpu_diff * 100_000
            contribution = incremental_revenue * margin - promo_cost * 100_000
            rows.append(
                {
                    "users": 100000,
                    "gross_incremental_revenue": incremental_revenue,
                    "contribution_margin_assumption": margin,
                    "promo_cost_per_user_assumption": promo_cost,
                    "illustrative_net_contribution": contribution,
                }
            )
    return pd.DataFrame(rows)
