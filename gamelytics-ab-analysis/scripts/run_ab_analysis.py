from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from gamelytics.inference import (
    bootstrap_mean_difference,
    bootstrap_relative_lift,
    permutation_test_mean_difference,
    two_proportion_z_test,
)
from gamelytics.io import load_ab_data, read_semicolon_csv
from gamelytics.metrics import compare_groups, group_summary, revenue_distribution_summary
from gamelytics.report import business_scenarios, write_ab_report, write_data_audit
from gamelytics.sensitivity import sensitivity_table
from gamelytics.validation import audit_ab_data, sample_ratio_mismatch_test
from gamelytics.visualization import (
    plot_bootstrap_distribution,
    plot_concentration,
    plot_metric_comparison,
    plot_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    tables = reports / "tables"
    figures = reports / "figures"
    reports.mkdir(exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    ab = load_ab_data(args.data_dir)
    audit = audit_ab_data(ab)
    overlap = None
    reg_path = Path(args.data_dir) / "reg_data.csv"
    if reg_path.exists():
        reg = read_semicolon_csv(reg_path, usecols=["uid"])
        ab_users = set(ab["user_id"])
        reg_users = set(reg["uid"])
        matching = len(ab_users & reg_users)
        overlap = {
            "ab_users": len(ab_users),
            "matching_reg_users": matching,
            "ab_users_in_reg_share": matching / len(ab_users),
        }
    write_data_audit(reports / "data_audit.md", audit, overlap)

    summary = group_summary(ab)
    distribution = revenue_distribution_summary(ab)
    comparison = compare_groups(summary)
    summary.to_csv(tables / "ab_group_summary.csv")
    distribution.to_csv(tables / "revenue_distribution_summary.csv")

    counts = summary["users"].to_dict()
    srm = sample_ratio_mismatch_test(counts)
    conversion_test = two_proportion_z_test(
        int(summary.loc["a", "payers"]),
        int(summary.loc["a", "users"]),
        int(summary.loc["b", "payers"]),
        int(summary.loc["b", "users"]),
    )

    a_revenue = ab.loc[ab["testgroup"] == "a", "revenue"].to_numpy()
    b_revenue = ab.loc[ab["testgroup"] == "b", "revenue"].to_numpy()
    arpu_bootstrap = bootstrap_mean_difference(
        a_revenue, b_revenue, iterations=args.bootstrap_iterations, seed=args.seed
    )
    lift_bootstrap = bootstrap_relative_lift(
        a_revenue, b_revenue, iterations=args.bootstrap_iterations, seed=args.seed
    )
    permutation = permutation_test_mean_difference(
        a_revenue, b_revenue, iterations=args.permutation_iterations, seed=args.seed
    )

    a_payer_revenue = ab.loc[(ab["testgroup"] == "a") & (ab["revenue"] > 0), "revenue"].to_numpy()
    b_payer_revenue = ab.loc[(ab["testgroup"] == "b") & (ab["revenue"] > 0), "revenue"].to_numpy()
    arppu_bootstrap = bootstrap_mean_difference(
        a_payer_revenue, b_payer_revenue, iterations=args.bootstrap_iterations, seed=args.seed
    )

    sensitivity = sensitivity_table(
        ab, iterations=args.bootstrap_iterations, seed=args.seed, control="a", treatment="b"
    )
    sensitivity.to_csv(tables / "whale_sensitivity_summary.csv", index=False)

    business = business_scenarios(comparison["arpu_diff"])
    business.to_csv(tables / "business_scenarios.csv", index=False)

    inference = pd.DataFrame(
        [
            {
                "metric": "arpu_b_minus_a",
                "observed": arpu_bootstrap.observed_diff,
                "ci_low": arpu_bootstrap.ci_low,
                "ci_high": arpu_bootstrap.ci_high,
                "p_value": arpu_bootstrap.p_value,
            },
            {
                "metric": "arpu_relative_lift",
                "observed": lift_bootstrap.observed_diff,
                "ci_low": lift_bootstrap.ci_low,
                "ci_high": lift_bootstrap.ci_high,
                "p_value": lift_bootstrap.p_value,
            },
            {
                "metric": "permutation_arpu_b_minus_a",
                "observed": permutation.observed_diff,
                "ci_low": permutation.ci_low,
                "ci_high": permutation.ci_high,
                "p_value": permutation.p_value,
            },
            {
                "metric": "arppu_b_minus_a_payers_only",
                "observed": arppu_bootstrap.observed_diff,
                "ci_low": arppu_bootstrap.ci_low,
                "ci_high": arppu_bootstrap.ci_high,
                "p_value": arppu_bootstrap.p_value,
            },
        ]
    )
    inference.to_csv(tables / "inference_summary.csv", index=False)

    plot_metric_comparison(summary, figures / "ab_metric_comparison")
    plot_concentration(distribution, figures / "revenue_concentration")
    plot_bootstrap_distribution(
        arpu_bootstrap.samples,
        arpu_bootstrap.observed_diff,
        arpu_bootstrap.ci_low,
        arpu_bootstrap.ci_high,
        figures / "arpu_bootstrap_distribution",
    )
    plot_sensitivity(sensitivity, figures / "whale_sensitivity")

    write_ab_report(
        reports / "ab_analysis.md",
        summary,
        distribution,
        comparison,
        srm,
        conversion_test,
        arpu_bootstrap,
        lift_bootstrap,
        permutation,
        arppu_bootstrap,
        sensitivity,
        business,
    )


if __name__ == "__main__":
    main()
