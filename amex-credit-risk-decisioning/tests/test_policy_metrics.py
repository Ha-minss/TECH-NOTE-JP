from __future__ import annotations

import pandas as pd

from amex_risk.evaluation.cost_scenario import (
    CostAssumption,
    calculate_cost_scenario,
    select_best_threshold_by_scenario,
)
from amex_risk.evaluation.topk_policy import (
    apply_nondefault_weight_scenario,
    simulate_topk_policy,
)


def _score_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_ID": [f"C{i:03d}" for i in range(1, 11)],
            "target": [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            "risk_score": [0.99, 0.95, 0.90, 0.85, 0.50, 0.45, 0.40, 0.30, 0.20, 0.10],
        }
    )


def test_topk_policy_separates_observed_and_weighted_precision() -> None:
    topk = simulate_topk_policy(_score_fixture(), "risk_score", "target", [0.3, 0.5])
    weighted = apply_nondefault_weight_scenario(topk, nondefault_weight=20)

    top30 = weighted.loc[weighted["policy"].eq("Top 30%")].iloc[0]
    assert top30["review_count"] == 3
    assert top30["tp"] == 2
    assert top30["fp"] == 1
    assert round(top30["observed_precision"], 6) == round(2 / 3, 6)
    assert round(top30["weighted_scenario_precision"], 6) == round(2 / 22, 6)
    assert top30["weighting_label"] == "AMEX competition sampling-adjusted scenario"


def test_cost_scenario_calculates_top17_fields_from_modeling_cutoff() -> None:
    policy_table = pd.DataFrame(
        {
            "policy": ["Top 17%"],
            "review_rate": [0.17],
            "review_count": [78016],
            "tp": [71205],
            "fp": [6811],
            "observed_precision": [0.9126973954060704],
            "capture_rate": [0.5992274548086309],
        }
    )
    weighted = apply_nondefault_weight_scenario(policy_table, nondefault_weight=20)
    scenario = calculate_cost_scenario(
        weighted,
        [CostAssumption("Base", ead=1.0, lgd=0.5, intervention_effect=0.2, review_cost=0.01, friction_cost=0.005)],
        modeled_population_count=458_913,
    )
    row = scenario.iloc[0]

    assert row["modeling_sample_cutoff"] == "Top 17%"
    assert row["effective_review_count"] == 207_425
    assert round(row["effective_review_rate"], 6) == round(207_425 / 458_913, 6)
    assert row["net_benefit"] == 4365.15
    assert row["cost_assumption"] == "EAD=1.0; LGD=0.5; intervention_effect=0.2; review_cost=0.01; friction_cost=0.005"


def test_select_best_threshold_uses_scenario_not_operational_claim() -> None:
    scenario = pd.DataFrame(
        {
            "scenario": ["Base", "Base", "Conservative"],
            "policy": ["Top 10%", "Top 17%", "Top 4%"],
            "net_benefit": [3519.54, 4365.15, 115.25],
        }
    )

    best = select_best_threshold_by_scenario(scenario)
    base = best.loc[best["scenario"].eq("Base")].iloc[0]
    assert base["policy"] == "Top 17%"
    assert "maximum simulated net benefit" in base["interpretation"]
    assert "operational policy" not in base["interpretation"].lower()

