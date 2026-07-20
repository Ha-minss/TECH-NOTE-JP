from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostAssumption:
    scenario: str
    ead: float
    lgd: float
    intervention_effect: float
    review_cost: float
    friction_cost: float

    def label(self) -> str:
        return (
            f"EAD={self.ead}; LGD={self.lgd}; intervention_effect={self.intervention_effect}; "
            f"review_cost={self.review_cost}; friction_cost={self.friction_cost}"
        )


def calculate_cost_scenario(
    weighted_policy_table: pd.DataFrame,
    assumptions: list[CostAssumption],
    modeled_population_count: int,
) -> pd.DataFrame:
    rows = []
    for _, policy in weighted_policy_table.iterrows():
        for assumption in assumptions:
            avoided_loss = policy["tp"] * assumption.ead * assumption.lgd * assumption.intervention_effect
            review_cost_total = policy["effective_review_count"] * assumption.review_cost
            friction_cost_total = policy["effective_fp"] * assumption.friction_cost
            operating_cost_total = review_cost_total + friction_cost_total
            net_benefit = avoided_loss - operating_cost_total
            rows.append(
                {
                    "scenario": assumption.scenario,
                    "modeling_sample_cutoff": policy["policy"],
                    "policy": policy["policy"],
                    "review_rate": policy["review_rate"],
                    "review_count": int(policy["review_count"]),
                    "tp": int(policy["tp"]),
                    "fp": int(policy["fp"]),
                    "observed_precision": policy["observed_precision"],
                    "effective_fp": policy["effective_fp"],
                    "effective_review_count": policy["effective_review_count"],
                    "effective_review_rate": policy["effective_review_count"] / modeled_population_count,
                    "weighted_scenario_precision": policy["weighted_scenario_precision"],
                    "avoided_loss": round(avoided_loss, 6),
                    "review_cost_total": round(review_cost_total, 6),
                    "friction_cost_total": round(friction_cost_total, 6),
                    "operating_cost_total": round(operating_cost_total, 6),
                    "net_benefit": round(net_benefit, 6),
                    "cost_assumption": assumption.label(),
                    "interpretation_boundary": (
                        "modeling sample cutoff with maximum simulated net benefit under the stated cost assumptions"
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_best_threshold_by_scenario(scenario_table: pd.DataFrame) -> pd.DataFrame:
    best = (
        scenario_table.sort_values(["scenario", "net_benefit"], ascending=[True, False])
        .groupby("scenario", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["interpretation"] = (
        "modeling sample cutoff with maximum simulated net benefit under the stated assumptions; "
        "not an automatic operating rule"
    )
    return best


