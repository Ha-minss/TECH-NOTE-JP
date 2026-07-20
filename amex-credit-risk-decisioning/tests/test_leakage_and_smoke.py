from __future__ import annotations

import pandas as pd

from amex_risk.data.synthetic import build_synthetic_scores
from amex_risk.modeling.cv import make_customer_stratified_folds, validate_no_customer_leakage
from amex_risk.evaluation.topk_policy import simulate_topk_policy


def test_group_fold_split_has_no_customer_leakage() -> None:
    frame = pd.DataFrame(
        {
            "customer_ID": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "target": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    folds = make_customer_stratified_folds(frame, customer_col="customer_ID", target_col="target", n_splits=2, random_state=42)

    assert len(folds) == 2
    validate_no_customer_leakage(frame, folds, customer_col="customer_ID")


def test_synthetic_fixture_smoke_test_is_not_portfolio_performance() -> None:
    synthetic = build_synthetic_scores()

    assert synthetic.attrs["data_kind"] == "synthetic_test_fixture"
    assert "customer_ID" in synthetic.columns
    assert synthetic["risk_score"].between(0, 1).all()

    topk = simulate_topk_policy(synthetic, "risk_score", "target", [0.1])
    assert topk.iloc[0]["policy"] == "Top 10%"
    assert topk.iloc[0]["source"] == "synthetic_test_fixture_not_portfolio_performance"

