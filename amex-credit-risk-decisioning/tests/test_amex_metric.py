from __future__ import annotations

import numpy as np

from amex_risk.modeling.metrics import amex_metric, normalized_gini, top_four_percent_captured


def test_amex_metric_matches_reference_formula_on_small_fixture() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_score = np.array([0.90, 0.10, 0.80, 0.20, 0.70])

    assert top_four_percent_captured(y_true, y_score) == 1 / 3
    assert round(normalized_gini(y_true, y_score), 6) == 1.0
    assert round(amex_metric(y_true, y_score), 6) == round(0.6666666666666666, 6)


def test_amex_metric_rejects_single_class_targets() -> None:
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])

    try:
        amex_metric(y_true, y_score)
    except ValueError as exc:
        assert "both default and non-default" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for single-class targets")


