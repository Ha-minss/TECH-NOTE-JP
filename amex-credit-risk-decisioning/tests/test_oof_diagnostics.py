from __future__ import annotations

import pandas as pd

from amex_risk.modeling.oof_diagnostics import (
    compute_incremental_gain,
    compute_model_correlation,
    compute_single_and_leave_one_out_blends,
)


def test_oof_diagnostics_compute_correlation_leave_one_out_and_gain() -> None:
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0],
            "model_a": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "model_b": [0.85, 0.15, 0.75, 0.25, 0.65, 0.35],
            "model_c": [0.6, 0.4, 0.55, 0.45, 0.5, 0.5],
        }
    )
    prediction_cols = ["model_a", "model_b", "model_c"]

    corr = compute_model_correlation(frame, prediction_cols)
    assert corr.shape == (3, 3)
    assert corr.loc["model_a", "model_b"] > 0.99

    comparison = compute_single_and_leave_one_out_blends(frame, "target", prediction_cols)
    assert set(comparison["evaluation_type"]) == {"single_model", "all_model_equal_blend", "leave_one_out_blend"}
    assert "removed_model" in comparison.columns

    gain = compute_incremental_gain(comparison)
    assert {"model_name", "single_model_amex", "leave_one_out_delta_vs_all", "best_single_delta_vs_model"}.issubset(gain.columns)

