from __future__ import annotations

import pandas as pd


def build_synthetic_scores() -> pd.DataFrame:
    """Small deterministic fixture for smoke tests only, not portfolio performance."""
    frame = pd.DataFrame(
        {
            "customer_ID": [f"SYN{i:03d}" for i in range(1, 21)],
            "target": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "risk_score": [
                0.98,
                0.93,
                0.90,
                0.86,
                0.77,
                0.71,
                0.66,
                0.58,
                0.53,
                0.47,
                0.42,
                0.37,
                0.32,
                0.28,
                0.24,
                0.19,
                0.14,
                0.10,
                0.06,
                0.02,
            ],
            "data_kind": ["synthetic_test_fixture"] * 20,
        }
    )
    frame.attrs["data_kind"] = "synthetic_test_fixture"
    return frame

