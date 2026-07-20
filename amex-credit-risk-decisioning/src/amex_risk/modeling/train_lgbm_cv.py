from __future__ import annotations

import pandas as pd


def get_lgbm_full_params() -> dict[str, object]:
    """Parameters recorded from the original full LightGBM CV experiment."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 5000,
        "random_state": 42,
    }


def expected_training_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["train_feat_no_diffmono_v1.parquet", "458,913 x 3,414 customer-level table"],
            ["target", "binary default label from AMEX train_labels.csv"],
            ["customer_ID", "group key; never used as model feature"],
        ],
        columns=["input", "description"],
    )

