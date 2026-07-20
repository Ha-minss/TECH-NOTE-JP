from __future__ import annotations

import numpy as np
import pandas as pd


def make_customer_stratified_folds(
    frame: pd.DataFrame,
    customer_col: str,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create customer-grouped stratified folds without splitting customers."""
    customer_targets = frame.groupby(customer_col, sort=False)[target_col].max().reset_index()
    rng = np.random.default_rng(random_state)
    fold_customers = [set() for _ in range(n_splits)]
    for _, group in customer_targets.groupby(target_col, sort=False):
        customers = group[customer_col].to_numpy(copy=True)
        rng.shuffle(customers)
        for position, customer in enumerate(customers):
            fold_customers[position % n_splits].add(customer)

    all_customers = set(customer_targets[customer_col])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for valid_customers in fold_customers:
        train_customers = all_customers.difference(valid_customers)
        train_idx = frame.index[frame[customer_col].isin(train_customers)].to_numpy()
        valid_idx = frame.index[frame[customer_col].isin(valid_customers)].to_numpy()
        folds.append((train_idx, valid_idx))
    return folds


def validate_no_customer_leakage(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    customer_col: str,
) -> None:
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_customers = set(frame.loc[train_idx, customer_col])
        valid_customers = set(frame.loc[valid_idx, customer_col])
        overlap = train_customers.intersection(valid_customers)
        if overlap:
            raise ValueError(f"Fold {fold_number} leaks customers across train/validation: {sorted(overlap)[:5]}")
