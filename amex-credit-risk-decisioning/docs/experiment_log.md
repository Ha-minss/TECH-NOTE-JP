# Experiment Log

## Feature Engineering

The original notebook converted AMEX customer-month records into customer-level features using:

- numeric aggregations: `last`, `first`, `mean`, `std`, `min`, `max`, `sum`, `median`, `count`
- change and ratio features: `last - mean`, `last - first`, `last / mean`, `last / first`
- temporal features: lag and recent-window statistics
- missingness counts and ratios
- categorical summary features
- pivot-lite representation for an additional LightGBM model

## Model Families Verified In The Notebook

- LightGBM full 5-fold OOF
- XGBoost full 5-fold OOF
- CatBoost full 5-fold OOF
- LightGBM Top1600 DART
- LightGBM Top1600 GOSS
- LightGBM recent/change GOSS
- MLP residual GELU BN/LN
- Ridge and logistic stacking checks
- Equal OOF blends

## Public Reconstruction Decisions

- Full OOF, fold prediction, model, feature parquet, and submission files are excluded.
- Aggregate CSVs are retained when they can be traced to visible notebook output.
- The fake OOF demonstration from the former public modeling notebook is not included.
- Synthetic data is used only for smoke tests.

