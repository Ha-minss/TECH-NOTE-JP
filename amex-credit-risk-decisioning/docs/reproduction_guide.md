# Reproduction Guide

## Scope

The public repository supports code review and smoke testing without raw AMEX files. Full retraining is intentionally not run as part of public verification because it requires large input files and long-running 5-fold model training.

## Required Inputs For Full Reproduction

Place private inputs outside version control:

```text
data/raw/
  train.parquet
  test.parquet
  train_labels.csv
```

Expected original shapes:

| File | Shape |
|---|---:|
| `train.parquet` | `5,531,451 x 190` |
| `test.parquet` | `11,363,762 x 190` |
| `train_labels.csv` | `458,913 x 2` |

## High-Level Reproduction Commands

The original notebook workflow maps to these module boundaries:

```bash
python -m amex_risk.data.feature_engineering
python -m amex_risk.modeling.train_lgbm_cv
python -m amex_risk.modeling.train_xgb_cv
python -m amex_risk.modeling.train_catboost_cv
python -m amex_risk.modeling.train_mlp_cv
python -m amex_risk.modeling.blend_oof
```

Some commands are documented as module boundaries rather than runnable public training jobs because full training depends on excluded raw data and model libraries.

## Expected Resource Profile

- Memory: high, because the engineered train table has `458,913` rows and more than `3,400` columns.
- Compute: GPU recommended for XGBoost and MLP; CPU runs are possible but slower.
- Storage: raw parquet, feature parquet, OOF predictions, and fold models are excluded from GitHub.

## Public Verification

Run:

```bash
python -m compileall src tests
python -m pytest tests -q -p no:cacheprovider
```

The synthetic fixture verifies code paths only. It is not used for AMEX performance reporting.

