# Baseline Results

## Scope

This report compares non-ML/early-stage baselines on the time-based validation split. No LightGBM, XGBoost, Optuna, or advanced model training was performed.

## Validation Split

- Train split: install_day 0 to 23 (127,450 rows)
- Valid split: install_day 24 to 30 (32,071 rows)

## Baselines

- Global Mean: train split mean LTV for every validation row.
- Segment Mean: platform + country_tier + channel_tier + install_week mean, with global mean fallback for unseen segments.
- Early Revenue Multiplier: validation D0-D7 revenue multiplied by train mean LTV / train mean D0-D7 revenue, clipped at 0.

## Metrics

| baseline | MAE | RMSE | RMSLE | Spearman | positive rate in predicted top decile | top 10% revenue capture | top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_mean | 24.1797 | 104.0161 | 2.7500 | 0.0000 | 40.34% | 6.51% | 0.65 |
| segment_mean | 16.6481 | 105.0316 | 2.0948 | 0.1139 | 36.81% | 17.15% | 1.71 |
| early_revenue_multiplier | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 79.21% | 72.06% | 7.20 |

## Prediction Checks

| baseline | rows match valid | null predictions | inf predictions | negative predictions |
|---|---|---:|---:|---:|
| global_mean | True | 0 | 0 | 0 |
| segment_mean | True | 0 | 0 | 0 |
| early_revenue_multiplier | True | 0 | 0 | 0 |

## Training Constants

- Global mean LTV: 21.918970
- Early revenue multiplier: 3.906909
- Segment count in train split: 1,117

## Interpretation

These baselines define the first bar for later ML models. Advanced models should improve not only RMSE/RMSLE, but also ranking-oriented metrics such as top-decile revenue capture.
