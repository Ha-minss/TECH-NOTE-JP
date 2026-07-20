# Feature Ablation And OOF Diagnostics

## Can We Compute These Results From The Public Repository?

No. The public repository intentionally excludes:

- full feature parquet files,
- full OOF prediction CSVs,
- fold-level validation predictions,
- trained model files,
- raw AMEX parquet data.

The repository can document verified results already visible in `Untitled41.ipynb`, but it cannot locally compute new feature-view ablations, model correlations, leave-one-out blends, or incremental gains without the private Colab artifacts.

## Single Colab Notebook

Use the notebook below for all additional diagnostics:

```text
notebooks/03_colab_feature_ablation_and_oof_diagnostics.ipynb
```

It auto-checks the original artifact roots used by the source Colab workflow:

```text
/content/drive/MyDrive/amex_features
/content/gdrive/MyDrive/amex_features
```

The notebook has two switches:

```python
RUN_OOF_DIAGNOSTICS = True
RUN_FEATURE_ABLATION = False
```

Keep `RUN_OOF_DIAGNOSTICS = True` to compute OOF contribution diagnostics when OOF files exist. Set `RUN_FEATURE_ABLATION = True` only when you are ready to retrain feature-view CV models.

## OOF Diagnostics

If the private Colab or Drive folder still contains OOF prediction files, the notebook can compute:

- model-to-model OOF prediction correlation,
- equal blend performance,
- leave-one-model-out blend performance,
- model contribution measured by AMEX metric drop when removed,
- single model vs all-model blend incremental gain.

Expected private inputs include:

```text
oof_full_lgbm.csv
oof_full_xgb_gpu.csv
oof_lgbm_top1600_dart_5fold.csv
oof_catboost_full_5fold.csv
oof_lgbm_top1600_goss_5fold.csv
oof_lgbm_recent_change_goss_5fold.csv
oof_mlp_residual_gelu_bnln_onecycle_5fold.csv
oof_pivot_lite_lgbm_5fold.csv
```

Expected aggregate outputs after running:

```text
portfolio_extra_diagnostics/tables/oof_model_correlation.csv
portfolio_extra_diagnostics/tables/oof_single_and_leave_one_out_blend.csv
portfolio_extra_diagnostics/tables/oof_incremental_gain.csv
```

Do not commit full OOF prediction files.

## Feature-View Ablation

The requested comparison is conceptually valid:

| Feature view | What it answers |
|---|---|
| Basic summary only | What happens if we use only customer-level aggregate variables? |
| Summary + change/ratio | Did recent deviation features help? |
| Summary + change/ratio + temporal | Did lag/time-position features help? |
| Summary + change/ratio + temporal + recent window | Did recent-window statistics help? |
| Pivot-lite | Did an alternative time representation help the ensemble? |

These results must be trained or recovered from existing CV outputs. They cannot be inferred from the current public aggregate tables.

To run ablation in Colab, open the single notebook and set:

```python
RUN_FEATURE_ABLATION = True
SMOKE_TEST_ROWS = None
```

Keep `SMOKE_TEST_ROWS = None` for publishable results. If you set `SMOKE_TEST_ROWS` to a smaller number, the run is only a pipeline smoke test and its metrics must not be reported as AMEX results.

## How To Write The Result If Computed

A valid write-up should look like this:

```text
Using only basic customer-level summary features, the model recorded AMEX Metric = A.
Adding change/ratio features recorded B, and adding temporal/recent-window features recorded C.
The best single OOF model was D, while the 8-model equal blend recorded 0.797631.
Leave-one-out blend diagnostics showed that removing MODEL_X changed the AMEX Metric by DELTA.
```

Only replace `A`, `B`, `C`, `D`, `MODEL_X`, and `DELTA` with values computed from OOF files or visible notebook outputs.

## Current Status

Current public evidence supports:

- final 8-model equal blend: AMEX `0.797631`,
- best verified single model in the public summary: LightGBM Top1600 DART AMEX `0.795134`,
- multiple time representations were included in the final blend.

Current public evidence does not support:

- exact basic-summary-only AMEX metric,
- exact temporal-only incremental lift,
- exact recent-window-only incremental lift,
- exact model correlation matrix,
- exact leave-one-out contribution table.
