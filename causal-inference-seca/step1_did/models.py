import numpy as np
from linearmodels.panel import PanelOLS
from .utils import used_panel_stats

def fit_did(df_panel, y_col, x_cols):
    d_used = df_panel.dropna(subset=[y_col] + x_cols).copy()

    y = d_used[y_col].astype(float)
    X = d_used[x_cols].astype(float)

    mod = PanelOLS(
        y, X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)  # ✅ 1-way station cluster
    stats = used_panel_stats(d_used.index)
    return res, stats, d_used

def extract(res, term="did"):
    if term not in res.params.index:
        return {"beta": np.nan, "se": np.nan, "p": np.nan, "t": np.nan}
    return {
        "beta": float(res.params[term]),
        "se": float(res.std_errors[term]),
        "p": float(res.pvalues[term]),
        "t": float(res.tstats[term]),
    }
