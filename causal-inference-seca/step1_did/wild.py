import numpy as np
import pandas as pd
import statsmodels.api as sm

def two_way_demean(df, cols, entity, time):
    out = df.copy()
    for c in cols:
        g_i  = out.groupby(entity)[c].transform("mean")
        g_t  = out.groupby(time)[c].transform("mean")
        g_all = out[c].mean()
        out[c + "_tw"] = out[c] - g_i - g_t + g_all
    return out

def wild_cluster_pvalue_two_way_fe(df_used_panel, ycol, xcols_full, did_name="did",
                                  B=399, seed=2025,
                                  entity_name="station_id", time_name="year_month"):
    """
    Wild cluster bootstrap p-value (Rademacher weights by station) after two-way demeaning.
    - H0: beta_did = 0
    - cluster: station
    - df_used_panel can be a MultiIndex (station_id, year_month)
    """
    rng = np.random.default_rng(seed)

    d = df_used_panel.reset_index().copy()
    d["cluster_station"] = d[entity_name].astype(str)

    tw_cols = [ycol] + xcols_full
    d = two_way_demean(d, tw_cols, entity=entity_name, time=time_name)

    y_tw = d[ycol + "_tw"].to_numpy()
    X_full_tw = d[[c + "_tw" for c in xcols_full]].to_numpy()
    j_did = xcols_full.index(did_name)

    xcols0 = [c for c in xcols_full if c != did_name]
    X0_tw = d[[c + "_tw" for c in xcols0]].to_numpy()

    g = d["cluster_station"].to_numpy()
    uniq = pd.unique(g)

    res_obs = sm.OLS(y_tw, X_full_tw).fit(cov_type="cluster", cov_kwds={"groups": g})
    t_obs = float(res_obs.tvalues[j_did])

    res0 = sm.OLS(y_tw, X0_tw).fit(cov_type="cluster", cov_kwds={"groups": g})
    yhat0 = res0.fittedvalues
    u0 = res0.resid

    t_star = np.empty(B, dtype=float)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(uniq))
        w_map = dict(zip(uniq, w))
        wb = np.vectorize(w_map.get)(g)

        y_star = yhat0 + u0 * wb
        res_b = sm.OLS(y_star, X_full_tw).fit(cov_type="cluster", cov_kwds={"groups": g})
        t_star[b] = float(res_b.tvalues[j_did])

    p_wild = (np.sum(np.abs(t_star) >= np.abs(t_obs)) + 1) / (B + 1)
    return float(t_obs), float(p_wild)
