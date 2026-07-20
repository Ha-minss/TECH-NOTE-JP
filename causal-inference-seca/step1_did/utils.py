import numpy as np
import pandas as pd

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def make_log_safe(x):
    x = safe_num(x)
    x = x.where(x > 0, np.nan)
    return np.log(x)

def add_lags(df, cols, lags=(1,), entity="station_id", time="year_month"):
    out = df.sort_values([entity, time]).copy()
    for v in cols:
        if v not in out.columns:
            continue
        for L in lags:
            out[f"{v}_lag{L}"] = out.groupby(entity)[v].shift(L)
    return out

def month_diff(dt_series, anchor_date):
    dt = pd.to_datetime(dt_series)
    a  = pd.to_datetime(anchor_date)
    return (dt.dt.year - a.year) * 12 + (dt.dt.month - a.month)

def must_have(df, cols, where=""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss} {where}")

def used_panel_stats(idx):
    st = idx.get_level_values(0)
    tm = idx.get_level_values(1)
    return {
        "nobs": int(len(idx)),
        "stations": int(pd.Index(st).nunique()),
        "months": int(pd.Index(tm).nunique()),
        "min_month": str(pd.to_datetime(tm.min()).date()),
        "max_month": str(pd.to_datetime(tm.max()).date()),
    }
