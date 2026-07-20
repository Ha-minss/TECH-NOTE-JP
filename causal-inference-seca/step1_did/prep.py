import numpy as np
import pandas as pd
from .config import CFG
from .utils import safe_num, make_log_safe, add_lags, must_have

def prepare_base(df_raw: pd.DataFrame) -> pd.DataFrame:
    need = [
        CFG.STATION, CFG.TIME, CFG.Y_RAW, CFG.COASTAL_COL,
        CFG.T2M, CFG.D2M, CFG.MSL, CFG.BLH,
        CFG.COL_DIESEL, CFG.COL_THERM, CFG.COL_IND,
    ]
    must_have(df_raw, need, where="(prepare_base)")

    d = df_raw.copy()
    d[CFG.TIME] = pd.to_datetime(d[CFG.TIME], errors="coerce").values.astype("datetime64[M]")
    d[CFG.STATION] = d[CFG.STATION].astype(str)

    if CFG.VALIDITY_FILTER and (CFG.VALIDITY_COL in d.columns):
        d = d[safe_num(d[CFG.VALIDITY_COL]) >= CFG.VALIDITY_TH].copy()

    # main outcome (log1p)
    d["Y"] = np.log1p(safe_num(d[CFG.Y_RAW]))

    # treatment flag
    d["Coastal"] = safe_num(d[CFG.COASTAL_COL]).fillna(0).astype(int)

    # weather proxy (dew-point depression)
    d["RH_proxy"] = safe_num(d[CFG.T2M]) - safe_num(d[CFG.D2M])

    # econ transforms (log safe)
    d["log_Diesel"] = make_log_safe(d[CFG.COL_DIESEL])
    d["log_Thermal"] = make_log_safe(d[CFG.COL_THERM])
    d["IndustrialIndex"] = safe_num(d[CFG.COL_IND])

    # lag1 (baseline)
    d = add_lags(d, ["log_Diesel","log_Thermal","IndustrialIndex"], lags=(1,),
                 entity=CFG.STATION, time=CFG.TIME)
    return d

def step1_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df[CFG.TIME] >= CFG.START) & (df[CFG.TIME] <= CFG.END)].copy()

def add_step1_did(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Post"] = (d[CFG.TIME] >= CFG.STEP1).astype(int)
    d["did"] = (d["Coastal"] * d["Post"]).astype(float)
    return d

def to_panel_index(df: pd.DataFrame):
    return df.set_index([CFG.STATION, CFG.TIME]).sort_index()
