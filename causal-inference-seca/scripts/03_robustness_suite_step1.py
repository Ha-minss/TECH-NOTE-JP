from pathlib import Path
import sys

# Allow running scripts via: python scripts/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from step1_did.config import CFG
from step1_did.io import load_df
from step1_did.prep import prepare_base, to_panel_index
from step1_did.utils import add_lags
from step1_did.models import fit_did, extract
from step1_did.wild import wild_cluster_pvalue_two_way_fe

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

PLACEBO_DATES = ["2018-09-01", "2019-03-01", "2019-09-01"]

def run_one(df_panel, name, xcols, do_wild=False, wild_B=399, wild_seed=2025):
    res, stats, d_used = fit_did(df_panel, "Y", xcols)
    did = extract(res)

    p_wild = np.nan
    t_wild = np.nan
    if do_wild:
        try:
            t_wild, p_wild = wild_cluster_pvalue_two_way_fe(
                df_used_panel=d_used,
                ycol="Y",
                xcols_full=xcols,
                did_name="did",
                B=wild_B,
                seed=wild_seed,
                entity_name=CFG.STATION,
                time_name=CFG.TIME,
            )
        except Exception as e:
            print(f"[WILD FAIL] {name}: {type(e).__name__}: {e}")
            p_wild = np.nan
            t_wild = np.nan

    row = {
        "Model": name,
        "beta(did)": did["beta"],
        "se(cluster station)": did["se"],
        "p(cluster station)": did["p"],
        "wild_t(two-way demeaned OLS)": t_wild,
        "wild_p(station)": p_wild,
        "B_wild": (wild_B if pd.notna(p_wild) else np.nan),
        "%(exp(beta)-1)": (np.exp(did["beta"])-1)*100,
        **stats,
    }
    msg = f"[OK] {name} | beta={did['beta']:+.4f} se={did['se']:.4f} p={did['p']:.4g} | nobs={stats['nobs']}"
    if pd.notna(p_wild):
        msg += f" | wild_p={p_wild:.4g} (B={wild_B})"
    print(msg)
    return row

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)

    # add lag2 for robustness
    d = add_lags(d, ["log_Diesel","log_Thermal","IndustrialIndex"], lags=(2,),
                 entity=CFG.STATION, time=CFG.TIME)

    # Step1 sample (2017-01~2021-12)
    dA = d[(d[CFG.TIME] >= CFG.START) & (d[CFG.TIME] <= CFG.END)].copy()
    dA["Post"] = (dA[CFG.TIME] >= CFG.STEP1).astype(int)
    dA["did"]  = (dA["Coastal"] * dA["Post"]).astype(float)
    dfA = to_panel_index(dA)

    econ_lag1  = ["log_Diesel_lag1","log_Thermal_lag1","IndustrialIndex_lag1"]
    econ_lag12 = ["log_Diesel_lag1","log_Thermal_lag1","IndustrialIndex_lag1",
                  "log_Diesel_lag2","log_Thermal_lag2","IndustrialIndex_lag2"]
    econ_lag1 = [c for c in econ_lag1 if c in dfA.columns]
    econ_lag12 = [c for c in econ_lag12 if c in dfA.columns]

    W1 = ["RH_proxy", CFG.MSL, CFG.BLH]
    W2 = [CFG.T2M, CFG.D2M, CFG.MSL, CFG.BLH]
    W3 = ["RH_proxy", CFG.SP, CFG.BLH]
    W4 = [CFG.BLH]
    weather_specs = {
        "W1(RH+msl+blh)": [c for c in W1 if c in dfA.columns],
        "W2(t2m+d2m+msl+blh)": [c for c in W2 if c in dfA.columns],
        "W3(RH+sp+blh)": [c for c in W3 if c in dfA.columns],
        "W4(blh only)": [c for c in W4 if c in dfA.columns],
    }

    rows = []

    # (0) Baseline + wild p
    base_x = ["did"] + weather_specs["W1(RH+msl+blh)"] + econ_lag1
    rows.append(run_one(dfA, "0 Baseline: W1 + econ lag1", base_x, do_wild=True, wild_B=399, wild_seed=2025))

    # (1) Econ lag2
    x_lag12 = ["did"] + weather_specs["W1(RH+msl+blh)"] + econ_lag12
    rows.append(run_one(dfA, "1 Econ: W1 + econ lag1+lag2", x_lag12, do_wild=False))

    # (2) Weather alternatives (econ lag1 fixed)
    for wname, wcols in weather_specs.items():
        x = ["did"] + wcols + econ_lag1
        rows.append(run_one(dfA, f"2 Weather: {wname} + econ lag1", x, do_wild=False))

    # (3) Placebos (pre-Step1 only, fake post dates)
    pre = d[d[CFG.TIME] < CFG.STEP1].copy()
    pre = pre[(pre[CFG.TIME] >= CFG.START) & (pre[CFG.TIME] <= pd.Timestamp('2020-08-01'))].copy()
    for pd_str in PLACEBO_DATES:
        pdt = pd.Timestamp(pd_str)
        dp = pre.copy()
        dp["PostP"] = (dp[CFG.TIME] >= pdt).astype(int)
        dp["did"] = (dp["Coastal"] * dp["PostP"]).astype(float)
        dfP = to_panel_index(dp)

        x = ["did"] + weather_specs["W1(RH+msl+blh)"] + econ_lag1
        rows.append(run_one(dfP, f"3 Placebo: fake={pd_str} (pre-Step1)", x, do_wild=False))

    out = pd.DataFrame(rows)
    out.to_csv(OUT/"robustness_suite_step1.csv", index=False)
    out.to_csv(OUT/"robustness_suite_step1.tsv", index=False, sep="\t")
    print("\n✅ Saved:", OUT/"robustness_suite_step1.csv")

if __name__ == "__main__":
    main()
