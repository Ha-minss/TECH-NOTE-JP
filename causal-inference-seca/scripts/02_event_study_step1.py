from pathlib import Path
import sys

# Allow running scripts via: python scripts/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

from step1_did.config import CFG
from step1_did.io import load_df
from step1_did.prep import prepare_base, step1_window, to_panel_index
from step1_did.utils import month_diff, used_panel_stats

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)
    d = step1_window(d)

    # M2 controls only
    ctrls = ["RH_proxy", CFG.MSL, CFG.BLH, "log_Diesel_lag1", "log_Thermal_lag1", "IndustrialIndex_lag1"]
    ctrls = [c for c in ctrls if c in d.columns]

    d = d.copy()
    d["k"] = month_diff(d[CFG.TIME], CFG.STEP1).astype(int)
    d = d[d["k"].between(CFG.ES_WINDOW_MIN, CFG.ES_WINDOW_MAX)].copy()

    ks = [k for k in range(CFG.ES_WINDOW_MIN, CFG.ES_WINDOW_MAX+1) if k != CFG.ES_REF_K]
    es_cols=[]
    for k in ks:
        c=f"es_{k}"
        d[c] = ((d["k"]==k).astype(float) * d["Coastal"].astype(float))
        es_cols.append(c)

    df0 = to_panel_index(d)
    d_used = df0.dropna(subset=["Y"] + ctrls + es_cols).copy()

    y = d_used["Y"]
    X = d_used[es_cols + ctrls].astype(float)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    stats = used_panel_stats(d_used.index)
    print("\n" + "="*120)
    print("EVENT STUDY Step1 (anchor=2020-09, REF=-1) | cluster(station)")
    print("="*120)
    print(stats)
    print(res.summary)

    rows=[]
    for k in ks:
        name=f"es_{k}"
        b=float(res.params.get(name, np.nan))
        se=float(res.std_errors.get(name, np.nan))
        p=float(res.pvalues.get(name, np.nan))
        rows.append([k,b,se,p])
    tab=pd.DataFrame(rows, columns=["k","coef","se","p"]).sort_values("k")
    tab["ci_l"]=tab["coef"]-1.96*tab["se"]
    tab["ci_u"]=tab["coef"]+1.96*tab["se"]
    tab["%(exp-1)"]=(np.exp(tab["coef"])-1)*100
    tab.to_csv(OUT/"event_study_step1_M2.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2,3.6))
    ax.axhline(0,color="black",lw=1)
    ax.axvline(0,color="red",lw=1.3)          # policy start
    ax.axvline(CFG.ES_REF_K,color="gray",lw=1,ls="--")  # reference period

    x=tab["k"].to_numpy()
    b=tab["coef"].to_numpy()
    lo=tab["ci_l"].to_numpy()
    hi=tab["ci_u"].to_numpy()
    yerr=np.vstack([b-lo, hi-b])

    ax.errorbar(x,b,yerr=yerr,fmt="none",elinewidth=1,capsize=2.5,alpha=0.85)
    ax.plot(x,b,"o-",ms=3.8,lw=2)

    ax.set_xlabel("Months relative to policy (k)")
    ax.set_ylabel("Coef on log(1+SO2)")
    ax.set_title("Event Study (Step1, M2)")
    ax.grid(True,axis="y",ls="--",alpha=0.85)
    fig.tight_layout()
    plt.savefig(OUT/"event_study_step1_M2.png", dpi=300)
    plt.savefig(OUT/"event_study_step1_M2.pdf")
    print("✅ Saved:", OUT/"event_study_step1_M2.csv", OUT/"event_study_step1_M2.png")

if __name__=="__main__":
    main()
