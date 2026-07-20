from pathlib import Path
import sys

# Allow running scripts via: python scripts/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from step1_did.config import CFG
from step1_did.io import load_df
from step1_did.prep import prepare_base
from step1_did.utils import safe_num, add_lags
from step1_did.wild import wild_cluster_pvalue_two_way_fe

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# --- user-provided columns (change if needed) ---
COL_COAST_DIST = "해안선까지 찐찐거리"
COL_BUSAN_DIST = "부산까지 거리"
COL_OPENNESS   = "해안개방성점수"

# treat specs: (name, coast_km, busan_km, open_th)
TREAT_SPECS = [
    ("Coast30 Busan220 Open0.60", 30, 220, 0.60),
    ("Coast30 Busan230 Open0.65", 30, 230, 0.65),
    ("Narrow10 Busan220 Open0.60", 10, 220, 0.60),
    ("Open↓20 Busan220 Open0.55", 20, 220, 0.55),
]

DO_WILD_FOR_BASELINE = True
WILD_B = 399
WILD_SEED = 123

def infer_km(series):
    s = safe_num(series)
    med = np.nanmedian(s.values)
    return (s/1000.0) if (np.isfinite(med) and med > 500) else s

def make_treat(df, coast_km, busan_km, open_th):
    return ((df["coast_km"] <= coast_km) &
            (df["busan_km"] <= busan_km) &
            (df["open"] >= open_th)).astype(int)

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)

    # require distance/open columns
    missing = [c for c in [COL_COAST_DIST, COL_BUSAN_DIST, COL_OPENNESS] if c not in d.columns]
    if missing:
        raise KeyError(
            f"Missing columns for treatment definitions: {missing}. "
            f"Edit scripts/05_treatment_defs_step1.py to match your actual column names."
        )

    # lag1 safety
    d = add_lags(d, ["log_Diesel","log_Thermal","IndustrialIndex"], lags=(1,),
                 entity=CFG.STATION, time=CFG.TIME)

    # Step1 window
    d = d[(d[CFG.TIME] >= CFG.START) & (d[CFG.TIME] <= CFG.END)].copy()
    d["Post"] = (d[CFG.TIME] >= CFG.STEP1).astype(int)

    # treatment ingredients
    d["coast_km"] = infer_km(d[COL_COAST_DIST])
    d["busan_km"] = infer_km(d[COL_BUSAN_DIST])
    d["open"] = safe_num(d[COL_OPENNESS])

    # controls = M2
    controls = ["RH_proxy", CFG.MSL, CFG.BLH, "log_Diesel_lag1", "log_Thermal_lag1", "IndustrialIndex_lag1"]
    controls = [c for c in controls if c in d.columns]

    rows=[]
    for idx, (nm, ck, bk, ot) in enumerate(TREAT_SPECS):
        dd = d.copy()
        dd["Coastal_tmp"] = make_treat(dd, ck, bk, ot).astype(int)
        dd["did"] = (dd["Coastal_tmp"] * dd["Post"]).astype(float)

        dfp = dd.set_index([CFG.STATION, CFG.TIME]).sort_index()
        xcols = ["did"] + controls
        used = dfp.dropna(subset=["Y"] + xcols).copy()

        mod = PanelOLS(used["Y"], used[xcols].astype(float),
                       entity_effects=True, time_effects=True,
                       drop_absorbed=True, check_rank=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        b = float(res.params.get("did", np.nan))
        se = float(res.std_errors.get("did", np.nan))
        p = float(res.pvalues.get("did", np.nan))
        eff = (np.exp(b)-1)*100

        wild_p = np.nan
        wild_t = np.nan
        if DO_WILD_FOR_BASELINE and idx == 0:
            wild_t, wild_p = wild_cluster_pvalue_two_way_fe(
                df_used_panel=used,
                ycol="Y",
                xcols_full=xcols,
                did_name="did",
                B=WILD_B,
                seed=WILD_SEED,
                entity_name=CFG.STATION,
                time_name=CFG.TIME,
            )

        rows.append({
            "Spec": nm, "coast_km": ck, "busan_km": bk, "open_th": ot,
            "beta(did)": b, "se(cluster station)": se, "p(cluster station)": p,
            "%(exp-1)": eff,
            "wild_t(base)": wild_t, "wild_p(base)": wild_p, "B_wild": (WILD_B if pd.notna(wild_p) else np.nan),
            "nobs": len(used),
            "stations": int(used.index.get_level_values(0).nunique()),
            "months": int(used.index.get_level_values(1).nunique()),
        })
        print(f"[OK] {nm} | beta={b:+.4f} se={se:.4f} p={p:.4g} nobs={len(used)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT/"robustness_treatment_defs_step1.csv", index=False)
    print("✅ Saved:", OUT/"robustness_treatment_defs_step1.csv")

if __name__ == "__main__":
    main()
