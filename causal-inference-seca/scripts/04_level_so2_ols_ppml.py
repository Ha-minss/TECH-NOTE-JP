from pathlib import Path
import sys

# Allow running scripts via: python scripts/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

from step1_did.config import CFG
from step1_did.io import load_df
from step1_did.prep import prepare_base, step1_window, add_step1_did

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)

    # Step1 window + did
    d = step1_window(d)
    d = add_step1_did(d)

    # M2 controls
    CTRL_M2 = ["RH_proxy", CFG.MSL, CFG.BLH, "log_Diesel_lag1", "log_Thermal_lag1", "IndustrialIndex_lag1"]
    CTRL_M2 = [c for c in CTRL_M2 if c in d.columns]

    # outcomes
    d["Y_level"] = pd.to_numeric(d[CFG.Y_RAW], errors="coerce")
    d["Y_ppml"]  = pd.to_numeric(d[CFG.Y_RAW], errors="coerce").fillna(0.0)

    # OLS on LEVEL
    d_ols = d.dropna(subset=["Y_level","did"] + CTRL_M2).copy()
    df_ols = d_ols.set_index([CFG.STATION, CFG.TIME]).sort_index()

    X_ols = df_ols[["did"] + CTRL_M2].astype(float)
    y_ols = df_ols["Y_level"].astype(float)

    mod_ols = PanelOLS(y_ols, X_ols, entity_effects=True, time_effects=True,
                      drop_absorbed=True, check_rank=False)
    res_ols = mod_ols.fit(cov_type="clustered", cluster_entity=True)

    b_ols  = float(res_ols.params.get("did", np.nan))
    se_ols = float(res_ols.std_errors.get("did", np.nan))
    p_ols  = float(res_ols.pvalues.get("did", np.nan))

    # PPML on LEVEL (Poisson GLM + FE dummies)
    d_ppml = d.dropna(subset=["did"] + CTRL_M2).copy()
    d_ppml["Y_ppml"] = d_ppml["Y_ppml"].fillna(0.0)

    st_d = pd.get_dummies(d_ppml[CFG.STATION].astype(str), prefix="st", drop_first=True, dtype=float)
    tm_d = pd.get_dummies(d_ppml[CFG.TIME].astype(str), prefix="tm", drop_first=True, dtype=float)

    X_ppml = pd.concat([d_ppml[["did"] + CTRL_M2], st_d, tm_d], axis=1)
    X_ppml = X_ppml.apply(pd.to_numeric, errors="coerce").astype(float)
    y_ppml = pd.to_numeric(d_ppml["Y_ppml"], errors="coerce").astype(float)

    X_ppml = sm.add_constant(X_ppml, has_constant="add")
    glm = sm.GLM(y_ppml, X_ppml, family=sm.families.Poisson())
    res_ppml = glm.fit(cov_type="cluster", cov_kwds={"groups": d_ppml[CFG.STATION].astype(str)})

    b_ppml  = float(res_ppml.params.get("did", np.nan))
    se_ppml = float(res_ppml.bse.get("did", np.nan))
    p_ppml  = float(res_ppml.pvalues.get("did", np.nan))

    out = pd.DataFrame([{
        "Spec": "OLS LEVEL (ppb) | FE st+ym | cluster(station)",
        "beta(did)": b_ols, "se": se_ols, "p": p_ols,
        "nobs": len(df_ols),
        "stations": int(df_ols.index.get_level_values(0).nunique()),
        "months": int(df_ols.index.get_level_values(1).nunique()),
        "notes": "PanelOLS on raw SO2_ppb",
    },{
        "Spec": "PPML LEVEL (ppb) | FE st+ym dummies | cluster(station)",
        "beta(did)": b_ppml, "se": se_ppml, "p": p_ppml,
        "nobs": int(len(d_ppml)),
        "stations": int(d_ppml[CFG.STATION].nunique()),
        "months": int(d_ppml[CFG.TIME].nunique()),
        "notes": "% effect uses exp(beta)-1",
        "%(exp(beta)-1)": (np.exp(b_ppml)-1)*100 if np.isfinite(b_ppml) else np.nan
    }])

    out.to_csv(OUT/"level_so2_ols_ppml_m2.csv", index=False)
    out.to_csv(OUT/"level_so2_ols_ppml_m2.tsv", index=False, sep="\t")

    print("\n" + "="*100)
    print("A) OLS LEVEL SO2 | FE st+ym | cluster(station)")
    print("="*100)
    print(res_ols.summary)

    print("\n" + "="*100)
    print("B) PPML LEVEL SO2 | FE dummies | cluster(station)")
    print("="*100)
    print("beta(did)=", b_ppml, "se=", se_ppml, "p=", p_ppml, " %(exp-1)=", (np.exp(b_ppml)-1)*100)

    print("\n✅ Saved:", OUT/"level_so2_ols_ppml_m2.csv")

if __name__ == "__main__":
    main()
