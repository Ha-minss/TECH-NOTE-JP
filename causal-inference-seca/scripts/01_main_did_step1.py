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
from step1_did.prep import prepare_base, step1_window, add_step1_did, to_panel_index
from step1_did.models import fit_did, extract

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)
    d = step1_window(d)
    d = add_step1_did(d)
    df0 = to_panel_index(d)

    M0 = []
    M1 = ["RH_proxy", CFG.MSL, CFG.BLH]
    M2 = ["RH_proxy", CFG.MSL, CFG.BLH, "log_Diesel_lag1", "log_Thermal_lag1", "IndustrialIndex_lag1"]

    specs = [("M0_FE_only", M0), ("M1_FE_Weather", M1), ("M2_FE_Weather_EconLag1", M2)]

    rows=[]
    for name, ctrls in specs:
        xcols = ["did"] + ctrls
        res, stats, _ = fit_did(df0, "Y", xcols)
        did = extract(res)

        rows.append({
            "Model": name,
            "Controls": ("[NONE]" if not ctrls else ", ".join(ctrls)),
            "beta(did)": did["beta"],
            "se(cluster station)": did["se"],
            "p(cluster station)": did["p"],
            "%(exp(beta)-1)": (np.exp(did["beta"])-1)*100,
            **stats,
        })

        print("\n" + "="*110)
        print(name, "|", rows[-1]["Controls"])
        print(f"beta={did['beta']:+.6f} se={did['se']:.6f} p={did['p']:.6g} | nobs={stats['nobs']}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT/"main_did_step1.csv", index=False)
    out.to_csv(OUT/"main_did_step1.tsv", index=False, sep="\t")  # Word-friendly
    print("\n✅ Saved:", OUT/"main_did_step1.csv", "and", OUT/"main_did_step1.tsv")

if __name__ == "__main__":
    main()
