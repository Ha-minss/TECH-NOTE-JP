from pathlib import Path
import sys

# Allow running scripts via: python scripts/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from step1_did.config import CFG
from step1_did.io import load_df
from step1_did.prep import prepare_base, step1_window

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def main():
    df_raw = load_df()
    d = prepare_base(df_raw)
    d = step1_window(d)

    # Keep only columns needed for Step1 scripts (plus raw identifiers)
    keep = [
        CFG.STATION, CFG.TIME, CFG.Y_RAW, CFG.VALIDITY_COL, CFG.COASTAL_COL,
        CFG.T2M, CFG.D2M, CFG.MSL, CFG.BLH, CFG.SP,
        CFG.COL_DIESEL, CFG.COL_THERM, CFG.COL_IND,
        "Y", "Coastal", "RH_proxy",
        "log_Diesel", "log_Thermal", "IndustrialIndex",
        "log_Diesel_lag1", "log_Thermal_lag1", "IndustrialIndex_lag1",
    ]
    keep = [c for c in keep if c in d.columns]
    d_out = d[keep].copy()

    out_parq = OUT / "df_iv_step1.parquet"
    out_csv  = OUT / "df_iv_step1.csv"
    d_out.to_parquet(out_parq, index=False)
    d_out.to_csv(out_csv, index=False)

    print("✅ Saved trimmed Step1 dataset:")
    print(" -", out_parq)
    print(" -", out_csv)
    print(f"   rows={len(d_out)} cols={len(d_out.columns)}")

if __name__ == "__main__":
    main()
