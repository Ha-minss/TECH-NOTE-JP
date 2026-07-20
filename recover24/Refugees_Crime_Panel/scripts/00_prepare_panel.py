from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pandas as pd

from src.ref_crime.features import add_rates_and_logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="data/raw/panel_raw.csv")
    ap.add_argument("--outfile", type=str, default="data/processed/panel_processed.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    df2 = add_rates_and_logs(df)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df2.to_csv(args.outfile, index=False)

    print(f"[OK] wrote: {args.outfile}  rows={len(df2)} cols={df2.shape[1]}")


if __name__ == "__main__":
    main()
