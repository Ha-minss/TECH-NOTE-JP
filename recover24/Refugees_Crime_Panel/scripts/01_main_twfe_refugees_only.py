from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pandas as pd

from src.ref_crime.regressions import run_twfe_ols, tidy_term

Y_LIST = ['log_Total_per_100k', 'log_Violent_per_100k', 'log_Property_per_100k', 'log_Narcotic_per_100k', 'log_Sexual_per_100k']
CONTROLS = ['GDP_per_pss', 'Social_Protection_per_GDP', 'Unemployment_per_Population', 'log_HigherEdu_per_100k', 'log_Police_per_100k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/panel_processed.csv")
    ap.add_argument("--out", type=str, default="outputs/tables/main_twfe_refugees_only.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    term = "log_Refugees_per_100k"
    rows = []
    for y in Y_LIST:
        use = df[[y, term, *CONTROLS, "Country", "Year"]].dropna()
        res, _ = run_twfe_ols(use, y=y, x_terms=[term], controls=CONTROLS)
        t = tidy_term(res, term)
        rows.append({"DV": y, **t})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"[OK] wrote: {args.out}")
    print(out)


if __name__ == "__main__":
    main()
