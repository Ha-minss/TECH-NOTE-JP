from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pandas as pd
import statsmodels.formula.api as smf

from src.ref_crime.features import first_difference

Y_LIST = ['log_Total_per_100k', 'log_Violent_per_100k', 'log_Property_per_100k', 'log_Narcotic_per_100k', 'log_Sexual_per_100k']
CONTROLS = ['GDP_per_pss', 'Social_Protection_per_GDP', 'Unemployment_per_Population', 'log_HigherEdu_per_100k', 'log_Police_per_100k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/panel_processed.csv")
    ap.add_argument("--out", type=str, default="outputs/tables/first_difference_year_fe.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    base_cols = ["log_Refugees_per_100k", *CONTROLS, *Y_LIST, "Country", "Year"]
    df = df[base_cols].dropna(subset=["Country", "Year"])

    diff_cols = ["log_Refugees_per_100k", *CONTROLS, *Y_LIST]
    dfd = first_difference(df, cols=diff_cols)

    term = "d_log_Refugees_per_100k"
    ctrl_d = ["d_" + c for c in CONTROLS]

    rows = []
    for y in Y_LIST[1:]:
        dy = "d_" + y
        cols = [dy, term, *ctrl_d, "Country", "Year"]
        use = dfd[cols].dropna()

        rhs = " + ".join([term, *ctrl_d, "C(Year)"])
        formula = f"{dy} ~ {rhs}"
        res = smf.ols(formula, data=use).fit(
            cov_type="cluster", cov_kwds={"groups": use["Country"]}
        )

        rows.append(
            {
                "DV": y,
                "b": float(res.params.get(term)),
                "se": float(res.bse.get(term)),
                "p": float(res.pvalues.get(term)),
                "N": int(res.nobs),
                "R2": float(res.rsquared),
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote: {args.out}")
    print(out)


if __name__ == "__main__":
    main()
