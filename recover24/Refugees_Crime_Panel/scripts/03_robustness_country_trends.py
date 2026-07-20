from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

Y_LIST = ['log_Total_per_100k', 'log_Violent_per_100k', 'log_Property_per_100k', 'log_Narcotic_per_100k', 'log_Sexual_per_100k']
CONTROLS = ['GDP_per_pss', 'Social_Protection_per_GDP', 'Unemployment_per_Population', 'log_HigherEdu_per_100k', 'log_Police_per_100k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/panel_processed.csv")
    ap.add_argument("--out", type=str, default="outputs/tables/robustness_country_trends.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["t"] = df["Year"] - df["Year"].min()

    term = "log_Refugees_per_100k"

    rows = []
    for y in Y_LIST[1:]:
        cols = [y, term, *CONTROLS, "Country", "Year", "t"]
        use = df[cols].dropna()

        rhs = " + ".join([term, *CONTROLS, "C(Country) + C(Year) + C(Country):t"])
        formula = f"{y} ~ {rhs}"
        res = smf.ols(formula, data=use).fit(
            cov_type="cluster", cov_kwds={"groups": use["Country"]}
        )

        rows.append(
            {
                "DV": y,
                "b": float(res.params.get(term, np.nan)),
                "se": float(res.bse.get(term, np.nan)),
                "p": float(res.pvalues.get(term, np.nan)),
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
