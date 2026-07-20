from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import pandas as pd

from src.ref_crime.features import add_lags_leads
from src.ref_crime.regressions import run_twfe_ols, tidy_term

Y_LIST = ['log_Total_per_100k', 'log_Violent_per_100k', 'log_Property_per_100k', 'log_Narcotic_per_100k', 'log_Sexual_per_100k']
CONTROLS = ['GDP_per_pss', 'Social_Protection_per_GDP', 'Unemployment_per_Population', 'log_HigherEdu_per_100k', 'log_Police_per_100k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/panel_processed.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/tables")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = add_lags_leads(df, col="log_Refugees_per_100k", lags=(1, 2), leads=(1,))

    blocks = [
        ("curr", "log_Refugees_per_100k"),
        ("lag1", "log_Refugees_per_100k_lag1"),
        ("lag2", "log_Refugees_per_100k_lag2"),
        ("lead1_placebo", "log_Refugees_per_100k_lead1"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    for lab, term in blocks:
        rows = []
        for y in Y_LIST[1:]:
            cols = [y, term, *CONTROLS, "Country", "Year"]
            use = df[cols].dropna()
            res, _ = run_twfe_ols(use, y=y, x_terms=[term], controls=CONTROLS)
            t = tidy_term(res, term)
            rows.append({"block": lab, "DV": y, "term": term, **t})

        out = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, "dynamic_" + lab + ".csv")
        out.to_csv(out_path, index=False)
        print(f"[OK] wrote: {out_path}")

    # Joint model on common sample per DV
    terms = [
        "log_Refugees_per_100k",
        "log_Refugees_per_100k_lag1",
        "log_Refugees_per_100k_lag2",
        "log_Refugees_per_100k_lead1",
    ]

    rows = []
    for y in Y_LIST[1:]:
        cols = [y, *terms, *CONTROLS, "Country", "Year"]
        use = df[cols].dropna()
        res, _ = run_twfe_ols(use, y=y, x_terms=terms, controls=CONTROLS)

        row = {"DV": y, "N": int(res.nobs), "R2": float(res.rsquared)}
        for term in terms:
            tt = tidy_term(res, term)
            row["b_" + term] = tt["b"]
            row["se_" + term] = tt["se"]
            row["p_" + term] = tt["p"]
        rows.append(row)

    out_joint = pd.DataFrame(rows)
    out_joint_path = os.path.join(args.out_dir, "dynamic_joint_t_lags_lead.csv")
    out_joint.to_csv(out_joint_path, index=False)
    print(f"[OK] wrote: {out_joint_path}")


if __name__ == "__main__":
    main()
