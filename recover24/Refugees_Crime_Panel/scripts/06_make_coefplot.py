from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, default="outputs/tables/main_twfe_refugees_only.csv")
    ap.add_argument("--out", type=str, default="outputs/figures/coefplot_main_twfe.png")
    args = ap.parse_args()

    df = pd.read_csv(args.table)

    order = list(df["DV"])
    y = np.arange(len(order))

    b = df["b"].to_numpy()
    se = df["se"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(b, y, xerr=1.96 * se, fmt="o")
    ax.axvline(0.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Coefficient (elasticity) on log(Refugees per 100k)")
    ax.set_title("TWFE estimates (Country + Year FE), clustered SE by Country")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
