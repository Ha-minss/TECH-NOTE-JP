from __future__ import annotations

import numpy as np
import pandas as pd


def add_rates_and_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-100k rates and log variables used across scripts."""
    out = df.copy()

    out["Total_Crimes"] = (
        out["Violent_Crimes"]
        + out["Property_Crimes"]
        + out["Narcotic_Crimes"]
        + out["Sexual_Crimes"]
    )

    scale = 100_000.0
    out["Total_per_100k"] = out["Total_Crimes"] / out["Population"] * scale
    out["Violent_per_100k"] = out["Violent_Crimes"] / out["Population"] * scale
    out["Property_per_100k"] = out["Property_Crimes"] / out["Population"] * scale
    out["Narcotic_per_100k"] = out["Narcotic_Crimes"] / out["Population"] * scale
    out["Sexual_per_100k"] = out["Sexual_Crimes"] / out["Population"] * scale

    out["Refugees_per_100k"] = out["Refugees"] / out["Population"] * scale
    out["Immigrants_per_100k"] = out["Immigrants"] / out["Population"] * scale

    out["HigherEdu_per_100k"] = out["Higher_Education_Participation"] / out["Population"] * scale
    out["Police_per_100k"] = out["Police_Officers"] / out["Population"] * scale

    log_map = {
        "log_Total_per_100k": "Total_per_100k",
        "log_Violent_per_100k": "Violent_per_100k",
        "log_Property_per_100k": "Property_per_100k",
        "log_Narcotic_per_100k": "Narcotic_per_100k",
        "log_Sexual_per_100k": "Sexual_per_100k",
        "log_Refugees_per_100k": "Refugees_per_100k",
        "log_Immigrants_per_100k": "Immigrants_per_100k",
        "log_HigherEdu_per_100k": "HigherEdu_per_100k",
        "log_Police_per_100k": "Police_per_100k",
    }
    for newc, oldc in log_map.items():
        out[newc] = np.where(out[oldc] > 0, np.log(out[oldc]), np.nan)

    return out


def add_lags_leads(df: pd.DataFrame, col: str, lags=(1, 2), leads=(1,)) -> pd.DataFrame:
    out = df.sort_values(["Country", "Year"]).copy()
    g = out.groupby("Country", sort=False)
    for k in lags:
        out[f"{col}_lag{k}"] = g[col].shift(k)
    for k in leads:
        out[f"{col}_lead{k}"] = g[col].shift(-k)
    return out


def first_difference(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["Country", "Year"]).copy()
    g = out.groupby("Country", sort=False)
    for c in cols:
        out[f"d_{c}"] = g[c].diff()
    return out
