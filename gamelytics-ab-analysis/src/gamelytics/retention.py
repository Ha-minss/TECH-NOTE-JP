from __future__ import annotations

import pandas as pd


def exact_day_retention(
    reg_df: pd.DataFrame,
    auth_df: pd.DataFrame,
    days: list[int],
    observation_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    reg = reg_df[["uid", "reg_ts"]].copy()
    auth = auth_df[["uid", "auth_ts"]].copy()
    reg["reg_date"] = pd.to_datetime(reg["reg_ts"], unit="s").dt.floor("D")
    auth["auth_date"] = pd.to_datetime(auth["auth_ts"], unit="s").dt.floor("D")
    observation_end_ts = (
        pd.Timestamp(observation_end).floor("D")
        if observation_end is not None
        else auth["auth_date"].max().floor("D")
    )

    logins = auth.drop_duplicates(["uid", "auth_date"]).merge(
        reg[["uid", "reg_date"]], on="uid", how="inner"
    )
    logins["day_n"] = (logins["auth_date"] - logins["reg_date"]).dt.days
    rows = []
    for day in days:
        eligible = reg[reg["reg_date"] <= observation_end_ts - pd.Timedelta(days=day)]
        eligible_users = eligible["uid"].nunique()
        retained_users = logins[
            (logins["day_n"] == day) & (logins["uid"].isin(eligible["uid"]))
        ]["uid"].nunique()
        rows.append(
            {
                "day": day,
                "eligible_users": eligible_users,
                "retained_users": retained_users,
                "retention_rate": retained_users / eligible_users if eligible_users else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("day")


def monthly_retention_heatmap_data(
    reg_df: pd.DataFrame,
    auth_df: pd.DataFrame,
    days: list[int],
    min_cohort_size: int = 100,
) -> pd.DataFrame:
    reg = reg_df[["uid", "reg_ts"]].copy()
    auth = auth_df[["uid", "auth_ts"]].copy()
    reg["reg_date"] = pd.to_datetime(reg["reg_ts"], unit="s").dt.floor("D")
    reg["cohort_month"] = reg["reg_date"].dt.to_period("M").astype(str)
    auth["auth_date"] = pd.to_datetime(auth["auth_ts"], unit="s").dt.floor("D")
    observation_end = auth["auth_date"].max().floor("D")
    logins = auth.drop_duplicates(["uid", "auth_date"]).merge(
        reg[["uid", "reg_date", "cohort_month"]], on="uid", how="inner"
    )
    logins["day_n"] = (logins["auth_date"] - logins["reg_date"]).dt.days
    rows = []
    for month, cohort in reg.groupby("cohort_month"):
        for day in days:
            eligible = cohort[cohort["reg_date"] <= observation_end - pd.Timedelta(days=day)]
            if len(eligible) < min_cohort_size:
                continue
            retained = logins[
                (logins["cohort_month"] == month)
                & (logins["day_n"] == day)
                & (logins["uid"].isin(eligible["uid"]))
            ]["uid"].nunique()
            rows.append(
                {
                    "cohort_month": month,
                    "day": f"D{day}",
                    "cohort_size": len(eligible),
                    "retention_rate": retained / len(eligible),
                }
            )
    return pd.DataFrame(rows)
