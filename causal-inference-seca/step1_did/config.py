from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Config:
    # data paths (relative to repo root)
    DATA_PARQUET: str = "data/df_iv.parquet"
    DATA_CSV: str = "data/df_iv.csv"

    # ids
    STATION: str = "station_id"
    TIME: str = "year_month"

    # outcome
    Y_RAW: str = "SO2_ppb"
    VALIDITY_COL: str = "SO2_validity_rate"

    # treatment definition (your baseline)
    COASTAL_COL: str = "20km, 220km, 0.6"

    # weather
    T2M: str = "t2m_m"
    D2M: str = "d2m_m"
    MSL: str = "msl_m"
    BLH: str = "blh_m"
    SP: str  = "sp_m"   # optional

    # econ (raw col names in df)
    COL_DIESEL: str = "Disel price"
    COL_THERM: str  = "화력발전"
    COL_IND: str    = "mining and manufacturing(heavy industry)"

    # step1 window
    START: pd.Timestamp = pd.Timestamp("2017-01-01")
    END: pd.Timestamp   = pd.Timestamp("2021-12-01")
    STEP1: pd.Timestamp = pd.Timestamp("2020-09-01")

    # optional validity filter
    VALIDITY_FILTER: bool = True
    VALIDITY_TH: float = 0.75

    # event study
    ES_WINDOW_MIN: int = -12
    ES_WINDOW_MAX: int = 12
    ES_REF_K: int = -1

CFG = Config()
