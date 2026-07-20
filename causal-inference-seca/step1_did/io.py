from pathlib import Path
import pandas as pd
from .config import CFG

def load_df():
    p_parq = Path(CFG.DATA_PARQUET)
    p_csv  = Path(CFG.DATA_CSV)

    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)

    raise FileNotFoundError(
        f"No data found. Put df_iv.parquet or df_iv.csv under ./data/. "
        f"(looked for {p_parq} and {p_csv})"
    )
