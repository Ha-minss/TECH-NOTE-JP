import pandas as pd
from pathlib import Path  # <--- 요 부분이 중요합니다!


current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "tourism.parquet"

def simple_search(query:  str, top_k: int = 3):
    df = pd.read_parquet(DATA_PATH)
    results = df[
        df["name"].str.contains(query, na =False) |
        df["explanation"].str.contains(query, na=False)
    ]

    return results.head(top_k)

if __name__ == "__main__":
    search_term = "北海道"
    print(search_term)
    print(simple_search(search_term))