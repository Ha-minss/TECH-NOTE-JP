import pandas as pd
from pathlib import Path

# ==========================================
# settings
# ==========================================
BASE_DIR = Path("city_rag_hub/data")
INPUT_FILE = BASE_DIR / "raw" / "tourism.xlsx.xlsm"
OUTPUT_FILE = BASE_DIR / "processed" / "tourism.parquet"

RENAME_MAP = {
    '通し番号': 'id',
    '都道府県': 'prefecture',
    '日本語タイトル': 'name',
    "日本語本文": "explanation"
}

FIX_PREFECTURE_MAP = {
    '鹿児島': '鹿児島県',
    '和歌山': '和歌山県'
}

# ==========================================
# raw data cleaning
# ==========================================
# tokyo・osaka and osaka・tokyo were mixed in raw data
def normalize_prefecture_name(text: str) -> str:
    if isinstance(text, str) and "・" in text:
        parts = text.split("・")    # tokyo・osaka >> [tokyo, osaka]
        parts.sort()
        return "・".join(parts)
    return text

#cleaning sapaces and rename
def clean_tourism_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[list(RENAME_MAP.keys())].rename(columns=RENAME_MAP)
    df['prefecture'] = (
        df['prefecture']
        .str.strip()
        .str.replace('\n', '', regex=False)
        .str.replace(' ', '', regex=False)
        .replace(FIX_PREFECTURE_MAP)
        .apply(normalize_prefecture_name)
    )

#delete duplicated columns
    df = df.sort_values(
        by="explanation", 
        key=lambda x: x.str.len(), 
        ascending=False
    )
    df = df.drop_duplicates(subset=["name"])

    return df

# ==========================================
#  I/O (Execution)
# ==========================================
def main():
    # 1. path check
    if not INPUT_FILE.exists():
        print(f"❌ [Error] we can't find that file : {INPUT_FILE}")
        return

    # 2. making folder
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {INPUT_FILE}")

    try:
        # data load
        df_raw = pd.read_excel(INPUT_FILE, sheet_name='Sheet1', engine='openpyxl')
        
        # data cleaning
        df_clean = clean_tourism_data(df_raw)
        
        print(f"📊 Summary: original {len(df_raw)}columns -> after cleaning {len(df_clean)}columns")

        # save
        df_clean.to_parquet(OUTPUT_FILE, index=False)
        print(f"✅ Success! Saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ Processing Failed: {e}")

if __name__ == "__main__":
    main()