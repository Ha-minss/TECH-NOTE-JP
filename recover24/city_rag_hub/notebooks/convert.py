import pandas as pd
import os

#file path setting
input_path = "city_rag_hub/data/raw/japan_tourist_data_v1.xlsx"
output_path = "city_rag_hub/data/raw/japan_tourist_data_v1.parquet"

#cheking data
df = pd.read_excel(input_path)
print(f"data checking {len(df)} columns are cheked")

#converting to parquet
df.to_parquet(output_path, index= False)

print("finished converting")
