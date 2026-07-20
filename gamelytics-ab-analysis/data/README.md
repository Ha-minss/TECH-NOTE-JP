# Data

元のCSVファイルは本リポジトリに含めません。実行時に外部データディレクトリを指定してください。

```bash
python scripts/run_ab_analysis.py --data-dir "<PATH_TO_DATA>" --bootstrap-iterations 5000 --seed 42
python scripts/run_retention_analysis.py --data-dir "<PATH_TO_DATA>"
```

CSVの区切り文字はセミコロン（`;`）です。
