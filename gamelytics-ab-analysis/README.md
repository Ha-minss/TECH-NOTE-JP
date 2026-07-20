# Gamelytics A/B Analysis

モバイルゲームのプロモーションA/Bテストで、ARPU、課金率、ARPPU、売上集中度、ARPU差の不確実性を検証した再現可能な分析プロジェクトです。

## 1. Problem

B群の観測ARPUはA群より高い一方、課金率は低下しました。またA群の売上は少数の高額課金者に集中しており、単純平均だけではプロモーションを選べません。

> 現在のデータだけでBを全面展開できるか、それとも追加実験が必要か。

## 2. Data Audit

- A群ユーザー: 202,103
- B群ユーザー: 202,667
- 重複user_id: 0
- 欠損: 0
- 負のrevenue: 0
- Sample Ratio Mismatch: p-value 0.375352

`ab_test.csv` を主分析のSource of Truthとし、Retention用の `reg_data.csv` / `auth_data.csv` とはユーザー単位で結合しません。

## 3. Metrics

Primary Metricは全割当ユーザー基準のARPUです。

```text
ARPU = total revenue / assigned users
ARPU = conversion rate × ARPPU
```

ARPPUは課金者に条件付けた事後指標であるため、独立したTreatment Effectとして解釈しません。

## 4. Main Results

| Metric | A | B |
|---|---:|---:|
| Users | 202,103 | 202,667 |
| Payers | 1,928 | 1,805 |
| Conversion rate | 0.9540% | 0.8906% |
| Total revenue | 5,136,189 | 5,421,603 |
| ARPU | 25.4137 | 26.7513 |
| ARPPU | 2,664.00 | 3,003.66 |

B − A:
- ARPU差: +1.3376
- 観測ARPU Lift: +5.2632%
- 課金率差: −0.0633%p
- ARPPU差: +339.66

## 5. Statistical Tests

- 課金率 z-test p-value: 0.035029
- ARPU Bootstrap 95% CI: −2.8733 ～ +5.4648
- ARPU Permutation Test p-value: 0.5272
- ARPPU Bootstrap 95% CI: −69.43 ～ +732.98

「有意でない」は効果がないという意味ではなく、現在の標本では安定した優位性を確定できないという意味です。

## 6. Whale Sensitivity

Whaleを恣意的に削除せず、全ユーザーの原結果をPrimary Analysisとして維持しました。共通閾値のWinsorizationを感度分析としてのみ使用します。

A群は上位5%課金者が売上の69.76%、上位10%が89.90%を占め、平均の不確実性が大きいことを確認しました。

## 7. Decision

Bの観測ARPUは高いものの、95%信頼区間が0を含み、課金率は有意に低下しました。したがってBを即時全面展開せず、新規ユーザー標本で追加A/Bテストを行うことを推奨します。

## 8. Reproduce

```bash
python scripts/run_ab_analysis.py \
  --data-dir "<PATH_TO_DATA>" \
  --bootstrap-iterations 5000 \
  --permutation-iterations 5000 \
  --seed 42

python scripts/run_retention_analysis.py \
  --data-dir "<PATH_TO_DATA>"

python -m pytest tests -q
```

## 9. Generated Outputs

- `reports/data_audit.md`
- `reports/ab_analysis.md`
- `reports/retention_analysis.md`
- `reports/tables/*.csv`
- `reports/figures/*.png`
- `reports/figures/*.svg`

## 10. Cleanup Policy

日本語版では個人PCの絶対パスを `<PATH_TO_DATA>` に置換し、重複していたCSV出力を `reports/tables/` に統一します。

## 11. Limitations

- プロモーション費用、粗利、割引情報がないためGross Revenue基準です。
- ARPU差の信頼区間が0を含みます。
- Retention分析はA/B結論と統合しません。

## 12. What This Project Demonstrates

平均値の大小だけでリリース判断をせず、割当品質、指標分解、Heavy-tail、不確実性、感度分析を用いて保守的な意思決定ができることを示します。
