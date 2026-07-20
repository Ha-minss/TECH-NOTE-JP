# Korea SECA → Northern Kyushu SO₂

韓国の船舶排出規制海域（SECA）Step 1導入後、北部九州の沿岸部と内陸部のSO₂差が縮小したかを分析した越境環境政策プロジェクトです。

## 1. Overview

船舶燃料硫黄規制の効果は規制海域内だけでなく、風向・距離・航路を通じて周辺地域へ波及する可能性があります。本研究は日本の観測所月次パネルを用いて越境影響を検証します。

## 2. Policy Timeline

- IMO 2020: 2020年1月
- Korea ECA Step 1: 2020年9月（停泊時0.1%S）
- Full Implementation: 2022年1月（航行時0.1%S）

## 3. Decision Question

> 韓国SECA導入後、北部九州の沿岸観測所におけるSO₂が内陸観測所と比べて相対的に低下したか。

## 4. Data

2017年1月から2023年12月までの観測所・月パネルを構成し、SO₂、気象、沿岸・内陸区分、政策時点、風向ベースの曝露指標を使用します。

## 5. Identification

- Difference-in-Differences
- Station / Year-Month Fixed Effects
- Event Study
- 気象統制
- 事前トレンド確認
- Placebo Test
- Treatment Definition変更
- OLS / PPMLによるLevel SO₂確認

## 6. Main Result

主要仕様では、Step 1後に北部九州沿岸部のSO₂が内陸部に対して相対的に低下する方向を確認しました。ただし韓国全体の結果は弱く、仕様・曝露定義による不確実性を明示します。

## 7. Interpretation

結果は「韓国規制が九州全域の大気質を改善した」と断定するものではありません。政策時点と沿岸曝露に整合的な相対変化を示す限定的な証拠として解釈します。

## 8. Reproduce

```bash
pip install -r requirements.txt
python scripts/00_prepare_df_iv_for_github.py
python scripts/01_main_did_step1.py
python scripts/02_event_study_step1.py
python scripts/03_robustness_suite_step1.py
```

## 9. Repository Structure

```text
scripts/                # Data prep, DID, event study, robustness
step1_did/              # Reusable estimation modules
data/README.md          # Data source and public exclusions
notebooks_run_in_colab_step1.ipynb
```

## 10. Robustness

Treatment定義、沿岸範囲、期間、Level/Log仕様、標準誤差、気象統制、Placebo時点を変更して結果安定性を確認します。

## 11. Limitations

- 同時期の排出・産業・交通変化を完全には除去できません。
- 観測所配置と風向指標には測定誤差があります。
- 越境輸送の機構を直接観測したものではありません。

## 12. What This Project Demonstrates

日本企業にも関連する九州データを用い、政策制度、空間曝露、パネル識別、事前トレンド、頑健性を一つの研究設計として統合できることを示します。
