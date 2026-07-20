# Dominick's Price Elasticity IV-DML

Dominick's小売スキャナーデータを用い、価格変化が販売量に与える影響を、固定効果、操作変数、分散ラグ、DMLで比較した価格反応分析です。

## 1. Overview

価格と販売量の単純回帰は、需要ショック、販促、店舗特性、商品特性による内生性を含みます。本プロジェクトは識別戦略によって推定値がどの程度変わるかを検証します。

## 2. Problem

需要が高い時に価格が上がる、在庫・販促に合わせて価格が設定されるなど、価格は外生的ではありません。そのため観測価格の係数を因果効果として解釈できません。

## 3. Decision Question

> コスト由来の価格変動と固定効果を用いた場合、商品カテゴリ別の価格反応はどの程度安定して推定できるか。

## 4. Data

Cereal、Canned Soup、Bottled Juices、Cookiesの店舗・商品・週スキャナーパネルと、公開可能な派生結果表を使用します。

## 5. Methods

- Product-Store Fixed Effects
- Week Fixed Effects
- FE-IV / 2SLS
- First-stage diagnostics
- Dynamic / Distributed-lag IV
- FD-PLIV-DML
- Category-level comparison

## 6. Validation

操作変数の強さ、符号、カテゴリ間差、現在価格とラグ価格、FE-IVとDMLの差を比較します。単一の弾力性値ではなく推定法への感度を報告します。

## 7. Result

多くの仕様でOwn-price Sales Responseは負でしたが、大きさは統制、Lag構造、推定法に敏感でした。そのため一つの数字を確定値として提示せず、頑健な方向と不確実な幅を分けて解釈します。

## 8. Reproduce

```bash
pip install -r requirements.txt
cd notebooks
python -m nbconvert --to notebook --execute 01_results_overview.ipynb --inplace
python -m nbconvert --to notebook --execute 02_fe_iv_baseline_results.ipynb --inplace
python -m nbconvert --to notebook --execute 03_dynamic_iv_robustness.ipynb --inplace
python -m nbconvert --to notebook --execute 04_dml_validation_interpretation.ipynb --inplace
```

## 9. Repository Structure

```text
notebooks/                      # Result overview and model comparisons
outputs/tables/model_results/   # Derived public result tables
outputs/figures/                # Reproducible figures
data/README.md                  # Raw data exclusion note
```

## 10. Business Interpretation

価格反応推定は価格設定の参考になりますが、カテゴリ平均を個別商品へ直接適用できません。利益率、競合、在庫、販促、長期需要を追加する必要があります。

## 11. Limitations

- 元パネルではなく派生結果表を公開しています。
- 操作変数の排除制約は直接検定できません。
- DMLもデータと仮定に依存し、内生性を自動的に解決するものではありません。

## 12. What This Project Demonstrates

機械学習の柔軟性と計量経済学の識別を比較し、推定値の大きさより「なぜその数字を信頼できるか」を説明できることを示します。
