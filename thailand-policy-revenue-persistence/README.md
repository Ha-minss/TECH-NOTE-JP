# Thailand Policy Revenue Persistence

タイの消費支援政策後、宿泊・飲食サービス業の売上増加が一時的な政策反応か、政策終了後にも残る持続的な需要シグナルかを分析した計量経済プロジェクトです。

## 1. Overview

政策後の売上上昇を直ちに返済能力や恒常需要として解釈せず、比較業種との相対変化、反応時点、持続期間を検証します。

## 2. Problem

政策支出は短期的に売上を押し上げますが、その効果が終了後も続くとは限りません。単純な前後比較は季節性や景気変動を政策効果と混同します。

## 3. Decision Question

> 政策対象業種の売上反応はいつ発生し、何か月持続し、政策後の信用判断に利用できるほど安定しているか。

## 4. Methods

- Lag Model
- Synthetic Comparator
- Event Study
- 累積反応
- Placebo / Robustness Check
- 対象業種と比較業種の相対トレンド

## 5. Result

政策後0–3か月の累積相対反応で正のシグナルを確認しました。ただし政策支出による一時的効果と持続需要を完全に分離したとは解釈しません。

## 6. Interpretation

売上上昇は信用力の直接的証拠ではなく、追加確認に使う補助シグナルです。政策露出、業種構成、季節性、政策終了後の減衰を合わせて判断する必要があります。

## 7. Reproduce

プロジェクト内のNotebookまたはScriptを順に実行し、派生表と図を再生成します。元データの公開条件に従い、大容量ファイルは除外しています。

## 8. Repository Structure

```text
notebooks/              # Data checks, models, event study
outputs/                # Derived tables and figures
data/README.md          # Data source and exclusions
```

## 9. Robustness

比較群変更、期間変更、Lag仕様、Placebo時点、累積反応の定義を変え、結論が特定仕様だけに依存しないか確認します。

## 10. Limitations

- 政策参加の自己選択を完全には除去できません。
- マクロショックや観光回復と政策効果が重なる可能性があります。
- 企業単位の返済能力を直接測定していません。

## 11. Key Technologies

Python, Panel Data, Lag Model, Synthetic Comparator, Event Study, Robustness Analysis

## 12. What This Project Demonstrates

政策後の増加を成功と断定せず、反応のタイミング・持続性・比較群・代替説明を検証して意思決定可能な範囲を示せることを証明します。
