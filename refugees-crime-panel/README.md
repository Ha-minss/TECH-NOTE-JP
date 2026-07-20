# Refugee Inflows → Crime Rates

難民流入と犯罪率の関係を国・年パネルで分析し、単一回帰の相関が固定効果、国別トレンド、先行・遅行関係を追加しても維持されるかを検証した社会データ研究です。

## 1. Overview

社会的に敏感なテーマであるため、強い結論を先に置かず、結果がどの仕様で現れ、どの統制で弱まるかを中心に報告します。

## 2. Problem

難民流入と犯罪率は景気、紛争、人口構成、制度、報告率など多くの要因と同時に変化します。単純相関はこれらを難民効果として誤認する可能性があります。

## 3. Decision Question

> 国固有要因、年共通ショック、国別トレンド、時間順序を考慮した後も、難民流入と犯罪率に一貫した関係が残るか。

## 4. Methods

- Country Fixed Effects
- Year Fixed Effects
- Country-specific Trend
- Lead / Lag Check
- Alternative Crime Outcomes
- Robust Standard Errors
- IV specification where applicable

## 5. Result

一部仕様では正の相関が見られましたが、追加トレンドや動学確認を入れると弱まり、複数犯罪指標にわたる一貫した強い正の関係は確認できませんでした。

## 6. Interpretation

「関係がない」と証明したのではなく、利用可能なデータと識別設計では強い一般化を支持できないという結論です。

## 7. Research Integrity

セミナー終了後も追加検証を続け、当初の結果が頑健でないことを確認しました。期待した結論よりも検証結果を優先しています。

## 8. Reproduce

プロジェクト内のNotebookを順に実行し、モデル仕様、動学チェック、頑健性表を再生成します。

## 9. Repository Structure

```text
notebooks/              # Panel construction and estimations
outputs/                # Tables and figures
data/README.md          # Sources and exclusions
```

## 10. Robustness

サンプル、犯罪指標、流入尺度、固定効果、トレンド、Lead/Lag、標準誤差仕様を変更して確認します。

## 11. Limitations

- 国際比較データの定義・報告率は国ごとに異なります。
- 難民受入政策の内生性を完全には解消できません。
- 国レベル結果を個人行動へ一般化できません。

## 12. What This Project Demonstrates

センシティブな社会課題で、望ましい物語に合わせず、追加検証により当初結論を修正し、限定的に報告できる研究姿勢を示します。
