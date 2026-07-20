# Mobile Game LTV Production-Style ML Pipeline

モバイルゲームのD0–D7イベントログからD8–D180のLTVを予測し、UA予算と高価値ユーザーの優先順位へ接続するProduction-style MLパイプラインです。

## 1. Overview

2,100万件規模のイベントログを、検証可能なモデリング粒度へ変換し、特徴量生成、時系列検証、モデル比較、最終学習、予測Artifact、モデルカード、事業分析まで一貫して実行します。

## 2. Problem

イベント行をそのまま学習するとユーザー粒度が崩れ、同一ユーザーの情報が学習・検証へ跨る危険があります。また平均LTV精度だけでは、UA判断や高価値ユーザー選定に十分ではありません。

## 3. Decision Question

> 初期7日間の行動から長期価値をどこまで安定して予測し、獲得・運用判断に利用できるか。

## 4. Pipeline

1. Raw Data Validation
2. Modeling Grain Diagnostics
3. Missing / Input Quality Checks
4. Feature Generation
5. Rolling Time Validation
6. Baseline / Linear / XGBoost / Two-Stage比較
7. Optuna Tuning
8. Final Refit and Prediction
9. Model Card and Business Analysis

## 5. Validation

ランダム分割ではなく時系列Rolling Foldを用い、将来情報の混入を防ぎます。RMSLEだけでなく、Top-K Capture、Fold間分散、セグメント別性能を確認します。

## 6. Modeling

単一回帰モデルと、課金有無を分類した後に正のLTVを回帰するTwo-Stage Modelを比較します。最終モデルは平均精度だけでなく安定性と事業利用性で選択します。

## 7. Result

Single Target EncodingモデルとTwo-Stageモデルの性能・安定性を比較し、LTV予測をUA予算と高価値ユーザー優先順位へ解釈するレポートを作成しました。

## 8. Reproduce

```bash
pip install -r requirements.txt
make test
make all
```

大規模元データは含めず、必要なパスを設定して実行します。

## 9. Repository Structure

```text
src/pipeline/           # Validation, features, training, prediction
src/experiments/        # Baselines, rolling validation, tuning
src/common/             # IO, metrics, preprocessing
reports/                # Diagnostics, experiments, model card
tests/                  # Pipeline and metric tests
```

## 10. Business Use

- UAチャネル別の期待価値比較
- 高価値ユーザー候補の優先順位
- モデル誤差と予算リスクの可視化
- 再学習・監視に必要な指標定義

## 11. Limitations

- 観測期間外のゲーム変更やマーケティング施策で分布が変わる可能性があります。
- 予測値は因果的な施策効果ではありません。
- 実運用では獲得単価、粗利、チャネル制約を追加する必要があります。

## 12. What This Project Demonstrates

大規模ログを安全な粒度へ変換し、リークを防いだ検証と再現可能なパイプラインを通じて、モデルを事業判断へ接続できることを示します。
