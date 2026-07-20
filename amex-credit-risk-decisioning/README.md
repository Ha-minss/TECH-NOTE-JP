# AMEX Credit Risk Decisioning

American Express Default Prediction公開データを用い、顧客の延滞リスクを予測し、限られた審査リソースの優先順位へ接続した信用リスク機械学習プロジェクトです。

## 1. Overview

目的は全顧客を二値分類することではなく、リスクスコアを用いて「誰を先に確認するか」「どのTop-Kまで確認するか」を設計することです。

## 2. Problem

匿名化された多数の時系列特徴量があり、単純な集計では顧客行動の変化を十分に表現できません。また、モデルスコアを絶対的なデフォルト確率として扱うと、運用上の誤解が生じます。

## 3. Decision Question

> 月次の審査リソースが限られる状況で、どの顧客群を優先的に確認すればリスク捕捉効率を高められるか。

## 4. Data and Features

- 顧客単位で時系列レコードを集計
- 水準、変動、直近値、傾向、欠損、カテゴリ情報を特徴化
- 顧客単位の分割でリークを防止
- 公開リポジトリには元データ・顧客ID・学習済みモデルを含めない

## 5. Modeling

LightGBM、XGBoost、CatBoost、Tabular MLPを比較し、OOF予測とブレンドを用いて安定性を確認します。

## 6. Evaluation

- AMEX competition metric
- ROC-AUC / PR-AUC
- Top-K Precision
- Capture Rate
- Lift
- Risk Decile
- 審査コストと見逃しコストを置いた感度分析

## 7. Operational Interpretation

スコアは絶対的なデフォルト確率ではなく、審査優先順位を決めるRanking Signalとして扱います。最適な閾値は固定値ではなく、業務コストと審査能力に依存します。

## 8. Result

上位リスク区間で高いPrecisionとLiftを確認し、Top-K別の捕捉率とコスト仮定を比較できる意思決定表を作成しました。

## 9. Reproduce

```bash
pip install -r requirements.txt
python -m compileall src tests
python -m pytest -q
```

公開サンプルでは合成スコアを使用し、非公開の元データや学習済みArtifactを要求しないSmoke Testを提供します。

## 10. Repository Structure

```text
configs/                # Model and policy scenarios
src/amex_risk/          # Features, CV, metrics, Top-K policy
outputs/tables/         # Verified derived summaries
docs/                   # Provenance, governance, reproduction
tests/                  # Leakage, metrics, repository integrity
```

## 11. Limitations

- 個人プロジェクトであり、カード会社の本番審査システムではありません。
- コスト値は業務仮定であり、実データでの検証が必要です。
- 分布変化、Calibration、公平性、説明責任の継続監視が必要です。

## 12. What This Project Demonstrates

高精度モデルの構築だけでなく、検証、ランキング評価、コスト感度、ガバナンスを通じてモデル出力を業務判断へ変換できることを示します。
