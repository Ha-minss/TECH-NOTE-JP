# Provider Directory Control Tower

医療提供者ディレクトリの名称、住所、電話番号、診療状態などを公式ソースと照合し、修正候補とレビュー優先順位を生成するデータ品質管理プロジェクトです。

## 1. Overview

公開情報や外部ソースをそのまま更新に使わず、正規化、証拠照合、ソース信頼度、競合検出、Confidence Score、人による確認を組み合わせます。

## 2. Problem

医療提供者情報の小さな誤りでも、検索失敗、予約ミス、問い合わせ増加、利用者の信頼低下につながります。ただし複数ソースが食い違う場合に自動修正すると新たな誤りを作ります。

## 3. Decision Question

> どのレコードを自動修正でき、どのレコードを人による確認へ送るべきか。

## 4. Pipeline

1. Provider Recordの読み込み
2. 名称・住所・電話番号の正規化
3. NPI/CMS等の公式情報との照合
4. フィールド単位の一致・競合判定
5. Confidence ScoreとEvidence Packet生成
6. Auto-accept / Review / Rejectへルーティング
7. Audit Logと提出ファイル生成

## 5. Reliability Design

- 公式ソースを優先
- フィールド単位で証拠を保持
- 一致率だけでなく競合を明示
- 低信頼・複数候補は人へ送る
- 更新理由を監査可能な形式で保存

## 6. Outputs

`evidence_packets.jsonl`、`recommendations.jsonl`、`submission.csv`、`connector_diagnostics`、`audit_log.jsonl`、`executive_summary.md` を生成します。

## 7. Result

検索結果の取得ではなく、「更新可能な証拠に変換して業務キューへ送る」運用パイプラインとして設計しました。

## 8. Reproduce

```bash
pip install -r requirements.txt
python run_pipeline.py
python scripts/evaluate_pipeline.py
```

## 9. Repository Structure

```text
configs/                # Pipeline configuration
data/input/             # Sample provider records
src/                    # Sources, normalization, confidence, decision
sample_outputs/         # Reproducible example outputs
```

## 10. Limitations

- 公開・サンプルデータを使用したポートフォリオ実装です。
- ソース更新頻度、利用規約、API制限を本番設計で管理する必要があります。
- Confidence Scoreは業務承認基準と継続的に調整する必要があります。

## 11. Key Technologies

Python, Data Quality, Entity Matching, Evidence Fusion, Confidence Scoring, Review Routing

## 12. What This Project Demonstrates

複数ソースの検索結果を、証拠・信頼度・競合・レビュー優先順位を持つ運用可能なデータ品質ワークフローへ変換できることを示します。
