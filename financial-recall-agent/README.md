# Financial Recall Agent

金融苦情1件を起点に、同じ原因で影響を受けた顧客を探索し、規約根拠・取引検証・金額計算・監査ログをまとめたレビュー用パッケージを生成する金融消費者保護システムです。

## 1. Overview

LLMに補償可否や金額を直接判断させず、苦情理解、ポリシー検索、SQLによる影響範囲確認、再現可能なルール計算、人による承認を分離しています。

## 2. Problem

個別苦情への回答だけでは、同一障害で申告していない顧客を見落とします。一方、LLMが規約解釈・対象者抽出・金額計算を一度に行うと、再現性と監査可能性が不足します。

## 3. Decision Question

> 一件の苦情から、同一原因の潜在的な影響顧客を安全に特定し、担当者が根拠を確認できる状態まで自動化できるか。

## 4. Architecture

1. Complaint Router：苦情の意図と候補原因を構造化
2. Policy RAG：規約根拠と適用条件を検索
3. Rule Registry：承認済みルールとデータ契約を読み込み
4. SQL / Repository：対象取引と影響顧客を再現可能に抽出
5. Reconciliation：期待値と実績を照合
6. Evidence Package：根拠、計算、対象者、監査ログを生成
7. Human Review：最終補償判断は担当者が承認

## 5. Safety Design

- LLMによるSQL生成・補償判断・金額確定を禁止
- Rule Bundle、Data Contract、SQL Hashを検証
- 読み取り専用処理を基本とし、書き込み処理を分離
- 根拠不足時は断定せず、追加確認へルーティング
- 最終結果に出典と監査ログを付与

## 6. Evaluation

Golden Caseと合成取引データを用いて、状態分類、原因分類、必要ツール呼び出し、禁止行為、安全な棄権、根拠引用を評価します。

## 7. Result

個別回答を生成するチャットボットではなく、苦情から影響範囲を探索し、人が確認できる証拠パッケージまで作る業務フローとして実装しました。

## 8. Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q
python -m recall_agent.interfaces.cli.h07_reward_missing_demo
```

## 9. Repository Structure

```text
src/recall_agent/       # Application, core rules, RAG, templates
data/demo/              # Synthetic datasets and approved rule assets
sql/reward_missing/     # Reproducible SQL stages
tests/                  # Architecture, rules, RAG, safety tests
```

## 10. Limitations

- 合成データを用いたポートフォリオ実装であり、銀行本番環境ではありません。
- 実運用には権限管理、個人情報保護、変更承認、監視、障害復旧が必要です。
- 他の商品・苦情タイプへの拡張には個別のRule Bundleと検証データが必要です。

## 11. Key Technologies

Python, LLM, RAG, DuckDB/SQL, Rule Engine, Data Contract, Audit Log, Human-in-the-Loop

## 12. What This Project Demonstrates

生成AIの能力を見せるだけでなく、金融業務で必要な再現性・根拠・安全性・人による承認をシステム境界として設計できることを示します。
