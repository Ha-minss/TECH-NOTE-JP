# StoreOps Triage Agent

店舗から届く決済障害の問い合わせを構造化し、確認すべき証拠と次の対応を整理するLLMベースの運用支援Agentです。

## 1. Overview

「決済できない」という同じ症状でも、端末設定、TID、POS接続、開通履歴、承認ログ、VAN/PSP、カード会社応答など原因候補は異なります。本プロジェクトは、LLMの推測ではなく読み取り専用ツールの証拠を中心に原因を絞り込みます。

## 2. Problem

電話・チャット担当者が複数システムを順番に確認すると、確認漏れ、誤案内、不要なエスカレーションが発生します。LLMが原因を即断すると、証拠のない誤判定につながります。

## 3. Decision Question

> 問い合わせ内容から必要な確認項目を選び、根拠が揃った範囲だけで安全に一次切り分けできるか。

## 4. Workflow

1. 症状と店舗情報を構造化
2. 原因候補に応じた確認チェックリストを生成
3. 端末・店舗・承認・接続情報を読み取り専用ツールで照会
4. 証拠の整合性と不足項目を検証
5. 原因候補、根拠、次の対応、エスカレーション先を提示
6. 運用担当者が確認して案内

## 5. Safety Rules

- ツール結果なしで原因を確定しない
- 書き込み・設定変更・返金などの操作を自動実行しない
- 証拠が矛盾する場合は人による確認へ送る
- 必須ツール未実行時は最終回答を制限する
- 監査可能なTraceを残す

## 6. Evaluation

50件の評価ケースで、状態・原因・必須ツール・禁止行為・根拠引用・棄権安全性を確認します。失敗ケースは、単純な正解率だけでなく「必要な証拠を取得したか」という観点で分析します。

## 7. Result

LLMの自由回答を中心にせず、Planner、Tool Layer、Evidence Normalization、Safety Gate、Human Reviewを分離した運用支援構造を実装しました。

## 8. Reproduce

プロジェクト内のREADME、設定、テスト手順に従って評価ケースを実行します。外部システムの代わりに合成データと読み取り専用アダプターを使用します。

## 9. Repository Structure

```text
src/                    # Agent orchestration and tool adapters
tests/                  # Evaluation and safety tests
experiments/            # Current evaluation artifacts
data/                   # Synthetic or public test data
```

## 10. Cleanup Policy

日本語版では、公開ポートフォリオとして不要な `storeops-triage-agent_old/` と過去のlegacy実装・旧評価レポートを除外し、現在の50ケース評価版だけを残します。

## 11. Limitations

- 実店舗・決済事業者の本番システムに接続したものではありません。
- 実運用では認証、権限、SLA、個人情報、障害対応手順が必要です。
- 原因候補の網羅性は利用可能なツールとデータに依存します。

## 12. What This Project Demonstrates

LLM Agentを「原因を当てるモデル」ではなく、「必要な証拠を漏れなく集め、安全な一次対応を支援する業務システム」として設計できることを示します。
