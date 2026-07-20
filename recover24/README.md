# Recover24

ボイスフィッシング・金融詐欺の被害者が説明した内容を事件情報へ構造化し、銀行や公的機関へ提出するための文書パッケージを生成するNLP/LLMシステムです。

## 1. Overview

被害直後の利用者は、送金時刻、口座、アプリインストール、本人確認情報の露出、連絡手段などを整理して提出する必要があります。本プロジェクトは自然言語の説明を事実・時系列・証拠へ変換し、提出可能性を確認します。

## 2. Problem

自由記述だけでは、重要情報の欠落や矛盾が残り、銀行・警察・通信会社が再確認する負担が増えます。一方、LLMが不足情報を推測して文書を完成させることは危険です。

## 3. Decision Question

> 被害者の負担を減らしつつ、推測を避け、提出先が確認可能な事実と証拠だけで文書を作成できるか。

## 4. Workflow

1. Incident Router：事件タイプと緊急度を分類
2. Dynamic Intake：事件に応じて追加質問
3. Case Fact Store：確認済み事実と未確認情報を分離
4. Timeline Builder：行動・送金・連絡を時系列化
5. Evidence Gate：証拠・同意・矛盾・必須項目を検証
6. Document Generator：提出先別の文書を生成
7. Human Confirmation：利用者が最終確認

## 5. Safety Design

- 未確認情報を事実として補完しない
- 緊急対応と文書作成を分離
- 個人情報・同意・提出先を明示
- 情報矛盾時は文書生成を停止
- 生成物に確認状態を付与

## 6. Evaluation

多様な詐欺パターンを含むGolden Setで、情報抽出、時系列、必須項目、危険な推測、文書完成条件を評価します。

## 7. Result

単なる相談チャットではなく、事件情報の構造化から提出文書までを一貫して扱うワークフローを構築しました。

## 8. Reproduce

`.env.example` を参考に環境変数を設定し、プロジェクト内の評価・デモコマンドを実行します。外部提出は自動化せず、ローカル生成物として確認します。

## 9. Repository Structure

```text
src/                    # Intake, fact store, planner, document generation
tests/                  # Golden cases and safety checks
data/                   # Synthetic evaluation data
templates/              # Submission document templates
```

## 10. Cleanup Policy

日本語版では誤って含まれていた `recover24/Causal_Inference_SECA/` を削除します。SECA分析は独立したトップレベルプロジェクトとして管理します。

## 11. Limitations

- 法律・金融機関の正式な判断を代替しません。
- 国・機関ごとに提出要件が異なるため、最新要件の確認が必要です。
- 実運用には暗号化、アクセス制御、保存期間管理が必要です。

## 12. What This Project Demonstrates

高ストレスな利用場面で、LLMの生成能力よりも事実管理、同意、証拠、提出可能性を優先した設計ができることを示します。
