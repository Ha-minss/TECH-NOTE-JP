# TECH-NOTE-JP

データを予測結果で終わらせず、**検証可能な意思決定や業務フローへ接続すること**を重視した、データサイエンス・機械学習・LLMシステムのポートフォリオです。

本リポジトリは、韓国語版 [`Ha-minss/TECH-NOTE`](https://github.com/Ha-minss/TECH-NOTE) を基に、日本企業の採用担当者・エンジニアが読みやすい構成へ整理した日本語版です。  
コード、モデル名、指標名、SQL、設定キー、テストデータの意味は原則として変更せず、READMEと公開説明を日本語向けに再構成しています。

## Featured Projects

| Project | Summary | Main Focus |
|---|---|---|
| [Korea SECA → Northern Kyushu SO₂](./causal-inference-seca) | 韓国の船舶燃料規制が北部九州のSO₂に与えた越境影響をDID・Event Studyで検証 | 政策評価、パネルデータ、頑健性検証 |
| [Financial Recall Agent](./financial-recall-agent) | 金融苦情1件から同一原因の影響顧客を探索し、証拠と監査ログを生成 | LLM Agent、RAG、Rule Engine、Human Review |
| [Mobile Game LTV Pipeline](./mobile-game-ltv-pipeline) | D0–D7行動から長期LTVを予測し、UA判断へ接続する再現可能なMLパイプライン | 時系列検証、XGBoost、Two-Stage Model |
| [AMEX Credit Risk Decisioning](./amex-credit-risk-decisioning) | 延滞リスクスコアを限られた審査リソースの優先順位へ変換 | Credit Risk、Top-K評価、コスト感度分析 |

# Projects

## 1. AIエージェント・LLMシステム

LLMを単なる回答生成器として扱うのではなく、業務上の根拠確認、ルール検証、ツール実行、監査ログ、人による承認と組み合わせています。

### [Financial Recall Agent](./financial-recall-agent)
金融苦情から同一原因の影響顧客を探索し、規約根拠・取引検証・金額計算・レビュー用レポートを分離した金融消費者保護システムです。

### [StoreOps Triage Agent](./storeops-triage-agent)
店舗の決済障害を構造化し、端末・POS・承認ログ・VAN/PSPなどの確認項目を読み取り専用ツールで収集する運用支援Agentです。

### [Recover24](./recover24)
ボイスフィッシング被害者の自然言語による説明を事件情報へ構造化し、銀行・警察等への提出資料を作成する文書支援システムです。

### [Provider Directory Control Tower](./provider-directory-control-tower)
医療提供者情報を公式ソースと照合し、自動修正可能なケースと人による確認が必要なケースを分離するデータ品質管理パイプラインです。

## 2. 機械学習・信用リスク

モデル精度だけでなく、データリーク防止、検証設計、Top-K運用、特徴量採用判断、業務コストとの接続を重視しています。

### [AMEX Credit Risk Decisioning](./amex-credit-risk-decisioning)
顧客単位の延滞リスクを推定し、限られた審査対象をどの範囲まで確認するかをPrecision・Capture Rate・Lift・コスト感度で評価します。

### [Xente Credit Feature Adoption](./xente-credit-feature-adoption)
返済履歴がない顧客において、取引行動が独立した信用リスクシグナルとして採用可能かを、顧客単位検証と特徴量除外実験で確認します。

## 3. ゲーム・プロダクト分析

予測値や平均差を提示するだけでなく、UA予算、高価値ユーザーの優先順位、プロモーション展開判断へ接続します。

### [Mobile Game LTV Pipeline](./mobile-game-ltv-pipeline)
大規模イベントログの粒度検証から特徴量生成、時系列検証、最終学習、モデルカード、事業分析までを一つの再現可能な流れにまとめています。

### [Gamelytics A/B Analysis](./gamelytics-ab-analysis)
ARPU上昇と課金率低下が同時に観測された実験について、Bootstrap・Permutation Test・Whale感度分析を行い、全面展開の可否を判断します。

## 4. 計量経済学・因果推論

単純な相関や前後比較ではなく、比較群、固定効果、事前トレンド、内生性、Placebo Test、頑健性検証を通じて解釈可能な範囲を明確にします。

### [Thailand Policy Revenue Persistence](./thailand-policy-revenue-persistence)
消費支援政策後の宿泊・飲食サービス売上が一時的な反応か、持続的な需要シグナルかをLag Model・Synthetic Comparator・Event Studyで検証します。

### [Korea SECA → Northern Kyushu SO₂](./causal-inference-seca)
韓国SECA Step 1導入後、北部九州の沿岸部と内陸部のSO₂差が縮小したかをDID・Event Study・各種頑健性検証で分析します。

### [Refugee Inflows → Crime Rates](./refugees-crime-panel)
難民流入と犯罪率の関係を国・年パネルで分析し、固定効果、国別トレンド、先行・遅行関係を追加した際の結果安定性を確認します。

### [Dominick's Price Elasticity IV-DML](./dominicks-price-elasticity-iv-dml)
小売スキャナーデータを用い、価格の内生性を考慮したFE-IV・Dynamic IV・DMLを比較し、価格反応推定の頑健性を検証します。

# Repository Policy

- 大容量または公開に適さない元データは含めません。
- 公開可能な派生表、再現コード、テスト、図表を中心に構成します。
- 数値はデータと識別仮定が許す範囲で限定的に解釈します。
- LLMシステムでは、自動判断よりも根拠、ルール、監査可能性、安全ゲート、人による承認を優先します。
- 日本語化によって評価データやGolden Setの意味が変わる箇所は、原文を維持します。

# Source

- Korean original: https://github.com/Ha-minss/TECH-NOTE
- Source commit: `dd92cb77c0ea6032515d09ce4244fab886a640b3`
