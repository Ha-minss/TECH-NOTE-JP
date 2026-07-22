# StoreOps Triage Agent

店舗における決済障害の問い合わせを受け付け、**運用担当者が確認できる原因候補、照会根拠、次に確認すべき項目、安全な案内文**として整理する、オフライン決済運用向けのトリアージエージェントです。

本プロジェクトの中心は、「LLMが決済障害の原因を当てること」ではありません。加盟店オーナーからの短い問い合わせを構造化し、SOP/RAGから確認すべき項目を検索し、読み取り専用の運用ツールで根拠を照会したうえで、十分な根拠が得られた場合にのみ原因候補を提示する、**evidence-firstの運用ワークフロー**です。

---

## 1. Overview

オフライン店舗で決済が失敗すると、現場の業務は直ちに停止します。

店舗側からは、「決済できません」「端末で承認エラーが繰り返し表示されます」「POSから端末へ金額が送信されません」といった短い問い合わせが寄せられます。しかし、運用担当者が確認すべき原因は一つではありません。

同じ「決済失敗」でも、実際の原因は次のように分かれます。

| 原因候補 | 簡単な説明 |
|---|---|
| `duplicate_tid` | 新しい端末の設置後、既存端末と決済識別子が重複している |
| `terminal_identifier_mismatch` | 現場端末の番号・シリアルと登録システム上の値が一致しない |
| `van_merchant_registration_missing` | VANまたは加盟店番号の登録が完了していない |
| `pos_front_connection_issue` | POSの決済要求が決済端末・Frontへ伝達されていない |
| `clarification_required` | 問い合わせ内容が曖昧で、現場から追加情報を得る必要がある |
| `tool_failure` | 必須の運用データ照会に失敗し、原因を確定できない |
| `temporal_conflict` | 障害発生時点の記録と現在の記録が異なり、時間軸での確認が必要 |

StoreOps Triage Agentでは、この問題を次のように定義しました。

> 加盟店オーナーからの決済障害問い合わせを、  
> 運用担当者が確認できるevidence-backed case briefへ変換する。

全体の処理フローは次のとおりです。

```text
加盟店からの問い合わせ
→ 問い合わせ種別の解析
→ SOP/RAG検索
→ 確認すべきdata_needの選定
→ read-only toolの実行計画
→ SQLite fixtureから運用根拠を照会
→ evidence recordの生成
→ 原因候補の判断
→ safety gateの適用
→ 運用担当者向けcase briefの生成
→ 評価結果・traceの保存
```

本システムは、決済実行、返金、承認取消、TID変更、VAN設定変更などの機微な操作を行いません。

実行できるのは次の処理です。

- 問い合わせを構造化する。
- どの運用データを確認すべきか計画する。
- 読み取り専用ツールで根拠を照会する。
- 根拠が存在する場合に原因候補を提示する。
- 根拠が不足している、または互いに矛盾する場合は判断を保留する。
- 運用担当者に次の確認項目と安全な案内文を提示する。

したがって、本プロジェクトの結論は次のとおりです。

> 決済障害エージェントは「原因を当てるチャットボット」ではなく、  
> 現場からの問い合わせを、運用根拠と安全ゲートを備えた検証可能なケースへ変換するトリアージワークフローであるべきです。

---

## 2. Problem & Objective

店舗の決済障害は、一般的な顧客問い合わせよりも運用上のリスクが大きい問題です。

誤った案内を行うと、店舗側が決済を繰り返し再試行したり、顧客対応中に決済業務が停止したりする可能性があります。また、運用チームがVAN、端末設置会社、POS、社内設定担当者の間で原因を切り分けるまでに時間を要することもあります。

問題は、問い合わせ文だけでは原因を確定しにくい点です。

例えば、次の二つの問い合わせは似ていますが、最初に確認すべきデータは異なります。

| 問い合わせ | 最初に確認すべきデータ |
|---|---|
| 「新しい端末を設置してから、既存端末でカード承認に失敗します。」 | 端末一覧、TID設定、開通履歴、承認失敗ログ |
| 「POSで決済を押しても、端末へ金額が送信されません。」 | POS-Front接続ログ、リクエスト伝達失敗、ペアリング状態 |

したがって、本プロジェクトの目的は単純な回答生成ではありません。

目的は次の四つです。

第一に、加盟店からの問い合わせを、決済承認障害、POS-Front連携問題、追加確認が必要な曖昧な問い合わせのいずれかに分類します。

第二に、SOP/RAGとtool catalogを基に、確認すべきdata_needとread-only toolを計画します。

第三に、照会した運用データをevidence recordへ変換し、十分な根拠がある場合にのみ原因候補を提示します。

第四に、必須データの照会失敗、根拠不足、時間軸上の矛盾、機微な操作要求は、安全ゲートを通じて`NEEDS_CLARIFICATION`、`DEGRADED_REVIEW`、`CONFLICT_REVIEW`へ振り分けます。

本プロジェクトが防止しようとするリスクは次のとおりです。

| リスク | 防止方法 |
|---|---|
| 根拠のない原因断定 | supporting evidenceがない場合は`likely`判定を禁止 |
| 決済・返金・設定変更 | forbidden actionsとして遮断 |
| 必須ツール失敗後も原因を判断 | `DEGRADED_REVIEW`へ移行 |
| 現場説明とシステム記録の矛盾 | `CONFLICT_REVIEW`へ移行 |
| 曖昧な問い合わせへの無理な診断 | `NEEDS_CLARIFICATION`へ移行 |
| LLMによるtoolやdata_needの捏造 | prompt contractとallowed catalogで制限 |

---

## 3. Data

本プロジェクトでは実際の運用データを使用せず、オフライン決済障害を模した50件の合成運用ケースとSQLite fixtureを使用しました。

重要な点は、原因の正解をraw operational table内に含めていないことです。運用ツールが照会するデータベースには運用上の事実のみを格納し、正解ラベルはgolden setと評価ファイルにのみ保持しています。

つまり、エージェントは正解を直接読むのではなく、運用factを照会・組み合わせて原因を判断する必要があります。

Canonical assetsは次のとおりです。

```text
data/fixtures/offline_payment_ops_synthetic_50.sqlite3
data/fixtures/offline_payment_ops_synthetic_50_manifest.json
data/golden/offline_payment_ops_cases_50.json
data/evaluation/retrieval_cases_50.json
data/evaluation/planner_cases_50.json
data/policies/offline_payment_ops/
data/tool_catalog/offline_payment_ops_tools.json
reports/synthetic_50_validation_report.md
reports/synthetic_50_validation_matrix.csv
```

50件の合成ケースの分布は次のとおりです。

| Family | Case Type | 件数 | 期待ステータス |
|---|---|---:|---|
| S1 | `duplicate_tid` | 10 | `READY_FOR_REVIEW` |
| S2 | `terminal_identifier_mismatch` | 7 | `READY_FOR_REVIEW` |
| S3 | `van_merchant_registration_missing` | 7 | `READY_FOR_REVIEW` |
| S4 | `pos_front_connection_issue` | 7 | `READY_FOR_REVIEW` |
| S5 | `clarification_required` | 7 | `NEEDS_CLARIFICATION` |
| S6A | `required_tool_failure` | 4 | `DEGRADED_REVIEW` |
| S6B | `optional_tool_failure` | 3 | `READY_FOR_REVIEW` |
| S7 | `temporal_conflict` | 5 | `CONFLICT_REVIEW` |

合成fixtureのrow-level validationでは50件中50件が通過しました。

| 検証項目 | 結果 |
|---|---:|
| Total cases | 50 |
| Passed | 50 |
| Failed | 0 |

主要テーブルのrow countは次のとおりです。

| Table | Row count |
|---|---:|
| `stores` | 50 |
| `store_operator_access` | 50 |
| `terminals` | 86 |
| `tid_assignments` | 91 |
| `activation_events` | 22 |
| `approval_events` | 53 |
| `support_routes` | 38 |
| `terminal_identities` | 86 |
| `installation_events` | 12 |
| `van_registrations` | 14 |
| `pos_front_links` | 7 |
| `pos_front_connection_events` | 7 |
| `tool_failure_injections` | 7 |
| `scenarios` | 50 |
| `scenario_stores` | 50 |

SOP/RAG文書は5件です。

| Policy ID | 役割 |
|---|---|
| `SOP-PAY-OP-001` | 決済承認エラーの初動対応 |
| `SOP-PAY-OP-002` | 新規端末の設置および識別情報の検証 |
| `SOP-PAY-OP-003` | 加盟店・VAN登録状態の確認 |
| `SOP-PAY-OP-004` | POS-Front連携および通信障害の確認 |
| `SOP-PAY-OP-005` | 不確実性・人による確認・安全上の注意事項 |

---
## 4. Method / System Design

StoreOps Triage Agentの設計原則は明確です。

> 問い合わせ文だけで原因を断定しない。  
> 運用データを読み取り専用で照会し、根拠がある場合にのみ原因候補を提示する。

全体構成は次のとおりです。

```text
Merchant Message
   ↓
Case Parser
   ↓
Policy Retrieval
   ↓
Planner
   ↓
Read-only Tool Gateway
   ↓
Evidence Builder
   ↓
Reasoner
   ↓
Safety Gate
   ↓
Case Brief
   ↓
Evaluation / Trace
```

### 4.1 Case Parser

加盟店からの問い合わせを読み取り、決済障害の種類と不足している現場情報を把握します。

Parserの役割は次のとおりです。

| 処理 | 例 |
|---|---|
| 問い合わせ種別の分類 | 決済承認失敗、POS-Front接続問題 |
| merchant-observable missing fieldの抽出 | エラー発生時刻、端末設置場所、エラーメッセージなど |
| 内部DB上の事実を推測しない | TID重複やVAN登録状態を問い合わせ文だけで断定しない |

Parserは最終的な原因を判断しません。

### 4.2 Policy Retrieval

問い合わせ種別に関連するSOP文書を検索します。

例えば、新規端末の設置後に既存端末で承認エラーが発生したという問い合わせでは、端末設置・識別情報の検証手順と決済承認エラー対応手順が重要になります。

RAGは回答を装飾するためではなく、Plannerがどのevidenceを確認すべきか決定する基準として使用します。

### 4.3 Planner

PlannerはSOPとtool catalogを参照し、確認すべき`data_need`とread-only toolを選択します。

例えば、`duplicate_tid`が疑われる場合には次のツールが必要です。

| Data need | Tool |
|---|---|
| 端末一覧 | `get_terminals` |
| 決済識別設定 | `get_tid_config` |
| 端末開通履歴 | `get_activation_history` |
| 承認失敗ログ | `get_recent_approval_errors` |

Plannerには次の行為を禁止しています。

| 禁止行為 | 理由 |
|---|---|
| 新しいdata_needを作り出す | 評価契約とtool catalogの整合性が崩れるため |
| 新しいtoolを作り出す | 実際に実行できるツールではないため |
| 原因を確定する | Plannerは調査計画のみを担当するため |
| 決済・返金・設定変更を要求する | 運用上の安全境界に違反するため |

### 4.4 Read-only Tool Gateway

Tool GatewayはSQLite fixtureから運用データを照会します。

すべてのツールは読み取り専用です。

| Tool | 確認内容 |
|---|---|
| `get_store_info` | 店舗の基本情報と運用状態 |
| `get_terminals` | 店舗端末一覧と設置・有効化時刻 |
| `get_tid_config` | 端末別の現在または過去のTID・識別設定 |
| `get_tid_history` | 障害発生時点と現在のTID設定履歴 |
| `get_terminal_identity` | 現場端末番号・シリアルと登録値の比較 |
| `get_installation_history` | 設置、交換、設定変更の履歴 |
| `get_activation_history` | 端末開通、有効化、決済テストの履歴 |
| `get_recent_approval_errors` | 承認失敗履歴と応答メッセージ |
| `get_van_registration` | 加盟店番号とVAN登録状態 |
| `get_pos_front_connection_logs` | POS-Front接続、要求伝達、タイムアウトのログ |
| `get_support_route` | 原因評価後の担当者確認ルート |

S6A・S6Bケースでは、tool failure injectionを利用して、必須ツールの失敗と任意ツールの失敗を区別しました。

### 4.5 Evidence Builder

照会結果をそのまま原因へ変換することはありません。

最初にevidence recordへ変換します。

```text
evidence_id
source_tool
source_record_id
fact_type
normalized_value
observed_at
supports
contradicts
sensitivity
```

例えば、`terminal_identity_mismatch` evidenceには、現場端末番号と登録システム上の番号が異なるという事実を記録します。

`temporal_conflict` evidenceには、障害発生時点ではTIDが重複していた一方、現在は正常化されているという事実を記録します。この場合、現在の値だけを確認すると原因を見落とす可能性があるため、時間軸を持つevidenceが重要です。

### 4.6 Reasoner

Reasonerはevidenceを集約して原因候補を判断します。

ただし、supporting evidenceがない場合は`likely`原因を表示しません。

| Evidence pattern | Cause |
|---|---|
| 同一店舗でactive TIDが重複 + 承認失敗ログ | `duplicate_tid` |
| 現場端末番号・シリアルと登録値が不一致 | `terminal_identifier_mismatch` |
| VAN登録状態がinactive/pending/missing + 登録関連の承認エラー | `van_merchant_registration_missing` |
| POS-Frontのpairing・request delivery失敗ログ | `pos_front_connection_issue` |
| 障害発生時点ではTID重複、現在は正常 | `temporal_conflict` |

### 4.7 Safety Gate

Safety Gateは最終ステータスを決定します。

| 条件 | ステータス |
|---|---|
| 原因候補とsupporting evidenceがある | `READY_FOR_REVIEW` |
| 問い合わせが曖昧でmerchant情報が不足している | `NEEDS_CLARIFICATION` |
| 必須toolが失敗した | `DEGRADED_REVIEW` |
| evidence同士が矛盾している | `CONFLICT_REVIEW` |
| 原因候補がなく、根拠も不足している | `DEGRADED_REVIEW` |

Safety Gateの目的は、できるだけ多くの原因を当てることではなく、根拠のない断定や危険な操作を防ぐことです。

---

## 5. Implementation

本プロジェクトでは、domain logic、core workflow、LLM components、evaluationを分離して実装しました。

主要モジュールは次のとおりです。

| モジュール | 役割 | 簡単な説明 |
|---|---|---|
| `core/contracts.py` | status、evidence、tool response、case briefの型定義 | システム全体の契約 |
| `core/planner.py` | rule-backed deterministic planner | 確認すべきデータを決定 |
| `core/safety.py` | generic safety gate | 根拠不足・矛盾・ツール失敗時のステータス遷移 |
| `domains/offline_payment_ops/parser.py` | 決済障害問い合わせの解析 | 問い合わせをケース候補として整理 |
| `domains/offline_payment_ops/evidence_rules.py` | tool結果をevidenceへ変換 | 照会結果を根拠カードへ変換 |
| `domains/offline_payment_ops/reasoner_rules.py` | evidenceに基づく原因判断 | evidence pattern → cause |
| `domains/offline_payment_ops/safety_rules.py` | 禁止行為の定義 | 決済・返金・設定変更を禁止 |
| `domains/offline_payment_ops/tool_gateway.py` | SQLite read-only toolの実行 | 運用factを照会 |
| `domains/offline_payment_ops/workflow.py` | domain workflowの構成 | parser→planner→tool→evidence→brief |
| `llm/` | bounded LLM parser/planner/drafting | LLMを契約範囲内に制限 |
| `evals/` | deterministic/LLM評価runner | 50-case評価とsmoke test |
| `observability/` | trace、metrics、serialization | 実行履歴と指標を保存 |

実行経路は二つあります。

### Deterministic path

ルールベースのparser・planner・reasonerを使用し、50件のsynthetic dataset全体を評価します。

```text
Golden cases
→ OfflinePaymentWorkflow
→ SQLite read-only tools
→ Evidence
→ Reasoner
→ Safety Gate
→ Case Brief
→ Evaluation Report
```

### LLM path

LLMをparser、planner、checklist、clarification、draftingの一部に利用しながら、同一のtool gatewayとsafety gateを使用します。

つまり、LLMを導入しても次の境界は維持されます。

| 維持される境界 | 意味 |
|---|---|
| allowed data_needのみ使用 | LLMは任意のdata_needを作れない |
| tool catalog内のtoolのみ使用 | LLMは新しいツールを作れない |
| read-only toolのみ実行 | 決済・返金・設定変更は不可 |
| evidenceのない原因断定を禁止 | supporting evidenceが必要 |
| forbidden actionsを遮断 | 機微な操作の出力を防止 |

---
## 6. Evaluation

StoreOps Triage Agentの評価は、二つの層に分けて実施しました。

第一は、**評価データそのものが論理的に正しく構成されているか**を確認する段階です。  
第二は、**エージェントがそのデータから期待されるステータスと原因候補を適切に判断できるか**を確認する段階です。

本プロジェクトで重要なのは、単に「正解を多く当てたか」ではありません。決済運用障害では、原因を誤ること以上に危険な行為があります。それは、**根拠なく原因を断定したり、決済・返金・設定変更などの禁止行為を提案したりすること**です。

そのため、評価では次の観点を併せて確認しました。

| 評価観点 | 確認内容 |
|---|---|
| ステータス判断 | `READY_FOR_REVIEW`、`NEEDS_CLARIFICATION`、`DEGRADED_REVIEW`、`CONFLICT_REVIEW`を正しく判断できたか |
| 原因判断 | `duplicate_tid`、`van_merchant_registration_missing`などの主要原因候補を正しく特定できたか |
| 必須ツール照会 | SOPが要求するread-only toolを漏れなく選択したか |
| 根拠に基づく判断 | evidence citationなしで原因を断定していないか |
| 安全性 | 決済実行、返金、設定変更などの禁止行為を提案していないか |
| 保留判断 | 情報不足、ツール失敗、根拠衝突時に無理に断定していないか |
| LLMの追跡性 | LLMがどの段階で使用され、fallbackが発生したか追跡できるか |

---

### 6.1 Synthetic Dataset Validation

まず、50件のsynthetic datasetそのものが論理的に正しく構成されているかを検証しました。

これはエージェント性能の評価ではありません。  
SQLite fixture内のraw operational factsが、golden labelに記載された原因とステータスを説明できるかを確認するデータ検証段階です。

| 項目 | 結果 |
|---|---:|
| Total cases | 50 |
| Passed | 50 |
| Failed | 0 |

この結果は、評価データ自体が破損していないことを示します。

つまり、各ケースの運用データは、意図した原因とステータスを説明できるよう構成されており、エージェントはそのデータを基に実際に根拠を照会し、原因を判断する必要があります。

重要なのは、raw SQLite table内に正解原因を格納していないことです。  
正解ラベルはgolden setにのみ存在し、エージェントがアクセスする運用テーブルには、端末、TID、承認失敗ログ、VAN登録状態、POS-Front接続ログなどの運用上の事実だけが格納されています。

したがって、エージェントは正解を直接読むのではなく、運用factを照会・組み合わせて原因候補を判断する必要があります。

---

### 6.2 Guardrailを適用したLive LLM Evaluation

DeepSeekを使用したlive LLM経路を、50件のsynthetic case全体に対して実行しました。

実行コマンドは次のとおりです。

```powershell
python -m storeops.evals.llm_runner `
  --provider live `
  --dataset data/golden/offline_payment_ops_cases_50.json `
  --fixture-db data/fixtures/offline_payment_ops_synthetic_50.sqlite3 `
  --output-dir data/eval_reports/llm/deepseek_synthetic_50
```

評価結果は次のとおりです。

| 指標 | 結果 |
|---|---:|
| Total cases | 50 |
| Passed cases | 35 |
| State accuracy | 0.92 |
| Cause accuracy | 0.98 |
| Required tool recall | 0.866 |
| Forbidden action safety | 1.00 |
| Evidence citation coverage | 0.98 |
| Abstention safety accuracy | 1.00 |
| Clarification safety | 1.00 |
| Merchant response safety | 1.00 |
| LLM trace coverage | 0.96 |
| Fallback rate | 1.00 |
| Unsupported claim count | 0 |

この結果は、実際にLLMを導入した経路でも、原因候補とステータス判断が高い水準で維持されたことを示しています。

`state_accuracy`は0.92、`cause_accuracy`は0.98でした。  
また、`unsupported_claim_count`は0でした。つまり、LLM経路においても、根拠のない原因断定は発生しませんでした。

特に重要な安全指標は次のとおりです。

| 安全指標 | 結果 | 意味 |
|---|---:|---|
| Forbidden action safety | 1.00 | 決済実行、返金、設定変更などの禁止行為を提案しなかった |
| Abstention safety accuracy | 1.00 | 根拠不足時に無理に断定しなかった |
| Clarification safety | 1.00 | 曖昧な問い合わせでは追加確認質問へ移行できた |
| Merchant response safety | 1.00 | 店舗側へ危険な案内文を生成しなかった |
| Evidence citation coverage | 0.98 | ほとんどの原因判断がevidenceと接続されていた |

ただし、この結果を「LLMだけで50件を完全に処理できた」と解釈してはいけません。

`fallback_rate = 1.00`であるため、この評価は純粋なLLM単独評価ではなく、**tool catalog、allowed data_need、deterministic fallback、safety gateが共同で動作したguarded LLM evaluation**として解釈する必要があります。

つまり、本プロジェクトにおいてLLMは単独の判断主体ではありません。  
LLMはparser、planner、clarification、draftingなどの段階を補助しますが、実際の運用安全性はread-only tool、evidence rule、safety gate、fallbackによって共同で確保されています。

---

### 6.3 Live LLM Failure Analysis

Live LLM評価では50件中35件が通過し、15件が失敗しました。

失敗ケースの大半は、ステータスや原因そのものを完全に誤ったのではなく、**SOPが要求する必須照会ツールを一部選択しなかったこと**によるものでした。

失敗ケースで漏れていたrequired toolは次のとおりです。

| 漏れていたツール | 漏れた回数 |
|---|---:|
| `get_store_info` | 7 |
| `get_terminal_identity` | 3 |
| `get_activation_history` | 3 |
| `get_support_route` | 3 |
| `get_recent_approval_errors` | 2 |
| `get_tid_config` | 2 |
| `get_terminals` | 1 |
| `get_tid_history` | 1 |

代表的な失敗パターンは次のとおりです。

| ケース種別 | 発生した問題 | 解釈 |
|---|---|---|
| VAN登録未完了ケース | 原因とステータスは正しかったが`get_terminal_identity`を漏らした | VAN問題でも端末identity確認がSOP上必要 |
| 曖昧な問い合わせケース | `get_store_info`を漏らした、または`NEEDS_CLARIFICATION`ではなく`DEGRADED_REVIEW`とした | clarification前に最低限の店舗情報を照会する方針が必要 |
| 必須ツール失敗ケース | `get_activation_history`を漏らした | degraded状態でも、どの必須ツールが失敗したか明示的に記録する必要がある |
| 任意ツール失敗ケース | `get_support_route`を漏らした | 原因判断後の運用引継ぎルート照会を独立したrequired stepとして強制する必要がある |
| 時間軸衝突ケース | `get_tid_history`、`get_activation_history`、`get_recent_approval_errors`を漏らした | 現在状態と障害発生時状態を分離するincident-time照会の強化が必要 |

この分析が重要なのは、LLMの限界が明確に表れているためです。

LLMは問い合わせ文を理解し、原因候補を見つける点では有用です。しかし、SOPが要求するすべての照会ツールを漏れなく選択できるかは、別途検証する必要があります。

そのため、本プロジェクトではLLMをそのまま運用に任せず、次の仕組みを維持しました。

| 安全装置 | 役割 |
|---|---|
| Tool catalog | LLMが使用できるツール一覧を制限 |
| Allowed data_need | LLMが要求できる確認項目を制限 |
| Required tool checklist | 原因種別ごとの必須照会ツールを管理 |
| Deterministic fallback | LLM出力が不足する場合にルールベース経路で補完 |
| Safety gate | 根拠不足、ツール失敗、衝突状況で断定を防止 |
| Evaluation report | 漏れたツールとfailure modeをケース単位で記録 |

最終的に、Live LLM評価は次のメッセージを示しています。

> LLMは決済障害問い合わせを理解し、原因候補を見つける点で有用である。  
> しかし、運用SOPが要求するすべてのevidenceを漏れなく収集させるには、tool checklist、fallback、safety gate、evaluationが不可欠である。

---

### 6.4 Evaluation Takeaway

StoreOps Triage Agentの評価結果を、単に「精度が高い」と要約するだけでは不十分です。

本プロジェクトが示す要点は次のとおりです。

| 評価結果 | 意味 |
|---|---|
| Synthetic validation 50/50通過 | 評価データ自体が論理的に妥当 |
| Deterministic state accuracy 0.90 | ルールベースworkflowが大半のステータスを正しく判断 |
| Deterministic cause accuracy 0.98 | evidenceに基づく原因判断が高い水準で機能 |
| Live LLM state accuracy 0.92 | LLMを導入してもステータス判断が維持 |
| Live LLM cause accuracy 0.98 | LLM経路でも原因候補判断が高い水準で維持 |
| Required tool recall 0.866 | LLMが一部のSOP必須ツールを漏らした |
| Forbidden action safety 1.00 | 危険な決済・返金・設定変更の提案なし |
| Unsupported claim count 0 | 根拠のない原因断定なし |
| Fallback rate 1.00 | LLM単独ではなくguardrail付きの評価 |

したがって、最終的な解釈は次のとおりです。

> StoreOps Triage Agentは、決済障害問い合わせをevidence-backed caseへ変換するうえで有効だった。  
> 一方、実運用水準へ進めるには、required tool checklist、clarification policy、incident-time evidence、support route照会をさらに強化する必要がある。

この結果は、LLM Agentを運用業務へ適用する際の重要な教訓を示しています。

> LLMを導入すること以上に重要なのは、  
> LLMが漏らす可能性のある運用根拠をどのように強制し、  
> 根拠不足の状況でどのように安全に停止させるかである。

---
## 7. Key Design Decisions

### 7.1 決済障害を回答生成の問題ではなく、根拠収集の問題として定義した

加盟店オーナーは「決済できない」と伝えますが、運用担当者は端末、TID、VAN登録、承認ログ、POS接続状態を照合する必要があります。

そのため、本プロジェクトでは「親切な回答を生成すること」よりも、「どの根拠を確認し、その根拠からどの原因候補を提示できるか」に重点を置きました。

### 7.2 すべてのtoolを読み取り専用に制限した

決済障害対応で最も危険なのは、原因が確定していない状態で設定を変更したり、決済取消・返金・外部への引継ぎを案内したりすることです。

そのため、tool gatewayは照会のみを実行し、config mutation、payment execution、refund、cancellationはforbidden actionとして遮断しました。

### 7.3 PlannerとReasonerを分離した

Plannerは、どのデータを確認すべきかを決定します。

Reasonerは、照会されたevidenceを基に原因候補を判断します。

両者を分離した理由は、計画段階で原因を先に断定しないためです。まず必要な根拠を確認し、その後にevidence patternから判断することで、運用上の事故を減らせます。

### 7.4 現在状態と障害発生時点の状態を分離した

決済障害では時間軸が重要です。

現在のTIDが正常であっても、障害発生時点も正常だったとは限りません。S7ケースのように、発生時点ではTID重複が存在したものの、現在は正常化されている場合があります。

そのため、`get_tid_history`とincident-time evidenceを利用し、現在のsnapshotと障害発生時点の状態を分離しました。

### 7.5 必須ツールの失敗と任意ツールの失敗を異なる形で扱った

必須ツールが失敗した場合は、原因を確定すべきではありません。

一方、任意ツールが失敗しても、主要なevidenceが存在すれば、担当者が確認可能なステータスとして残せます。この違いを評価するため、S6A required tool failureとS6B optional tool failureを分離しました。

### 7.6 LLMは契約範囲内でのみ使用した

LLMはcase parser、planner、checklist extractor、clarification、draftingに利用できます。

ただし、LLMが新しいtoolを作成すること、allowed data_need外の項目を要求すること、expected causeを直接出力すること、決済・返金・設定変更を提案することは禁止しました。

---

## 8. Development Notes

本プロジェクトは当初、「店舗の決済障害をLLMが分類するデモ」のように見える可能性がありました。

しかし、開発を進める中で、中心となるのはLLMそのものではなく、**運用データを安全に照会し、根拠に基づいて判断するworkflow**であることが明確になりました。

第一の転換点は、ドメイン原因の分離でした。同じ承認失敗であっても、duplicate TID、端末識別子の不一致、VAN登録未完了、POS-Front接続問題では、確認すべきデータが異なります。そのため、case familyごとにgolden setとrequired tool setを作成しました。

第二の転換点は、synthetic fixtureの設計でした。raw DBに正解原因を入れると、エージェントが実際に推論したのではなく、正解を読み取っただけになります。そのため、正解はgolden JSONにのみ格納し、SQLiteには運用factだけを格納しました。

第三の転換点は、safety gateでした。決済運用では、原因を誤ること以上に、根拠のない確信が危険です。そのため、supporting evidenceのないlikely claimは失敗として扱い、ツール失敗と矛盾する根拠は別のステータスへ振り分けました。

第四の転換点は、LLM評価でした。LLM smoke testではステータスと原因は正しく判断しましたが、required toolを一つ漏らしました。この結果から、LLMは問い合わせ理解には優れている一方、SOP coverageについては別途guardrailとevaluationが必要であることを確認しました。

最終的にStoreOps Triage Agentは「決済障害チャットボット」ではなく、**運用ポリシー、読み取り専用ツール、evidence record、safety gate、evaluationを組み合わせた決済障害調査ワークフロー**として整理されました。

---

## 9. Limitations

本プロジェクトは合成データに基づくポートフォリオ向けMVPであり、実運用システムへ拡張するには追加検証が必要です。

第一に、データはsynthetic 50-case fixtureです。実際の店舗運用データでは、ネットワーク障害、決済ネットワークの遅延、カード会社の応答、複数症状、部分障害がより複雑に混在する可能性があります。

第二に、deterministic evaluationは38/50通過であり、改善の余地があります。特に、曖昧な問い合わせ、required tool recall、optional tool failure、temporal conflictの処理について追加強化が必要です。

第三に、LLM smoke testは1ケースを対象としています。実際のLLM経路を評価するには、50件全体でrequired tool recall、forbidden action safety、evidence citation coverageを繰り返し測定する必要があります。

第四に、現在のtool gatewayはSQLite fixtureに基づいています。実運用環境では、ログシステム、端末管理システム、VAN状態照会、POS連携ログとのconnectorが必要です。

第五に、本システムは操作を実行しません。実運用では、担当者承認後の設定変更、設置会社への確認、VANへの問い合わせ、顧客案内まで接続するworkflowが必要です。

第六に、ポリシー文書はsynthetic SOPです。実際の企業へ適用する際には、実運用ポリシー、障害対応権限、顧客案内文、個人情報・セキュリティ基準を反映する必要があります。

---

## 10. How to Run

### Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

または、requirementsを使用して実行します。

```powershell
pip install -r requirements.txt
```

### Run deterministic evaluation

プロジェクトルートで実行します。

```powershell
$env:PYTHONPATH = "src"
python -m storeops.evals.runner
```

想定されるsummaryは次のとおりです。

```json
{
  "total_cases": 50,
  "passed_cases": 38,
  "state_accuracy": 0.9,
  "cause_accuracy": 0.98,
  "abstention_safety_accuracy": 1.0,
  "unsupported_claim_count": 0
}
```

### Run live LLM smoke test

OpenAI-compatible APIを使用できます。

```powershell
$env:PYTHONPATH = "src"
$env:LIVE_LLM_API_KEY = "your_key"
$env:LIVE_LLM_BASE_URL = "https://api.deepseek.com"
$env:LIVE_LLM_MODEL = "deepseek-chat"
$env:LIVE_LLM_TIMEOUT_SECONDS = "20"

python -m storeops.evals.llm_runner `
  --provider live `
  --fixture-key SYN-001 `
  --output-dir experiments/eval_runs/llm/deepseek_smoke_SYN001
```

50件すべてを実行する場合は、`--fixture-key SYN-001`を削除します。

### Run tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -q -p no:cacheprovider
```

---

## 11. Project Structure

```text
storeops-triage-agent/
├── README.md
├── pyproject.toml
├── config/
├── data/
│   ├── fixtures/
│   │   ├── offline_payment_ops_synthetic_50.sqlite3
│   │   └── offline_payment_ops_synthetic_50_manifest.json
│   ├── golden/
│   │   └── offline_payment_ops_cases_50.json
│   ├── evaluation/
│   │   ├── planner_cases_50.json
│   │   └── retrieval_cases_50.json
│   ├── policies/
│   │   └── offline_payment_ops/
│   └── tool_catalog/
│       └── offline_payment_ops_tools.json
├── docs/
│   ├── architecture.md
│   ├── data-contract.md
│   └── evaluation.md
├── reports/
│   ├── synthetic_50_validation_report.md
│   └── synthetic_50_validation_matrix.csv
├── scripts/
│   └── generate_offline_payment_synthetic_50.py
├── src/
│   └── storeops/
│       ├── core/
│       ├── domains/
│       │   └── offline_payment_ops/
│       ├── evals/
│       ├── infra/
│       ├── llm/
│       └── observability/
├── tests/
└── experiments/
    ├── eval_runs/
    ├── legacy_code/
    ├── legacy_docs/
    └── legacy_s1_s7/
```

`experiments/`には、以前のデモ、legacy S1-S7 assets、過去の出力を保存しています。提出・ポートフォリオ基準のcanonical runtimeは、synthetic 50-case datasetと`src/storeops/`経路です。

---

## 12. What This Project Demonstrates

本プロジェクトは、LLMをオフライン決済の運用障害対応へ適用する際に必要となる、安全なエージェント設計を示します。

第一に、加盟店オーナーからの短い問い合わせを決済運用ケースとして構造化し、原因候補を直ちに断定しないworkflowを構築しました。

第二に、SOP/RAGとtool catalogを利用して確認すべきdata_needを計画し、すべての運用照会をread-only toolに制限しました。

第三に、raw SQLite fixtureには正解を格納せず運用factのみを保持し、エージェントが実際のevidenceを組み合わせて原因を判断するよう設計しました。

第四に、duplicate TID、端末識別子の不一致、VAN登録未完了、POS-Front接続障害、曖昧な問い合わせ、ツール失敗、時間軸上の衝突を含む50件の合成評価セットを作成しました。

第五に、tool responseをそのまま判断へ使用せず、evidence recordとして正規化し、supports・contradictsの関係を記録しました。

第六に、supporting evidenceのないlikely claim、必須ツール失敗、現場説明とシステム記録の矛盾をsafety gateで遮断しました。

第七に、deterministic評価において、state accuracy 0.90、cause accuracy 0.98、abstention safety accuracy 1.00、unsupported claim count 0を記録しました。

第八に、LLMを導入した場合も、allowed data_need、tool catalog、prompt contract、safety gateの範囲内でのみ動作するよう制限しました。

本プロジェクトの中心は、単に決済障害を分類したことではなく、**店舗運用の問い合わせを、根拠照会、安全判断、運用担当者による確認が可能なevidence-backed triage workflowへ変換したこと**です。
