# Financial Recall Agent

金融苦情1件を起点として、同じ原因で被害を受けながらまだ申告していない顧客まで特定し、約款根拠・支給台帳・承認済みルールの実行結果をまとめた、**運用担当者が確認できるリコール調査パッケージ**を生成するLLM Agentプロジェクトです。

本プロジェクトの中心は、「LLMが返金可否を判断すること」ではありません。LLMは苦情内容を理解し、必要な調査フローを整理するために使用します。一方、実際の被害顧客の特定、金額計算、リコール対象の確定は、**承認済みのRule Template、Product Config、SQL、Data Contract、Audit Log**によって統制します。

---

## 1. Overview

金融機関への苦情は、通常、顧客1人からの申告として受け付けられます。

たとえば、ある顧客が「デビットカードのキャッシュバックが付与されていない」と問い合わせた場合、一般的な対応は、その顧客の取引と特典付与状況だけを確認して終了する可能性があります。

しかし、実際の問題はより広い範囲に及ぶことがあります。同じ商品、同じキャンペーン、同じ付与条件で同一の不具合が発生していた場合、苦情を申し立てていない他の顧客も同様の被害を受けている可能性があります。

Financial Recall Agentでは、この問題を次のように定義しました。

> 苦情1件を1人の顧客対応だけで終わらせず、  
> 同じ原因による未申告の被害顧客まで特定する金融消費者保護ワークフローへ拡張する。

代表的なMVPは、`H07 Reward/Cashback/Point/Mileage Missing`です。

顧客がリワード、キャッシュバック、ポイント、マイレージの未付与を申告すると、システムはその苦情を商品約款と支給台帳へ接続し、承認済みルールを実行して、同じ条件下で付与が漏れている顧客を特定します。

代表的な実行結果は、以下のとおりです。

| 項目 | 結果 |
|---|---:|
| 基準苦情 | `EVAL_BASE_0001` |
| Rule Template | `H07_REWARD_MISSING` |
| Product Config | `JB_SMART_CASHBACK_CHECK__2022-07__v2` |
| 被害候補顧客数 | 44人 |
| 未申告の被害候補顧客数 | 43人 |
| 推定被害額 | 70,030ウォン |
| 自動返金の許可 | False |
| 人による確認 | True |
| LLMによるSQL生成 | False |
| Free-form SQLの実行 | False |

この結果の意味は、単に「44人を見つけた」ということではありません。

重要なのは、システムが任意に顧客を抽出したのではなく、承認済みbundle、承認済みSQL、検証済みproduct config、約款根拠、data contractをすべて通過した場合にのみ結果を生成したことです。

したがって、本プロジェクトの結論は次のとおりです。

> LLMは金融判断の主体ではなく、苦情を調査可能な構造へ整理するオーケストレーターである。  
> 実際のリコール判断は、承認済みルール、支給台帳、約款根拠、監査可能な実行記録によって統制しなければならない。

---

## 2. Problem & Objective

金融機関の苦情対応は、顧客1人単位で完結しやすい傾向があります。

しかし、リワード未付与、手数料の誤請求、金利適用の誤り、ポイント失効案内の漏れといった問題は、1人の顧客だけに発生するとは限りません。商品条件や支給ロジックに不具合があれば、同じ条件を持つ複数の顧客に繰り返し発生する可能性があります。

単純な個別顧客対応だけでは、次のような問題が生じます。

| 問題 | リスク |
|---|---|
| 苦情を申し立てた顧客1人だけを処理 | 同じ被害を受けた未申告顧客を見落とす可能性がある |
| LLMが直ちに判断 | 約款、台帳、付与条件を検証せずに断定する可能性がある |
| 任意SQLの生成 | 顧客データ全体を誤って照会したり、過剰検知したりする可能性がある |
| ルールと商品条件の混在 | 別商品へ拡張するたびにハードコーディングが増える |
| 計算結果だけを出力 | なぜその顧客が対象なのかを運用担当者が確認しにくい |
| 自動補償への直結 | 誤返金、重複支給、内部承認漏れのリスクがある |

したがって、本プロジェクトの目的は、単なる「苦情対応チャットボット」ではありません。

目標は、次の3点です。

1. 苦情1件を読み取り、どの調査タイプに該当するかを分類する。
2. そのタイプに対応する承認済みrule bundleだけを実行する。
3. 被害候補顧客一覧、被害額、約款根拠、実行ログ、確認が必要な理由を1つのevidence packageとして生成する。

中心となる問いは、次のとおりです。

> 金融苦情1件を、同一原因による被害顧客の特定と、運用担当者が確認可能なリコール調査パッケージへどのように接続するか？

---

## 3. Data

本プロジェクトでは、公開金融データではなく、金融苦情業務を模したsynthetic datasetと承認済みdemo rule assetを使用しました。

データは単なるモデル学習用テーブルではなく、実際のリコール調査を再現するための業務データ構造として分割しました。

| データグループ | 例 | 役割 |
|---|---|---|
| 苦情データ | `complaint_id`, `customer_id`, `complaint_text`, `product_hint`, `channel` | 苦情タイプの分類と調査の開始点 |
| 顧客契約データ | 顧客ごとの商品加入情報、契約開始日、商品設定 | 当該商品条件が適用される顧客かを確認 |
| 取引台帳 | カード利用履歴、決済日、金額、加盟店／キャンペーンコード | キャッシュバック付与条件を満たすかを確認 |
| 支給台帳 | 実際のキャッシュバック／ポイント付与記録 | 付与漏れの有無を確認 |
| 商品設定 | キャッシュバック率、月間上限、対象外取引、適用期間 | Product Configに基づく計算 |
| 約款・ポリシー根拠 | 商品約款、キャンペーン条件、内部付与基準 | Evidence packageに含める根拠文 |
| 承認済み実行資産 | Rule Template, Product Config, SQL, Bundle | 任意実行を防ぐ統制装置 |
| Audit Log | 実行ID、ルールID、SQL hash、config hash、結果要約 | 事後確認と再現性の確保 |

MVPで使用した代表的な設定は、以下のとおりです。

| 項目 | 値 |
|---|---|
| Rule Template | `H07_REWARD_MISSING` |
| Rule ID | `H07-REWARD-MISSING-TEMPLATE` |
| Product Config | `JB_SMART_CASHBACK_CHECK__2022-07__v2` |
| Data Contract | H07 reward missing investigation contract |
| SQL実行方式 | 承認済みSQLファイルだけを実行 |
| LLMによるSQL生成 | 禁止 |
| 自動返金 | 禁止 |
| 最終判断 | 運用担当者による確認が必要 |

ここで重要な設計は、Rule TemplateとProduct Configの分離です。

`H07_REWARD_MISSING`は、「リワード／キャッシュバック／ポイント／マイレージ未付与」という共通の調査パターンです。一方、`JB_SMART_CASHBACK_CHECK__2022-07__v2`は、特定商品のキャッシュバック率、付与条件、対象外取引、適用期間を保持する設定です。

つまり、同じH07 templateを維持しながら、商品が変わった場合はProduct Configだけを追加する構造です。

---

## 4. Method / System Design

Financial Recall Agentの設計原則は明確です。

> LLMは苦情を理解し、調査フローを整理する。  
> 顧客の特定と金額計算は、承認済みのdeterministic ruleだけで実行する。

全体構造は、以下のとおりです。

```text
苦情1件を入力
   ↓
苦情タイプを分類
   ↓
H07該当性を判断
   ↓
承認済みbundleをロード
   ↓
Rule Templateを検証
   ↓
Product Configを検証
   ↓
Data Contractを検証
   ↓
SQL hashを検証
   ↓
承認済みSQLを実行
   ↓
被害候補顧客 / 未申告顧客 / 被害額を計算
   ↓
約款根拠を接続
   ↓
Evidence Packageを生成
   ↓
Safety Gate
   ↓
Human Review Queue
   ↓
Audit Logを記録
```

### 4.1 LLMの役割

LLMは、苦情テキストを読み、調査フローを整理するために使用します。

たとえば、顧客は次のように申告する可能性があります。

```text
先月付与されるはずだったカード特典のキャッシュバックが入っていないようです。
アプリ上では条件を満たしたと思いますが、付与履歴がありません。
確認をお願いします。
```

LLMまたはrouterは、この苦情がH07のリワード／キャッシュバック未付与に該当する可能性を判断し、追加で確認すべき事項を整理できます。

ただし、LLMが行わない処理は明確です。

| LLMが行わない処理 | 理由 |
|---|---|
| SQL生成 | 顧客台帳全体を任意に照会するのは危険 |
| 被害顧客の確定 | 付与条件と台帳の検証が必要 |
| 被害額の計算 | 承認済み計算ロジックにより再現可能である必要がある |
| 自動返金判断 | 内部承認と人による確認が必要 |
| 約款条件の任意変更 | 承認済みProduct Configとpolicy evidenceが基準 |

つまり、LLMは調査オーケストレーターであり、最終的な計算主体ではありません。

### 4.2 Rule TemplateとProduct Configの分離

初期構造をH07 Smart Cashback専用にハードコーディングすると、商品が変わるたびにコードを修正する必要があります。

これを防ぐため、共通調査パターンと商品別条件を分離しました。

| 区分 | 役割 | 例 |
|---|---|---|
| Rule Template | 共通調査パターン | リワード／キャッシュバック／ポイント／マイレージ未付与の照合 |
| Product Config | 商品別の付与条件 | キャッシュバック率、月間上限、対象外取引、キャンペーンコード |
| Approved SQL | 台帳照合の実行ロジック | 付与条件を満たす顧客のうち未付与顧客を抽出 |
| Bundle | 承認済み実行組み合わせ | template + config + SQL + data contract |

この構造の利点は、拡張性です。

新しい商品が追加されても、`H07_REWARD_MISSING` templateはそのまま使用し、その商品のProduct Configだけを追加できます。

### 4.3 Approved Bundle

Financial Recall Agentは、任意のruleを実行しません。

実行前にbundleを検証します。

Bundleは、次の情報をまとめます。

| 項目 | 確認内容 |
|---|---|
| `rule_template_id` | 承認済みtemplateか |
| `rule_id` | registryに存在し、statusがapprovedか |
| `product_config_id` | 許可されたproduct configか |
| `sql_path` | 承認済みSQLファイルか |
| `sql_sha256` | SQLファイルが変更されていないか |
| `data_contract_id` | 必要なカラムとテーブルが一致しているか |
| `policy_basis_id` | 接続すべき約款／ポリシー根拠が存在するか |
| `approval_status` | 運用承認済みか |

この検証を通過しなければ、ruleは実行されません。

### 4.4 Data Contract

金融台帳の照会で特に危険なのは、データ構造が変更されたにもかかわらず、既存ロジックがそのまま実行されることです。

たとえば、支給台帳のカラム名が変わった、campaign codeが欠落した、取引日の基準が変更されたにもかかわらずSQLを実行すると、誤った対象者を抽出する可能性があります。

そのため、実行前にdata contractを確認します。

| 検証項目 | 例 |
|---|---|
| 必須テーブルの存在 | complaints, card_contracts, transactions, reward_postings |
| 必須カラムの存在 | customer_id, product_id, transaction_date, amount, reward_paid |
| 日付範囲の妥当性 | キャンペーン適用期間と取引期間を確認 |
| 商品configの適用可能性 | product_id, config_id, campaign_codeを確認 |
| 結果スキーマ | affected_customer_count, harm_amountなど必須outputを確認 |

### 4.5 Evidence Package

最終結果は、数値だけを出力しません。

運用担当者が確認できるように、evidence packageを生成します。

| 構成要素 | 説明 |
|---|---|
| Complaint Summary | 苦情原文と要約 |
| Route Result | H07に分類された理由 |
| Policy Basis | 約款／ポリシーの根拠文 |
| Rule Template ID | 使用した共通調査template |
| Product Config ID | 適用した商品別の付与条件 |
| SQL Hash | 実行した承認済みSQLのhash |
| Affected Customer List | 被害候補顧客一覧 |
| Unreported Customer List | 苦情を申し立てていない被害候補顧客 |
| Harm Amount | 顧客別／合計被害額 |
| Supporting Evidence | 対象者判定の根拠 |
| Missing Evidence | 追加確認が必要な情報 |
| Safety Gate Result | 自動返金禁止、人による確認が必要 |
| Audit Log | 実行ID、時刻、入力／出力要約 |

この構造により、運用担当者は「AIがそう判断したから」ではなく、どの約款とどの台帳条件に基づき、その顧客が対象となったのかを確認できます。

---

## 5. Implementation

本プロジェクトは、単純なnotebookではなく、承認済みruleを安全に実行するworkflowとして実装しました。

主なモジュールは、以下のとおりです。

| モジュール | 役割 | 端的な説明 |
|---|---|---|
| `core/bundle_loader.py` | 承認済みbundleのロードと検証 | 許可された組み合わせだけを実行 |
| `core/artifact_hash.py` | SQL/config hashの計算 | 実行資産の改変を検知 |
| `core/data_contract.py` | 入力データスキーマの検証 | 必要なテーブル／カラムを確認 |
| `core/runtime_controls.py` | 禁止動作の遮断 | LLM SQL、自動返金、placeholder ruleを遮断 |
| `templates/h07_reward_missing/` | H07共通調査ロジック | キャッシュバック／ポイント未付与の照合パターン |
| `rules/registry` | 承認済みrule一覧 | 実行可能なruleを管理 |
| `product_configs/` | 商品別の付与条件 | キャッシュバック率、上限、対象外取引 |
| `sql/approved/` | 承認済みSQL | deterministic ledger reconciliation |
| `policy_rag/` | 約款／ポリシー根拠 | reportへ接続する根拠文 |
| `interfaces/cli/` | デモ実行CLI | smoke testとポートフォリオ実行 |
| `evaluation/` | テストと評価 | routing、safety、evidenceを検証 |

実行フローは、以下のとおりです。

```text
1. 苦情recordを入力
2. routerがH07該当性を判断
3. approved bundleをロード
4. rule registryを検証
5. product configを検証
6. SQL hashを検証
7. data contractを検証
8. 承認済みSQLを実行
9. 被害候補顧客と被害額を計算
10. 約款根拠を接続
11. evidence packageを生成
12. safety gateの通過可否を記録
13. audit logへ追記
```

本プロジェクトでは、以下の処理を意図的に禁止しています。

| 禁止事項 | 理由 |
|---|---|
| LLM-generated SQL | 顧客台帳に対する任意照会のリスク |
| Free-form SQL | 承認されていない計算ロジックを実行するリスク |
| Automatic refund | 誤返金や重複補償のリスク |
| Placeholder rule execution | デモ用の暫定ruleが運用結果に利用されるリスク |
| Product configの任意変更 | 商品条件はコード外で承認される必要がある |
| Policy根拠のない計算結果 | 約款根拠のないリコール判断のリスク |

---

## 6. Evaluation

本プロジェクトの評価は、単純な正解率ではなく、**金融業務において安全に実行可能か**を中心に設計しました。

代表的なsmoke testの結果は、以下のとおりです。

```text
complaint_id: EVAL_BASE_0001
rule_id: H07-REWARD-MISSING-TEMPLATE
rule_template_id: H07_REWARD_MISSING
product_config_id: JB_SMART_CASHBACK_CHECK__2022-07__v2
affected_customer_count: 44
unreported_customer_count: 43
total_harm_amount: 70030
decision_status: REQUIRES_HUMAN_CONFIRMATION
human_review_required: True
automatic_refund_allowed: False
used_private_ground_truth: False
llm_generated_sql: False
free_form_sql_allowed: False
```

### 6.1 Business Result

| 項目 | 結果 | 解釈 |
|---|---:|---|
| 基準苦情 | 1件 | 顧客1人からの未付与に関する問い合わせ |
| 被害候補顧客 | 44人 | 同じ条件下で付与漏れの可能性がある顧客 |
| 未申告の被害候補顧客 | 43人 | 苦情を申し立てていないが同様の被害可能性がある顧客 |
| 合計被害額 | 70,030ウォン | 支給台帳照合に基づく推定額 |
| 自動返金 | False | 運用担当者の確認前は補償を禁止 |
| 人による確認 | True | 約款、台帳、対象者の確認が必要 |

この結果は、苦情1件を単純な顧客対応で終わらせず、同一原因による被害顧客の特定へ拡張できることを示しています。

### 6.2 Safety Evaluation

| 検証項目 | 期待結果 | 意味 |
|---|---|---|
| Approved bundleのみ実行 | 合格 | 承認済みrule/config/SQLの組み合わせのみ許可 |
| SQL hash検証 | 合格 | SQLファイルの変更有無を確認 |
| Product config検証 | 合格 | 商品付与条件の承認状態を確認 |
| Data contract検証 | 合格 | 必要なカラムと結果スキーマを確認 |
| Placeholder ruleの遮断 | 合格 | 暫定ruleが実行されないよう防止 |
| LLM SQLの禁止 | 合格 | LLMが台帳照会ロジックを生成しない |
| Free-form SQLの禁止 | 合格 | 未承認SQLの実行を遮断 |
| Automatic refundの禁止 | 合格 | 結果が直ちに補償へ接続されない |
| Human review gate | 合格 | 運用担当者の確認後に後続処理が可能 |
| Audit logへの追記 | 合格 | 実行根拠と結果を再現可能 |

### 6.3 Routing Evaluation

苦情routerは、すべての苦情をH07へ送ってはいけません。

たとえば、単純なアプリ問い合わせ、相談要約、感情的な不満、商品情報の不足、H07ではない手数料・金利・決済問題については、直ちにruleを実行せず、manual reviewまたは別queueへ送る必要があります。

評価データには、以下のcase typeを混在させました。

| Case Type | 目的 |
|---|---|
| 明確なH07未付与申告 | approved ruleが実行されるかを確認 |
| 曖昧な特典問い合わせ | 無理なrule実行を防止 |
| 非H07苦情 | 別queueまたはmanual reviewへ送る |
| 感情的／不完全な苦情 | clarificationまたはmanual reviewへ送る |
| 商品ヒント不足 | product verificationを要求 |
| オペレーター要約型苦情 | 間接表現からrouteを判断 |

重要なのはrecallだけを高めることではなく、誤ったrule実行を防ぐことです。

### 6.4 Evidence Quality

本プロジェクトで重要な品質基準は、結果の数値よりも、**なぜその結果が得られたのかを説明できるか**です。

| 項目 | 確認内容 |
|---|---|
| 約款根拠の接続 | キャッシュバック付与条件と対象外条件がevidenceに含まれるか |
| Rule Templateの記録 | 共通調査パターンが明確に残るか |
| Product Configの記録 | 適用した商品条件が明確に残るか |
| SQL Hashの記録 | どのSQLで計算したかを再現できるか |
| 顧客別被害額 | 合計額だけでなく、顧客別の算出根拠があるか |
| Supporting/Missing Evidence | 根拠と不足情報が分離されているか |
| Safety Gate | 自動返金禁止と人による確認の必要性が表示されるか |

---

## 7. Key Design Decisions

### 7.1 苦情チャットボットではなく、リコール調査workflowとして定義した

金融苦情は、個別顧客への回答だけで終わらせるべきではない場合があります。

1人の顧客による未付与の申告は、同じ商品条件を持つ他の顧客にも被害が生じているシグナルかもしれません。そのため、本プロジェクトは「苦情回答の生成」ではなく、「同一原因による被害顧客の特定と運用担当者向け確認パッケージの生成」として問題を再定義しました。

### 7.2 LLMを判断主体ではなく、調査オーケストレーターに限定した

LLMは、苦情テキストを読みH07の可能性を把握したり、約款根拠を要約したりするうえで有用です。

しかし、LLMが顧客台帳を直接照会し、SQLを生成し、返金可否まで判断することは危険です。そのため、LLMは調査フローを支援する役割に限定し、実際の計算には承認済みruleとSQLだけを使用しました。

### 7.3 Rule TemplateとProduct Configを分離した

初期のH07 Smart Cashback専用構造は短期間で構築できますが、商品が変わるたびにコードが増加します。

そこで、「リワード未付与照合」という共通templateと、「特定商品の付与条件」であるproduct configを分離しました。この構造により、新しいキャッシュバック商品やポイント商品が追加されてもtemplateを再利用できます。

### 7.4 承認済みbundleだけが実行されるようにした

金融データの照会では、実行資産の統制が重要です。

Rule、Product Config、SQL、Data Contract、Policy Basisをbundleとしてまとめ、承認状態とhashを検証した後にのみ実行するようにしました。これにより、運用担当者が承認していないロジックが顧客データへ適用されることを防ぎます。

### 7.5 結果を自動補償へ接続しなかった

被害候補顧客と被害額が計算されても、結果は自動返金へ直結しません。

金融補償では、約款解釈、例外取引、顧客ごとの事情、内部承認、法務・コンプライアンス確認が必要になる可能性があるためです。そのため、結果はhuman review queueへ送り、担当者がevidence packageを確認できるようにしました。

### 7.6 Audit Logを主要成果物とした

金融業務では、結果と同じくらい「どのようにその結果へ到達したか」が重要です。

そのため、実行ID、rule ID、template ID、product config ID、SQL hash、data contract ID、affected count、harm amount、safety gate結果をaudit logへ記録しました。これにより、後から同じ入力と同じ実行資産を用いて結果を再現できます。

---

## 8. Development Notes

本プロジェクトは、当初H07キャッシュバック未付与のデモとして始まりました。

しかし、開発を進める中で、重要なのは「キャッシュバック未付与を1回検出すること」ではなく、金融機関で拡張可能なリコール調査構造を作ることだと明確になりました。

第1の転換点は、ハードコーディングの除去でした。Smart Cashbackだけを処理するコードはデモには適していますが、他のポイント、マイレージ、キャッシュバック商品へ拡張しにくいため、H07 Reward Missing templateとProduct Configを分離しました。

第2の転換点は、SQLの統制でした。「LLM Agent」という名称から、LLMにSQLを生成させ台帳を照会させたくなる可能性がありますが、金融台帳では危険です。そのため、必ず承認済みSQLファイルだけを実行し、hashが一致しなければ遮断する構造にしました。

第3の転換点は、evidence中心の設計でした。単に「被害顧客44人」と出力するだけでは、運用担当者は結果を信頼しにくいと考えました。そのため、約款根拠、product config、SQL hash、顧客別の付与漏れ根拠、missing evidence、safety gateをまとめたevidence packageとして整理しました。

第4の転換点は、自動補償の禁止でした。リコール候補を適切に特定することと、実際の返金を承認することは別の問題です。そのため、`automatic_refund_allowed=False`、`human_review_required=True`を結果に明示しました。

最終的にFinancial Recall Agentは、「金融苦情チャットボット」ではなく、**苦情1件を同一被害顧客の特定と内部確認パッケージへ拡張するcontrolled investigation engine**として整理されました。

---

## 9. Limitations

本プロジェクトはポートフォリオ向けMVPであり、実際の金融機関の運用へ適用するには追加検証が必要です。

1. データはsynthetic datasetに基づいています。実際の金融台帳には、取引取消、部分付与、遡及調整、例外承認、顧客ごとの商品変更など、より複雑なケースが存在します。
2. MVPはH07 Reward/Cashback/Point/Mileage Missingに集中しています。手数料の誤請求、金利誤り、為替スプレッド告知漏れなど、別のタイプには個別のtemplateとproduct configが必要です。
3. 約款文書と内部付与ポリシーのバージョン管理をさらに強化する必要があります。実運用では、商品約款変更日、公示バージョン、内部運用指針のバージョンをすべてauditへ記録しなければなりません。
4. 苦情routerには誤分類の可能性があります。曖昧な苦情に対して無理にruleを実行せず、product verificationまたはmanual reviewへ送る必要があります。
5. 被害額の計算はapproved SQLに基づく推定です。実際の補償前には、顧客別の例外取引、すでに支給された調整額、重複返金の有無を確認する必要があります。
6. 現在の結果は自動返金へ接続していません。実運用には、承認workflow、権限管理、顧客通知、会計処理、異議申立て対応が必要です。
7. LLMが約款根拠を要約する場合にもhallucinationを防ぐ必要があります。実サービスでは、約款文のcitation、version pinning、retrieval evaluationが必要です。

---

## 10. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run H07 smoke test

代表的なH07リワード未付与デモを実行します。

```bash
python -m src.recall_agent.interfaces.cli.h07_reward_missing_demo EVAL_BASE_0001 --json
```

期待される主な出力は、以下のとおりです。

```text
affected_customer_count: 44
unreported_customer_count: 43
total_harm_amount: 70030
decision_status: REQUIRES_HUMAN_CONFIRMATION
human_review_required: True
automatic_refund_allowed: False
llm_generated_sql: False
free_form_sql_allowed: False
```

### Run tests

```bash
python -m pytest tests -q
```

### Recommended local checks

```bash
python -m pytest tests -q
python -m src.recall_agent.interfaces.cli.h07_reward_missing_demo EVAL_BASE_0001 --json
```

---

## 11. Project Structure

```text
financial-recall-agent/
├── README.md
├── requirements.txt
├── data/
│   └── demo/
│       ├── datasets/
│       │   ├── complaints.csv
│       │   ├── card_contracts.csv
│       │   ├── transactions.csv
│       │   └── reward_postings.csv
│       ├── rules/
│       │   ├── rule_registry.json
│       │   ├── product_configs/
│       │   ├── bundles/
│       │   └── data_contracts/
│       ├── policy_rag/
│       │   └── policy_basis.json
│       └── audit/
├── sql/
│   └── approved/
│       └── h07_reward_missing.sql
├── src/
│   └── recall_agent/
│       ├── core/
│       │   ├── artifact_hash.py
│       │   ├── bundle_loader.py
│       │   ├── data_contract.py
│       │   └── runtime_controls.py
│       ├── templates/
│       │   └── h07_reward_missing/
│       ├── policy/
│       ├── evidence/
│       ├── interfaces/
│       │   └── cli/
│       └── evaluation/
├── tests/
│   ├── test_bundle_validation.py
│   ├── test_data_contract.py
│   ├── test_h07_reward_missing.py
│   ├── test_runtime_controls.py
│   └── test_evidence_package.py
└── reports/
    └── demo_outputs/
```

---

## 12. What This Project Demonstrates

本プロジェクトは、LLM Agentを金融苦情業務へ安全に適用するためのエンジニアリング設計を示しています。

1. 苦情1件を顧客1人への対応で終わらせず、同一原因による被害顧客の特定問題へ拡張しました。
2. LLMをSQL生成者や返金判断者ではなく、苦情解釈と調査オーケストレーションの役割に限定しました。
3. H07 Reward MissingをRule Templateとして一般化し、商品別の付与条件をProduct Configへ分離しました。
4. 承認済みbundle、SQL hash、product config hash、data contractを通過した場合にのみ台帳照会が実行される構造にしました。
5. 被害候補顧客数、未申告顧客数、被害額だけでなく、約款根拠とaudit logを含むevidence packageを生成しました。
6. `automatic_refund_allowed=False`、`human_review_required=True`を明示し、自動補償を防ぎ、運用担当者の確認を前提に設計しました。
7. placeholder rule、free-form SQL、LLM-generated SQL、data contract mismatchといった危険な実行経路を遮断しました。
8. 代表的なsmoke testでは、苦情1件から被害候補顧客44人、未申告の被害候補顧客43人、推定被害額70,030ウォンを検出しました。

本プロジェクトの中心は、単に金融苦情チャットボットを作成したことではなく、**苦情シグナルを、約款・台帳・承認済みルール・監査ログが接続されたリコール調査ワークフローへ変換したこと**です。
