# TECH-NOTE

本リポジトリは、データ分析、機械学習、ゲーム／プロダクト分析、計量経済学・因果推論、NLP／LLMベースのシステム開発プロジェクトをまとめた技術ポートフォリオです。

各プロジェクトの詳しい課題設定、データ、実装方法、評価結果、限界については、それぞれのプロジェクトフォルダ内にある `README.md` をご覧ください。

---

## Projects

## 1. Applied AI / LLM Systems

このセクションでは、LLMを単なる回答生成器としてではなく、実際の業務フローの中でテキストを理解し、根拠を整理し、人が確認可能な成果物へ変換するコンポーネントとして活用したプロジェクトを紹介します。

主な関心は「LLMで何ができるか」ではなく、「どこまでをLLMに任せ、どの部分をルール検証、データ確認、人によるレビューで制御すべきか」にあります。

### Financial Recall Agent

Financial Recall Agentは、1件の金融苦情を起点として、同じ原因で影響を受けた顧客を特定し、レビュー用の証拠パッケージを生成する、LLMベースの金融消費者保護プロジェクトです。

本プロジェクトでは、LLMを金融判断の主体として使用していません。LLMは苦情内容の理解や問題候補の分類に使用しますが、実際の影響範囲の特定、金額計算、補償可能性の判断は、承認済みルール、取引データの検証、監査ログ、人によるレビューに分離しています。

中心となる設計思想は、LLMで回答を生成することではなく、金融業務に必要な根拠確認、再現性、監査可能性、人による確認をシステムの中核に置くことです。

- 主な技術・概念: LLM, RAG, Rule-based Verification, Audit Log, Human Review
- プロジェクトフォルダ: [financial-recall-agent](./financial-recall-agent)

### StoreOps Triage Agent

StoreOps Triage Agentは、実店舗から寄せられる決済障害の問い合わせを構造化し、運用担当者が確認すべき証拠と対応方針を整理する、LLMベースのオペレーション支援プロジェクトです。

本プロジェクトでは、店舗からの問い合わせを単純なチャットボット回答だけで処理しません。同じ「決済失敗」であっても、原因は端末設定、POS接続、承認ログ、VAN／PSP、カード会社の応答など、複数のシステムに分かれるためです。

そのため、LLMは症状理解と確認項目の整理に使用し、実際の判断は読み取り専用ツールの照会結果とルールベースの検証によって支援する構成としました。最終的な目的は、運用担当者が障害原因をより早く絞り込み、不要なエスカレーションや誤案内を減らすことです。

- 主な技術・概念: LLM Agent, Tool Calling, RAG, Payment Operations, Safety Gate
- プロジェクトフォルダ: [storeops-triage-agent](./storeops-triage-agent)

### Recover24

Recover24は、ボイスフィッシングや金融詐欺の被害者による申告内容をもとに、被害回復に必要な文書と事案概要を生成する、NLP／LLMベースの文書自動化プロジェクトです。

被害者は被害直後、どの情報をどこへ提出すべきか判断しにくく、銀行や関係機関は被害経緯、送金情報、漏えい情報、証拠資料を構造化して確認する必要があります。本プロジェクトでは、被害者の自然言語による申告と入力情報をもとに事案を整理し、文書生成の可否を確認したうえで、提出可能な形式の資料を作成するフローを設計しました。

LLMは申告内容の意味理解や不足情報の確認に使用しますが、文書生成前には、情報の矛盾、必須項目の欠落、安全に提出可能かどうかを別途確認する構成としています。

- 主な技術・概念: NLP, LLM, Information Extraction, Document Automation, Safety Check
- プロジェクトフォルダ: [recover24](./recover24)

### Provider Directory Control Tower

Provider Directory Control Towerは、医療提供者情報の正確性を公式データと照合し、修正の必要性を判断する、運用型データ品質管理プロジェクトです。

医療提供者の氏名、住所、電話番号、診療状況といった情報は、小さな誤りであっても、ユーザーの検索体験や運用上の信頼性に影響します。本プロジェクトでは、既存の提供者情報を公式情報源と比較して変更候補を生成し、自動修正可能なケースと人による確認が必要なケースを分離します。

重要なのは、検索結果をそのまま正しいとみなすのではなく、公式情報源に基づく証拠、情報の競合、信頼度スコア、レビュールーティングを一体として設計した点です。

- 主な技術・概念: Data Quality, Evidence Matching, Confidence Scoring, Review Routing
- プロジェクトフォルダ: [provider-directory-control-tower](./provider-directory-control-tower)

---

## 2. Machine Learning / Credit Risk

このセクションでは、機械学習を単なる予測精度の問題としてではなく、実際の意思決定者が「誰を優先的に確認するか」「どの変数を採用するか」「どのリスク区間で対応するか」へつなげることを重視したプロジェクトを紹介します。

主な関心は、モデル性能だけでなく、データリークの防止、検証設計、リスク区間の設計、特徴量採用の妥当性、実運用への接続可能性にあります。

### AMEX Credit Risk Decisioning

AMEX Credit Risk Decisioningは、クレジットカード顧客の債務不履行リスクを予測し、その結果をリスク部門の月次レビュー優先順位へ変換した機械学習プロジェクトです。

本プロジェクトでは、債務不履行の有無を単に当てる分類問題としてだけ扱っていません。実務では、すべての顧客を同じように確認することはできないため、モデルスコアをもとに「誰を先に確認すべきか」を決めることの方が重要です。

そこで、顧客ごとのリスクスコアを生成し、上位リスク区間におけるPrecisionとLiftを確認したうえで、月次レビュー対象の範囲をどのように設定できるかまで検討しました。

- 主な技術・概念: LightGBM, XGBoost, CatBoost, Tabular MLP, Ranking, Top-K Evaluation
- 主な結果: 上位リスク区間で高いPrecisionとLiftを確認し、リスクスコアをレビュー優先順位として解釈
- プロジェクトフォルダ: [amex-credit-risk-decisioning](./amex-credit-risk-decisioning)

### Xente Credit Feature Adoption

Xente Credit Feature Adoptionは、返済履歴を持たない顧客群において、取引行動変数が信用評価の特徴量として採用できるほど独立したリスクシグナルを持つかを検証したプロジェクトです。

当初の問いは「取引履歴のない顧客は、より高リスクなのか」という単純なものでした。しかし分析を進めると、取引行動変数は顧客自身のリスクだけでなく、サービスフロー、商品カテゴリ、貸付事業者の構造とも強く結びついていることが分かりました。

そのため、取引行動変数を直ちに中核的な信用評価変数として採用するのではなく、既存情報に対して追加的な予測価値を持つかを検証しました。結果として、主要変数として直ちに採用するよりも、特定の高リスクセグメントを観察する補助変数として活用する方が適切だと判断しました。

- 主な技術・概念: LightGBM, Logistic Regression, Stratified Group K-Fold, Permutation Test, Feature Adoption
- 主な観点: 予測力があるように見える変数でも、独立した意思決定価値を持つかを検証する必要がある
- プロジェクトフォルダ: [xente-credit-feature-adoption](./xente-credit-feature-adoption)

---

## 3. Game / Product Analytics

このセクションでは、ゲームやデジタルプロダクトのデータを用いて、将来価値の予測、施策効果の検証、ユーザー行動の分解、収益構造の理解を行ったプロジェクトを紹介します。

主な関心は、モデル精度や有意差だけでなく、その結果がユーザー獲得、マーケティング投資、施策判断、運用上の意思決定にどのようにつながるかにあります。

### Mobile Game LTV Production-Style ML Pipeline

Mobile Game LTV Production-Style ML Pipelineは、モバイルゲームのLTVを予測するために、データ検証、モデリング粒度の確認、特徴量生成、時系列検証、最終モデルの再学習、予測成果物、モデルカード、UA施策分析までを一連のパイプラインとして構成したプロジェクトです。

単一モデルの性能だけを示すのではなく、生データの品質確認から実験管理、最終モデル選定、ビジネス利用可能な出力の生成までを再現可能な形で整理しました。

- 主な技術・概念: ML Pipeline, LTV Forecasting, Feature Engineering, Two-Stage Model, XGBoost, Optuna, Time-Based Validation, Business Analysis
- プロジェクトフォルダ: [mobile-game-ltv-pipeline](./mobile-game-ltv-pipeline)

### Gamelytics A/B Analysis

Gamelytics A/B Analysisは、モバイルゲームのプロモーション施策を対象に、ARPUの分解、課金転換率の検定、ブートストラップによる不確実性評価、置換検定、Whaleユーザーへの感度分析、リテンション補足分析を行った再現可能なA/Bテスト分析プロジェクトです。

平均値の差だけで結論を出すのではなく、収益差が課金者数の増加によるものか、少数の高額課金者に依存したものか、結果がどの程度安定しているかを複数の観点から検証しました。

- 主な技術・概念: A/B Testing, ARPU, Conversion Rate, Bootstrap Confidence Interval, Permutation Test, Revenue Concentration, Retention Analysis
- プロジェクトフォルダ: [gamelytics-ab-analysis](./gamelytics-ab-analysis)

---

## 4. Econometrics / Causal Inference

このセクションでは、単純な相関や政策前後の比較ではなく、政策・社会・市場データにおいて、どの比較が意味を持つのか、どの仮定のもとで結果を解釈できるのかを検討したプロジェクトを紹介します。

主な関心は、DID、Event Study、Fixed Effects、頑健性検証、Placebo Test、識別仮定、結果を解釈できる範囲と限界にあります。

### Thailand Policy Revenue Persistence

Thailand Policy Revenue Persistenceは、消費支援政策後における宿泊・飲食サービス業の売上反応が一時的な効果なのか、それともその後も残る持続的な需要シグナルなのかを分析したプロジェクトです。

政策後に売上が増加したとしても、それを直ちに返済余力と解釈することはできません。政策による消費は短期的に売上を押し上げる可能性がありますが、その効果がその後も持続するかは別途検証する必要があります。

本プロジェクトでは、政策対象業種と比較業種の相対的な動きを構成し、Lag Model、Synthetic Comparator、Event Studyを用いて、政策後の反応がいつ現れ、どの程度持続したかを確認しました。

- 主な技術・概念: Lag Model, Synthetic Comparator, Event Study, Robustness Check
- 主な結果: 政策後0〜3か月の累積相対反応において、正のシグナルを確認
- プロジェクトフォルダ: [thailand-policy-revenue-persistence](./thailand-policy-revenue-persistence)

### Korea SECA → Kyushu SO₂

Korea SECA → Kyushu SO₂は、韓国のSECA Step 1施行後、日本の九州地域における沿岸部と内陸部のSO₂格差が縮小したかを分析した環境政策の因果推論プロジェクトです。

本プロジェクトでは、船舶燃料規制が規制区域内だけでなく、風によって移動する大気汚染にも影響を与える可能性を検討します。そのため、観測局・月単位のパネルデータを構築し、沿岸地域と内陸地域の変化の差を比較しました。

DID、Fixed Effects、Event Study、頑健性検証を用いて政策後のSO₂変化を確認し、主要な仕様では沿岸地域のSO₂が相対的に低下する方向性を確認しました。

- 主な技術・概念: DID, Event Study, Fixed Effects, Robustness Check
- 主な観点: 環境政策の効果を評価するには、単純な前後比較ではなく、適切な比較群と事前トレンドの確認が重要
- プロジェクトフォルダ: [causal-inference-seca](./causal-inference-seca)

### Refugee Inflows → Crime Rates

Refugee Inflows → Crime Ratesは、難民流入規模と犯罪率の変化の関係を、国・年パネルデータを用いて分析した社会データプロジェクトです。

難民と犯罪率の関係は社会的に敏感なテーマであるため、単一の回帰結果だけで強い結論を出すことは危険です。本プロジェクトでは、国ごとの固定的特性、年ごとの共通ショック、国別トレンド、先行・遅行関係をあわせて検討し、結果の安定性を確認しました。

主な結果として、難民流入と複数の犯罪指標の間に、一貫して強い正の関係を確認することはできませんでした。特に一部の相関は、追加的なトレンド統制を導入すると弱まりました。

- 主な技術・概念: Panel Data, Fixed Effects, Dynamic Check, Robustness Check
- 主な観点: 社会的に敏感な主張は、単一の係数ではなく、複数の検証と限定的な解釈に基づく必要がある
- プロジェクトフォルダ: [refugees-crime-panel](./refugees-crime-panel)

### Dominick's Price Elasticity IV-DML

Dominick's Price Elasticity IV-DMLは、小売スキャナーデータを用いて、価格変化が販売数量に与える影響を推定した価格弾力性分析プロジェクトです。

価格と販売数量の関係を単純回帰だけで解釈することは困難です。価格は需要変化、プロモーション、店舗特性、商品特性と同時に動く可能性があるためです。そこで本プロジェクトでは、固定効果、IV、DMLベースのアプローチを比較し、価格弾力性の推定値が手法や統制方法にどの程度敏感かを確認しました。

重要なのは、単一の価格弾力性の数値を提示することではなく、どの識別戦略と頑健性検証を通じて、その数値をどこまで信頼できるかを説明することです。

- 主な技術・概念: Price Elasticity, Fixed Effects, IV, DML, Robustness Check
- 主な観点: 価格弾力性は、単一の回帰係数よりも識別戦略と検証プロセスが重要
- プロジェクトフォルダ: [dominicks-price-elasticity-iv-dml](./dominicks-price-elasticity-iv-dml)
