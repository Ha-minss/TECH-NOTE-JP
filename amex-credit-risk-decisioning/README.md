# AMEX Credit Risk Decisioning

## カード会社は、誰を優先して確認すべきか？

クレジットカード顧客の債務不履行リスク予測スコアを、リスク部門の月次レビュー優先順位、Top-Kレビュー方針、コスト感応度シミュレーションへ変換したCredit Risk Decisioningプロジェクトです。

本プロジェクトは、「誰がデフォルトするか」を予測するだけでは終わりません。顧客ごとのリスクスコアを作成し、限られたレビュー人員の中で「今月、誰を優先して確認すべきか」という運用上の意思決定問題へ変換します。

本リポジトリは、American Expressが公開したKaggleデータを用いた個人プロジェクトを、金融機関における信用リスクモデルの開発・検証職向けポートフォリオとして整理したものです。実際のカード会社の運用システムではなく、自動承認・自動否決モデルでもありません。

---

## 1. Overview

クレジットカード会社のリスク部門が直面する課題は、単に「誰が債務不履行に陥るか」を予測することではありません。

実際の運用では、すべての顧客を毎月詳細に確認することはできません。限られたレビュー人員の中で、今月誰を優先して確認し、どのリスク区間まで介入するかを決める必要があります。

リスクを十分に把握できなければ信用損失が拡大する可能性があります。一方で、過度に保守的な管理を行うと、実際には正常に返済する顧客まで不必要に確認することになり、顧客体験や営業機会を損なう可能性があります。

本プロジェクトでは、American Expressの公開データを用いてクレジットカード顧客の債務不履行リスクスコアを作成し、それをレビュー優先順位の設計へ接続しました。

中心となる問いは、次のとおりです。

> クレジットカードの債務不履行リスクスコアを作成したとき、リスク部門は今月誰を優先して確認すべきか？

最終モデルのスコアは、顧客ごとの絶対的なデフォルト確率を断定する値ではなく、限られたレビューリソースの中で高リスク顧客を上位に配置するための**ランキング指向のリスクスコア**として解釈しました。

主な結果は以下のとおりです。

| 結果 | 値 | 解釈 |
|---|---:|---|
| 8モデル OOF equal blend | AMEX 0.797631, ROC AUC 0.962782 | 元ノートブックで確認された公開可能な最良の結合結果 |
| Ridge stacking | AMEX 0.797538, ROC AUC 0.962718 | OOF stackingの検証結果 |
| Top 10% Capture Rate | 37.30% | リスクスコア上位10%が観測デフォルトの37.30%を捕捉 |
| Top 10% Lift | 3.73 | モデリング標本内におけるデフォルトの集中度 |
| D1 観測デフォルト率 | 96.59% | 最もリスクが高いデシル |
| D10 観測デフォルト率 | 0.04% | 最もリスクが低いデシル |

コスト感応度シミュレーションでは、基準コスト仮定のもとでTop 17%が最大のシミュレーション純便益を記録したモデリング標本上のカットオフでした。一方、ConservativeシナリオではTop 4%、AggressiveシナリオではTop 28%が最大のシミュレーション純便益を示しました。

したがって、本プロジェクトの結論は次のとおりです。

> モデルはリスクの順位を作るが、実際のレビュー範囲は、損失規模、介入効果、レビューコスト、顧客負担コストといったビジネス上の仮定によって決まる。

---

## 2. Problem & Objective

信用リスクモデルは、顧客ごとの債務不履行リスクスコアを作成できます。しかし、実際のカード会社の運用では、スコアそのものが直ちに意思決定になるわけではありません。

リスク部門が確認すべきなのは、単に「誰が危険か」ではなく、限られたレビュー人員の中で「今月誰を優先して確認するか」です。

| 判断上の誤り | 発生し得る問題 |
|---|---|
| レビュー範囲が狭すぎる | 実際の高リスク顧客を見落とし、信用損失が拡大する可能性がある |
| レビュー範囲が広すぎる | 正常顧客まで確認し、運用負荷と顧客体験上の問題が生じる |

そのため、リスク部門はリスクスコアだけでなく、レビュー対象の規模、高リスク顧客の捕捉範囲、正常顧客を確認する負担を同時に考慮する必要があります。

本プロジェクトでは、AMEXデータを単純なdefault prediction問題としてのみ扱いませんでした。顧客ごとのリスクスコアを作成したうえで、それをリスク区間、レビュー優先順位、Top-K方針の比較、コスト感応度シミュレーションへ変換する問題として再定義しました。

分析目標は以下のとおりです。

1. 顧客・月単位のデータを顧客単位のrisk profileへ変換する。
2. 複数モデルを用いて、安定したリスク順位スコアを生成する。
3. リスクスコア上位区間ごとにPrecision、Capture Rate、Lift、正常顧客のレビュー負担を計算する。
4. AMEX competition sampling-adjusted scenarioとして、非デフォルト顧客を20倍に加重したシナリオを別途計算する。
5. EAD、LGD、介入効果、レビューコスト、顧客負担コストの仮定に応じて、レビュー範囲がどのように変わるかをシミュレーションする。
6. 公開リポジトリには元データ、全feature parquet、全OOF予測値、学習済みモデルを含めず、検証可能なコードと小規模な集計結果表のみを残す。

---

## 3. Data

元データは、顧客ごとに複数月の記録で構成されています。予測対象は顧客単位の債務不履行有無であるため、顧客・月データをそのまま用いるのではなく、顧客1人の過去行動を要約したcustomer-level risk profileへ変換しました。

変数名は匿名化されているため、個々の変数の金融上の意味を直接解釈することは困難です。特定の変数を所得、利用限度額、延滞などと断定するのではなく、顧客の長期的な水準、変動性、直近の変化、欠損パターンが安定したリスクシグナルになり得ると考えました。

| Feature block | 着目する観点 | リスク上の解釈 |
|---|---|---|
| Summary | 顧客の全体的な水準 | 長期的な行動水準 |
| Temporal | 初回と最終時点の変化 | 時間経過に伴う悪化または改善 |
| Recent window | 直近3か月・6か月のパターン | 最近のリスクシグナルを反映 |
| Missingness | 欠損数と欠損率 | 観測可能性またはデータ空白の変化 |
| Pivot-lite | 月ごとの位置情報を一部保持 | 集約時に失われる時系列順序情報を緩和 |

最終スコアは、絶対確率ではなくリスク順位スコアとして解釈しました。

その理由は、公開データの標本構造が実際のカードポートフォリオの分布と異なる可能性があり、AMEXデータのnon-default標本も実際の母集団比率とは異なる形で構成されているためです。そのため、本プロジェクトでは「確率補正された実際のデフォルト率」よりも、「リスク部門が優先して確認すべき顧客を適切に並べられるか」に焦点を置きました。

---

## 4. Method / System Design

本プロジェクトの中心は、単一モデルの性能ではなく、モデルスコアを運用上の意思決定へ変換する構造です。

```text
Raw Customer-Month Data
        |
        v
Customer-level Risk Profile
(summary / temporal / recent / missingness / pivot-lite)
        |
        v
Default Risk Score
(LightGBM / XGBoost / CatBoost / Tabular MLP / OOF blending)
        |
        v
Risk Ranking
(customer-level review priority)
        |
        v
Top-K Policy Simulation
(precision / capture / lift / observed and 20x weighted workload)
        |
        v
Cost Scenario Analysis
(EAD / LGD / intervention effect / review cost / customer friction)
        |
        v
Validation Artifacts
(aggregate tables / provenance docs / smoke-testable modules)
```

モデリングは、以下の観点で構成しました。

| 区分 | 使用モデル・変数の観点 | 役割 |
|---|---|---|
| ベースラインモデル | 全変数を用いたLightGBM | 大規模な表形式データで高速に基準性能を作成 |
| 補助ブースティングモデル | XGBoost, CatBoost | 別のブースティング実装でもリスク順位が維持されるかを確認 |
| LightGBM派生モデル | DART, GOSS | 同一feature空間で学習方式の違いによるOOF性能を比較 |
| 直近変化モデル | Recent/change featureの観点 | 長期平均に埋もれ得る直近の悪化シグナルを補完 |
| 月別位置モデル | Pivot-lite featureの観点 | 顧客・月集約時に失われ得る時間的位置情報を保持 |
| 非ツリーモデル | Tabular MLP | ツリーモデルとは異なる関数形を持つ補助候補 |
| 最終結合 | 8モデルのOOF予測値を等加重平均 | 単一モデル依存を抑えた最終リスク順位スコア |

ベースラインモデルにはLightGBMを採用しました。顧客数と変数数が多いtabular dataであり、変数の意味が匿名化され、欠損パターンと非線形関係が同時に作用する可能性が高いと考えたためです。

ただし、最終的なリスク順位を1つのベースラインモデルだけに依存させないため、複数のモデルと異なる変数セットを併せて検証しました。各モデルのスコアは、学習fold外で予測されたOOFスコアを基準に比較しました。その後、複数モデルのOOFスコアを結合し、最終的なリスク順位スコアを作成しました。

---

## 5. Implementation

本リポジトリは、元のColab実験を公開ポートフォリオ向けに整理したclean repositoryです。元データ、全feature parquet、全OOF予測値、学習済みモデルファイルは含まれていません。

実装フローは以下のとおりです。

1. 顧客・月データを顧客単位のrisk profileへ変換する。
2. Summary、Temporal、Recent window、Missingness、Pivot-lite featureを生成する。
3. LightGBM、XGBoost、CatBoost、Tabular MLPなど複数モデルを学習するための設定とコードを残す。
4. fold外のOOF予測値を基準にモデル性能を比較する。
5. 8モデルのOOFスコアを等加重平均し、最終リスク順位スコアを作成する。
6. 顧客をリスクスコア順に並べる。
7. Top 1%、5%、10%、20%のレビュー区間ごとにtrade-offを計算する。
8. Observed Precisionと20x weighted scenario Precisionを分けて算出する。
9. EAD、LGD、介入効果、レビューコスト、顧客負担コストを仮定し、threshold感応度分析を行う。
10. 結果の出所を`docs/results_provenance.md`に記録する。

運用観点で公開する成果物は、以下のとおりです。

| 成果物 | 保存先 | 役割 |
|---|---|---|
| Model CV summary | `outputs/tables/model_cv_summary.csv` | 元ノートブックに基づくモデル別検証性能 |
| Blend comparison | `outputs/tables/blend_comparison.csv` | OOF blendとstackingの比較 |
| Top-K policy table | `outputs/tables/topk_policy_tradeoff.csv` | 観測ベースのPrecision、Capture、Lift |
| 20x weighted policy table | `outputs/tables/weighted_policy_tradeoff.csv` | 非デフォルト顧客を20倍に加重したシナリオ |
| Cost scenario table | `outputs/tables/top17_base_cost_scenario.csv` | 基準コスト仮定におけるTop 17%カットオフの検証 |
| Decile summary | `outputs/tables/risk_decile_summary.csv` | リスクデシル別の観測デフォルト率 |
| Synthetic smoke data | `data/sample/synthetic_scores.csv` | テスト専用の合成データ |

この構造の目的は、単に優れたモデルを1つ作ることではなく、モデルスコアをリスク部門が利用できるレビュー優先順位とthreshold方針へ変換することです。

---

## 6. Evaluation

評価は単純なモデル性能表ではなく、リスクスコア上位の顧客群に実際の債務不履行顧客がどの程度集中しているかを基準に整理しました。

主な評価観点は、次の4点です。

1. リスクスコア上位区間が実際のデフォルトを適切に捕捉しているか。
2. レビュー範囲を広げたとき、Capture Rateと正常顧客のレビュー負担がどのように変化するか。
3. スコアデシル別のデフォルト率が単調に並んでいるか。
4. コスト仮定によって適切なレビューthresholdがどのように変化するか。

### 6.1 Top-K Review Policy Simulation

OOF検証スコアで顧客をリスク順に並べたところ、上位区間には観測されたデフォルト顧客が強く集中していました。

| レビュー区間 | レビュー対象数 | 捕捉したデフォルト顧客 | Capture Rate | Lift | 観測正常顧客 | 20x weighted正常顧客 | Observed Precision | 20x weighted scenario Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top 1% | 4,590 | 4,589 | 3.86% | 3.86 | 1 | 20 | 99.98% | 99.57% |
| Top 5% | 22,946 | 22,742 | 19.14% | 3.83 | 204 | 4,080 | 99.11% | 84.79% |
| Top 10% | 45,892 | 44,326 | 37.30% | 3.73 | 1,566 | 31,320 | 96.59% | 58.60% |
| Top 20% | 91,783 | 81,176 | 68.31% | 3.42 | 10,607 | 212,140 | 88.44% | 27.68% |

Top 5%区間のObserved Precisionは99.11%、Liftは3.83と高い一方、全デフォルト顧客に対する捕捉率は19.14%でした。そのため、Top 5%は全高リスク顧客を十分に捕捉する方針というよりも、優先的に確認すべき高リスク候補群として解釈するのが適切です。

レビュー範囲をTop 20%まで広げるとCapture Rateは68.31%まで上昇しますが、正常顧客を確認する負担も増加します。AMEX competition sampling-adjusted scenarioとして非デフォルト顧客を20倍に加重すると、Top 20%における20x weighted正常顧客負担は212,140人まで増え、20x weighted scenario Precisionは27.68%まで低下します。

したがって、レビュー区間はPrecisionだけではなく、捕捉率とレビュー負担のバランスで判断する必要があります。

### 6.2 Score Band Analysis

顧客をリスクスコアに基づいて10デシルに分けて確認しました。

| スコアデシル | 解釈 |
|---|---|
| D1 | 最もリスクが高い区間。観測デフォルト率96.59% |
| D10 | 最もリスクが低い区間。観測デフォルト率0.04% |

この結果は、最終スコアが一部の上位顧客だけを選別したのではなく、全顧客をリスク順に区分するうえでも一貫して機能したことを示しています。

ただし、この値も公開データの標本構造を反映した観測デフォルト率であり、実際のカードポートフォリオにおける絶対的なデフォルト率として解釈してはいけません。

重要な解釈は次のとおりです。

> 最終リスクスコアは、顧客を相対的なリスク順に並べるうえで有用だった。ただし、実際の運用では、このスコアを確率ではなくレビュー優先順位として利用する方が安全である。

### 6.3 Threshold Sensitivity Analysis

これまでの結果は、リスクスコアが顧客をリスク順に適切に並べることを示しています。しかし、実際の運用で重要な問いは「上位何%まで確認するか」です。

このレビュー範囲は、モデル性能だけで決まるものではなく、損失規模、介入効果、レビューコスト、正常顧客を確認する負担によって変化します。

基準シナリオは、以下のように設定しました。

| 項目 | 基準値 | 意味 |
|---|---:|---|
| EAD | 1.00 | エクスポージャー額を正規化 |
| LGD | 0.50 | デフォルト発生時の損失率 |
| 介入効果 | 0.20 | レビュー・介入によって損失を削減できる割合 |
| 1件当たりレビューコスト | 0.010 | 顧客1人を確認するコスト |
| 顧客負担コスト | 0.005 | 正常顧客を確認することによるfriction cost |
| 正常顧客の加重 | 20x | AMEX competition sampling-adjusted scenario |

正規化純便益は、以下の方法で計算しました。

```text
想定損失削減額 = 捕捉したデフォルト顧客数 * EAD * LGD * 介入効果
レビューコスト = 20x weighted実質レビュー件数 * 1件当たりレビューコスト
顧客負担コスト = 20x weighted正常顧客レビュー負担 * 1件当たり顧客負担コスト
シミュレーション純便益 = 想定損失削減額 - レビューコスト - 顧客負担コスト
```

基準シナリオの結果は、以下のとおりです。

| レビュー区間 | 捕捉したデフォルト顧客 | 20x weighted正常顧客負担 | 20x weighted実質レビュー件数 | 20x weighted実質レビュー率 | シミュレーション純便益 |
|---|---:|---:|---:|---:|---:|
| Top 5% | 22,742 | 4,080 | 26,822 | 5.84% | 1,985.58 |
| Top 10% | 44,326 | 31,320 | 75,646 | 16.48% | 3,519.54 |
| Top 17% | 71,205 | 136,220 | 207,425 | 45.20% | 4,365.15 |
| Top 20% | 81,176 | 212,140 | 293,316 | 63.91% | 4,123.74 |

基準コスト仮定のもとで、Top 17%は最大のシミュレーション純便益を記録したモデリング標本上のカットオフでした。ただし、これは最適な運用方針であることを意味しません。実運用前には、顧客別EAD、実際のLGD、回収率、介入効果、レビューコスト、顧客体験コストを追加で測定する必要があります。

つまり、Capture Rateを高める方針が、常により良い運用方針になるとは限りません。

### 6.4 コストシナリオ別のレビュー範囲

コスト仮定を変更すると、最大のシミュレーション純便益を記録するレビュー範囲も変化しました。

| シナリオ | 解釈 | 最大のシミュレーション純便益を記録したモデリング標本上のカットオフ |
|---|---|---:|
| Conservative | レビューコストと顧客負担コストをより大きく見る仮定 | Top 4% |
| Base | 基準仮定 | Top 17% |
| Aggressive | 損失規模と介入効果をより大きく見る仮定 | Top 28% |

この分析の目的は、特定の区間を唯一の正解として提示することではありません。

モデルはリスク順位を作りますが、レビュー範囲はビジネス上の仮定によって決まります。リスクスコアは誰を先に確認するかを示しますが、どこまで確認するかはモデルではなくコスト構造が決定します。

### 6.5 運用上の解釈

| 質問 | 結果 | 運用上の解釈 |
|---|---|---|
| リスクスコア上位区間にデフォルトが集中しているか | Top 10%が全デフォルトの37.30%を捕捉 | レビュー優先順位スコアとして利用可能 |
| 上位区間のPrecisionは高いか | Observed Top 10% Precision 96.59% | モデリング標本内ではリスク集中度が高い |
| 20x weighted正常顧客負担はどの程度か | Top 20%で212,140人 | 広いレビュー方針は運用負担が大きい |
| 最良のthresholdは固定されているか | コストシナリオによりTop 4%〜28%へ変化 | thresholdはビジネス上のコスト構造によって決まる |
| スコアを自動措置に利用できるか | 公開データ、匿名変数、EAD/LGD不在 | 自動措置よりreview prioritization用途が適切 |

したがって、最終結果は次のように解釈するのが安全です。

> 本モデルは、顧客をリスク順に並べるうえで有用である。ただし、実際のカード会社の運用では、自動措置システムではなく、リスク部門の月次レビュー優先順位とthresholdシミュレーションのためのツールとして利用することが適切である。

---

## 7. Key Design Decisions

### なぜdefault predictionではなくreview prioritizationとして定義したのか？

カード会社は、すべての顧客を毎月詳細に確認することはできません。そのため、重要な問いは「誰がデフォルトするか」ではなく、「今月誰を優先して確認するか」です。

本プロジェクトでは、予測スコアを実運用で利用できるように、Top-K review policyとthreshold simulationへ変換しました。

### なぜ顧客・月データをcustomer-level profileへ変換したのか？

元データは複数月の顧客記録で構成されていますが、予測対象は顧客単位のdefault有無です。

そのため、顧客・月データをそのまま使用するのではなく、顧客ごとの長期水準、直近変化、変動性、欠損パターンを要約したcustomer-level profileへ変換しました。

### なぜ変数の意味を直接解釈しなかったのか？

AMEXデータでは変数名が匿名化されています。特定の変数を所得、限度額、延滞などと断定すると、誤った金融上の解釈につながる可能性があります。

そのため、変数名そのものよりも、時系列パターン、欠損パターン、直近変化、複数モデルで繰り返し現れるリスクシグナルを中心に解釈しました。

### なぜ複数モデルを結合したのか？

単一モデルは、特定のfeatureや特定の学習方式に過度に依存する可能性があります。

LightGBM、XGBoost、CatBoost、Tabular MLP、LightGBM派生モデルを併用し、OOFベースで結合することで、複数モデルが共通して高く評価したリスクシグナルを最終スコアへ反映しました。

### なぜTop-K基準で評価したのか？

実際のリスク部門は、すべての顧客を確認できず、リスク上位区間から優先的にレビューします。

そのため、全体AUCやAccuracyよりも、Top 1%、5%、10%、20%区間におけるPrecision、Capture Rate、Lift、正常顧客のレビュー負担の方が、運用に近い評価基準です。

### なぜnon-default workloadを20倍に加重したのか？

AMEX公開データは、実際のカードポートフォリオにおけるdefault/non-default比率をそのまま反映していない可能性があります。

そこで、AMEX competition sampling-adjusted scenarioとして、非デフォルト顧客を20倍に加重したシナリオを別途計算し、レビュー範囲を広げたときに発生し得るworkloadをより保守的に確認しました。この値は、実際のカード母集団に対する補正値やpopulation calibrationとして解釈するものではありません。

### なぜコスト感応度分析を追加したのか？

リスクスコアは顧客の順位を作りますが、レビュー範囲はビジネス上のコスト構造によって決まります。

EAD、LGD、介入効果、レビューコスト、顧客負担コストに応じて、適切なthresholdは変化します。そのため、Base、Conservative、Aggressiveのシナリオに分け、レビュー範囲がどのように変わるかを確認しました。

---

## 8. Development Notes

本プロジェクトは、当初default predictionのモデリング問題のように見えました。

しかし、分析を進める中で、中心となる課題は「最も性能の高いモデルを1つ作ること」ではなく、そのモデルスコアをリスク部門が利用できる意思決定構造へ変換することだと明確になりました。

第1の転換点はfeature設計でした。変数名が匿名化されているため、個々の変数の意味を金融上の意味として断定せず、summary、temporal、recent、missingness、pivot-liteのfeature blockを用いて顧客行動パターンを要約しました。

第2の転換点は評価基準でした。単純なAUCやAccuracyよりも、リスク部門が実際に利用するTop-K review policyの方が重要でした。そのため、リスクスコア上位1%、5%、10%、20%におけるdefault concentration、capture rate、lift、false positive loadを比較しました。

第3の転換点は、AMEX competition sampling-adjusted scenarioでした。Observed Precisionだけを見ると上位区間の性能は非常に高く見えますが、実運用では正常顧客数がはるかに多い可能性があります。そのため、非デフォルト顧客を20倍に加重したシナリオを別途計算し、thresholdをより保守的に解釈しました。

第4の転換点はcost scenarioでした。Top 20%はより多くのdefaultを捕捉しますが、正常顧客のレビュー負担が大きくなるため、基準シナリオではTop 17%よりもシミュレーション純便益が低くなりました。この結果から、「より多く捕捉する方針」が常に良い方針とは限らないことを確認しました。

最終的に、本プロジェクトは次のメッセージに整理されました。

> 優れたdefault modelは、確率を当てるだけでは不十分である。カード会社では、そのスコアを限られたreview capacityの中で誰を優先して確認するかを決めるrisk decisioning構造へ変換できなければならない。

---

## 9. Limitations

本プロジェクトはAmerican Expressの公開データを用いたポートフォリオ向けプロトタイプであり、実際のカード会社の運用方針として直接解釈するには限界があります。

1. 公開データの標本分布は、実際のカードポートフォリオの分布と異なる可能性があります。そのため、観測precisionやdefault rateを実際の母集団確率として直接解釈してはいけません。
2. 変数名が匿名化されているため、個々の変数の金融上の意味を直接解釈することは困難です。Feature importanceを直ちに金融政策上の根拠として使用することは適切ではありません。
3. 顧客別EAD、実際のLGD、回収率、介入効果、レビューコストに関する情報は含まれていません。そのため、threshold分析は実際の損益推定ではなく、仮定に応じてレビュー範囲がどのように変わり得るかを確認するための感応度分析です。
4. モデル介入の因果効果は検証していません。高リスク顧客をレビューしたときに実際のdefaultがどの程度減少するかは、別途実験またはshadow modeでの運用により確認する必要があります。
5. 実際のカード会社の運用では、規制、説明可能性、公平性、顧客通知、adverse actionに関する検討が必要です。本プロジェクトは自動措置システムではなく、review prioritization prototypeとして解釈すべきです。
6. 公開リポジトリには、大容量の元データ、全feature parquet、全OOF予測値、学習済みモデルを含めていません。完全な再学習や追加ablationには、別途データへのアクセスが必要です。

---

## 10. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify the public repository

```bash
python -m compileall src tests
python -m pytest tests -q -p no:cacheprovider
```

### Review the results

公開結果表は`outputs/tables/`にあります。

主要結果の出所は`docs/results_provenance.md`に記録しています。元のColab実験と照合されていない値を、公開性能値として主張しません。

### Reproduce or extend the experiments

完全な再学習には、AMEX competitionの元データ、integer parquet形式のデータ、十分な5-foldモデル学習リソースが必要です。詳細な入力構造と再現方法は`docs/reproduction_guide.md`を参照してください。

基本集約feature、temporal feature、recent-window feature、pivot-lite featureの追加ablation、およびOOFモデル間の相関・leave-one-out blend診断は、以下のColabノートブックで実行できます。

```text
notebooks/03_colab_feature_ablation_and_oof_diagnostics.ipynb
```

このノートブックは、元のOOFファイルとfeature parquetが保存された個人のColab／Drive環境で実行するためのものです。これらのファイルが公開リポジトリに存在しない場合、診断結果を計算することはできません。

---

## 11. Project Structure

```text
amex-credit-risk-decisioning/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- catboost_full.yaml
|   |-- lgbm_full.yaml
|   |-- mlp.yaml
|   |-- policy_simulation.yaml
|   `-- xgb_full.yaml
|-- data/
|   |-- README.md
|   `-- sample/
|       `-- synthetic_scores.csv
|-- docs/
|   |-- ablation_and_oof_diagnostics.md
|   |-- experiment_log.md
|   |-- governance_and_limitations.md
|   |-- reproduction_guide.md
|   `-- results_provenance.md
|-- notebooks/
|   |-- 01_model_development_summary.ipynb
|   |-- 02_risk_ranking_and_policy_analysis.ipynb
|   `-- 03_colab_feature_ablation_and_oof_diagnostics.ipynb
|-- outputs/
|   `-- tables/
|       |-- blend_comparison.csv
|       |-- model_cv_summary.csv
|       |-- risk_decile_summary.csv
|       |-- top17_base_cost_scenario.csv
|       |-- topk_policy_tradeoff.csv
|       `-- weighted_policy_tradeoff.csv
|-- src/
|   `-- amex_risk/
|       |-- data/
|       |-- evaluation/
|       `-- modeling/
`-- tests/
```

本リポジトリには`app.py`、`scripts/`、`decision_mart/`、ダッシュボードアプリは含まれていません。現在の公開範囲は、検証可能なモデリング・評価モジュール、設定値、ドキュメント、小規模な集計結果表です。

---

## 12. What This Project Demonstrates

本プロジェクトは、クレジットカードの債務不履行予測スコアを、実際のリスク運用上の意思決定へ変換する過程を示しています。

1. 顧客・月単位の匿名データをcustomer-level risk profileへ変換し、summary、temporal、recent、missingness、pivot-liteのfeature blockを設計しました。
2. LightGBM、XGBoost、CatBoost、Tabular MLPなど複数モデルを用い、単一モデルへの依存を抑えました。
3. OOFベースのblendingにより、最終的なリスク順位スコアを作成しました。
4. モデルスコアを絶対確率ではなく、限られたレビューリソースの中で利用するranking-oriented scoreとして解釈しました。
5. Top-K policy simulationにより、レビュー区間ごとのPrecision、Capture Rate、Lift、正常顧客のレビュー負担を比較しました。
6. Observed Precisionと20x weighted scenario Precisionを分離しました。
7. EAD、LGD、介入効果、レビューコスト、顧客負担コストを仮定したcost-sensitive threshold analysisを実施しました。
8. 元ノートブックで確認されていないモデル、数値、feature、結果を公開結果として主張しないよう、provenanceを記録しました。

本プロジェクトの中心は、単にdefault predictionモデルを作成したことではなく、モデルスコアをカード会社のリスク部門が利用できるレビュー優先順位とthreshold意思決定構造へ変換したことです。
