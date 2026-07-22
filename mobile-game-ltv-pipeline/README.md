# Mobile Game LTV Pipeline

モバイルゲームのD0～D7イベントログを使用してD8～D180のLTVを予測し、その予測結果を単なるスコア提出で終わらせず、**UA予算判断、高価値ユーザーの優先順位付け、モデル検証レポート**まで接続したproduction-style ML pipelineです。

元データがevent log形式であり、`user_id`だけでは安定したモデリング単位にならないことを最初に診断した上で、**モデリングgrainの定義 → feature build → time-based validation → two-stage modeling → Optuna tuning → business report**という流れで構成しました。

---

## 1. Overview

モバイルゲームのLTV予測は、単純な回帰問題ではありません。

大半のユーザーは長期売上が0に近く、一部の高価値ユーザーが売上全体の大部分を占めます。そのため、平均誤差だけを小さくするモデルでは、実際のUA運用には十分ではありません。マーケティングチームが知りたいのは、「全ユーザーのLTVを少し正確に予測できたか」だけでなく、**上位の高価値ユーザーと高収益セグメントを正しく見つけられたか**です。

本プロジェクトでは、D0～D7の初期行動ログを基にD8～D180のLTVを予測します。

入力データはユーザーごとに1行のテーブルではなく、イベントログです。trainには約2,100万件、testには約519万件のイベントがあり、それぞれのイベントはsession、ad impression、IAPなどの行動を表します。

初期EDAの結果、`user_id`だけでgroupbyすると、同一ユーザー内にplatform、country、channel、install_day、targetが混在する問題が見つかりました。そこで単純な`user_id`集計ではなく、次のmodeling grainを使用しました。

```text
user_id + platform + country_tier + channel_tier + install_day
```

最終モデルは`optuna_two_stage_top_capture`です。

構造は次のとおりです。

```text
Stage 1: D8-D180 LTVが0より大きくなる確率を予測
Stage 2: LTVが正となるユーザーのlog1p(LTV)金額を予測
Final prediction = p_positive × predicted_ltv_if_positive
```

最終holdoutの結果は次のとおりです。

| 指標 | 結果 |
|---|---:|
| MAE | 4.7472 |
| RMSE | 76.5951 |
| RMSLE | 0.5272 |
| Spearman correlation | 0.7970 |
| Top 10% revenue capture | 78.77% |
| Positive LTV rate in predicted top decile | 98.32% |

本プロジェクトの結論は次のとおりです。

> モバイルゲームのLTV予測では、平均誤差だけでなく、  
> zero-inflated target、long-tail revenue、time-based validation、top-decile capture、UA意思決定を併せて評価する必要がある。

---

## 2. Problem & Objective

モバイルゲームのUAでは、インストール直後の短期間の行動だけから長期価値を推定する必要があります。

D0～D7の行動は既に観測できますが、D8～D180のLTVは時間が経過しなければ分かりません。マーケティングチームが長期LTVを待ってから予算を調整していては遅いため、早い段階で、どのユーザーとセグメントが長期売上を生み出す可能性が高いかを判断する必要があります。

しかし、この問題には複数の難しさがあります。

| 課題 | 難しい理由 |
|---|---|
| Targetがzero-inflated | 多くのユーザーのD8～D180 LTVが0 |
| Long-tail revenue | 一部の高価値ユーザーが売上全体の大部分を占める |
| Event-level raw data | 元データはユーザー1行ではなくD0～D7のイベントログ |
| User grainが不安定 | `user_id`だけで集約するとcontextとtargetが衝突 |
| Train/test submission grainの不一致 | context-levelの予測をuser-levelの提出形式へ変換する必要がある |
| 実際のCPIがない | ROAS判断を実コストではなくsimulationに限定する必要がある |
| リーダーボードスコアだけでは不十分 | UA意思決定ではtop user captureとsegment判断が重要 |

したがって、本プロジェクトの目的は、単に予測モデルを一つ作ることではありません。

目的は次のとおりです。

第一に、raw event logを安定したmodeling grainへ変換します。

第二に、D0～D7の行動を基に長期LTV予測featureを作成します。

第三に、zero-heavy targetに合わせてsingle-stage modelとtwo-stage modelを比較します。

第四に、time-based validationとrolling validationを用いて、モデルが時間軸上でも安定しているかを確認します。

第五に、RMSLEだけでなくtop 10% revenue captureも併せて評価します。

第六に、最終予測をUA segment decision simulationとbusiness reportへ接続します。

---

## 3. Data

元データはD0～D7期間のevent-levelログです。

| 項目 | Train | Test |
|---|---:|---:|
| Event rows | 21,006,238 | 5,192,340 |
| Unique `user_id` | 75,464 | 31,399 |
| Rows/user median | 32 | 9 |
| Rows/user p95 | 1,166 | 860 |
| Rows/user max | 178,821 | 96,075 |

イベントタイプの分布は次のとおりです。

| Event type | Train rows | Train share | Test rows | Test share |
|---|---:|---:|---:|---:|
| `ad_impression` | 18,590,666 | 88.50% | 4,618,468 | 88.95% |
| `session` | 2,359,512 | 11.23% | 558,644 | 10.76% |
| `iap` | 56,060 | 0.27% | 15,228 | 0.29% |

Targetは`ltv_d8_d180`です。

モデル入力検証時点のtarget分布は次のとおりです。

| 項目 | 値 |
|---|---:|
| Rows | 159,521 |
| Positive LTV rate | 40.45% |
| Zero LTV rate | 59.55% |
| Mean | 18.9218 |
| P50 | 0.0000 |
| P75 | 1.2754 |
| P95 | 20.2234 |
| P99 | 279.9575 |
| Max | 62,046.4734 |

EDAで最も重要だった発見は、`user_id`だけでは安定したモデリング単位にならないことでした。

| Grain | Train groups | Target collision groups | Collision rate |
|---|---:|---:|---:|
| `user_id` | 75,464 | 34,191 | 45.31% |
| `user_id + install_day` | 146,129 | 6,953 | 4.76% |
| `user_id + platform + country_tier + channel_tier + install_day` | 159,926 | 405 | 0.25% |

`user_id`基準で集約すると、target collision groupが34,191件、collision rateが45.31％でした。つまり、同一の`user_id`内に異なるcontextやtargetが混在する可能性がありました。

そこで最終modeling grainを次のように固定しました。

```text
user_id + platform + country_tier + channel_tier + install_day
```

残った405件のtarget-collision groupはsupervised trainingから除外し、`dropped_collision_groups.csv`へ記録しました。

---

## 4. Method / System Design

本プロジェクトの設計原則は次のとおりです。

> 最初にデータ単位を検証し、  
> 次にfeatureを作成し、  
> 最後にモデルを選択する。

pipeline全体は次のとおりです。

```text
Raw competition zip
   ↓
Raw data validation
   ↓
Modeling grain diagnostics
   ↓
Feature build
   ↓
Model input validation
   ↓
Baseline / Linear / XGBoost experiments
   ↓
Two-stage model experiment
   ↓
Rolling time validation
   ↓
Optuna tuning
   ↓
Final two-stage refit
   ↓
Prediction artifact generation
   ↓
Business analysis report
   ↓
Model card
```

### 4.1 Modeling Grain

当初は`user_id`単位のLTV予測問題に見えました。

しかしEDAの結果、`user_id`内でplatform、country、channel、install_dayが変わるケースが多く見つかりました。そのため、`user_id`だけでfeatureを集計すると、異なるインストールcontextが混在し、labelも衝突する可能性があります。

最終grainは次のとおりです。

```text
user_id + platform + country_tier + channel_tier + install_day
```

このgrainにより、target collision rateを45.31％から0.25％へ低下させました。

### 4.2 Feature Engineering

D0～D7のイベントログをgrain単位のfeatureへ変換しました。

Featureは大きく次のグループに分けられます。

| Feature group | 例 | 意味 |
|---|---|---|
| Activity | `event_count`, `session_count`, `active_days`, `last_event_day` | 初期活動性 |
| Ad behavior | `ad_impression_count`, `ads_per_session`, `ad_revenue_per_ad` | 広告消費と広告収益性 |
| Early revenue | `revenue_d0_d7`, `ad_revenue_d0_d7`, `iap_revenue_d0_d7` | 初期売上シグナル |
| Time bucket | `revenue_d0`, `revenue_d1`, `revenue_d2_d3`, `revenue_d4_d7` | 初期・後期の売上変化 |
| IAP behavior | `iap_count`, `avg_iap_amount`, `max_iap_amount`, `unique_product_count` | 課金行動 |
| Context | `platform`, `country_tier`, `channel_tier`, `install_day` | 流入環境 |
| Target encoding | `te_platform_country_channel_ltv_log_mean`など | segment-level prior |

Feature buildの結果は次のとおりです。

| 項目 | 結果 |
|---|---:|
| Train feature rows | 159,521 |
| Test feature rows | 40,105 |
| Dropped target-collision groups | 405 |
| Train feature columns | 40 |
| Test feature columns | 39 |

### 4.3 Two-stage Modeling

LTV targetには0が多く、正のLTV内でもlong-tailが大きくなっています。

そこで最終モデルを2段階に分けました。

| Stage | 役割 |
|---|---|
| Stage 1 classifier | D8～D180 LTVが0より大きくなる確率を予測 |
| Stage 2 regressor | 正のLTVを持つユーザーの`log1p(LTV)`金額を予測 |
| Final prediction | `p_positive × predicted_ltv_if_positive` |

この構造の利点は、zero-heavy targetを分離して扱える点です。

単一の回帰モデルは、LTVが0のユーザーと高価値ユーザーを同時に当てようとするため、予測が平均へ押しつぶされる可能性があります。Two-stage構造では、「正のLTVになる可能性」と「正である場合にどの程度大きくなるか」を分けてモデル化します。

### 4.4 Time-based Validation

本プロジェクトでは、random KFoldを中心とした評価は行いませんでした。

モバイルゲームのLTV予測にはinstall_dayに基づく時間の流れがあるため、将来のユーザーを予測する状況に近いvalidationを構成する必要があります。

Primary holdout splitは次のとおりです。

```text
Train: install_day 0-23
Valid: install_day 24-30
```

さらに、expanding rolling validationを実施しました。

| Fold | Train days | Valid days | Train rows | Valid rows |
|---:|---|---|---:|---:|
| 1 | 0-13 | 14-16 | 81,185 | 14,186 |
| 2 | 0-16 | 17-19 | 95,371 | 13,825 |
| 3 | 0-19 | 20-23 | 109,196 | 18,254 |
| 4 | 0-23 | 24-30 | 127,450 | 32,071 |

Target encodingは各foldのtrain rowのみでfitし、valid rowへmapしました。p1/p99 clippingとpreprocessingもfold trainを基準に作成しました。

---

## 5. Implementation

本プロジェクトは実験用Notebookではなく、`make all`で再現可能なpipelineとして構成しました。

`make all`では、最終選択モデルの実行に必要な工程のみを実行します。

```text
1. raw data validation
2. feature build
3. model input validation
4. final two-stage model train/refit
5. prediction artifact generation
6. business analysis report generation
7. model card generation
```

実験用のbaseline、linear model、feature ablation、rolling validation、Optuna tuningは最終pipelineから分離しました。これらの実験はモデル選択の根拠として保存し、必要に応じて`make experiments`で再現できます。

主なモジュールは次のとおりです。

| モジュール | 役割 | 簡単な説明 |
|---|---|---|
| `validate_raw_data.py` | raw train/test event logの検証 | 入力データの異常を確認 |
| `build_features.py` | modeling grain featureの生成 | event logをモデル入力へ変換 |
| `validate_model_input.py` | train/test feature schemaの検証 | モデル入力前にnull/inf/schemaを確認 |
| `train_final_model.py` | 最終two-stage modelのrefit | 選択モデルを全trainデータで再学習 |
| `predict_submission.py` | test predictionの生成 | context-level予測をuser-levelへ集約 |
| `build_business_report.py` | business analysisの生成 | top decile、UA simulation、feature importance |
| `run_experiments.py` | モデル選択実験の再現 | baselineからOptunaまでの選択実験を実行 |
| `common/metrics.py` | 評価指標 | RMSLE、Spearman、top captureなど |
| `common/preprocessing.py` | 前処理 | clipping、categorical encodingなど |
| `common/target_encoding.py` | target encoding | leakage防止型segment encoding |

モデル入力検証の結果は次のとおりです。

| 項目 | 結果 |
|---|---|
| Train/test feature columns match | True |
| Target exists in train | True |
| Target exists in test | False |
| Train duplicate grain rows | 0 |
| Test duplicate grain rows | 0 |
| Post-preprocessing numeric null | 0 |
| Post-preprocessing categorical null | 0 |
| Time split possible | True |

---

## 6. Evaluation

評価は3つの層に分けました。

第一に、データとfeatureがモデル学習可能な構造になっているかを検証しました。

第二に、baselineから最終モデルまで段階的に性能を比較しました。

第三に、UA運用の観点から、上位ユーザーとセグメントをどの程度正しく特定できるかを確認しました。

---

### 6.1 Baseline → Linear → XGBoost

まず単純なbaselineとlinear modelを作成し、その後XGBoostを追加しました。

| Model | MAE | RMSE | RMSLE | Spearman | Top 10% revenue capture |
|---|---:|---:|---:|---:|---:|
| `global_mean` | 24.1797 | 104.0161 | 2.7500 | 0.0000 | 6.51% |
| `segment_mean` | 16.6481 | 105.0316 | 2.0948 | 0.1139 | 17.15% |
| `early_revenue_multiplier` | 10.2615 | 169.3231 | 0.9160 | 0.7235 | 72.06% |
| `ridge_log_linear` | 6.2176 | 97.1298 | 0.6645 | 0.7686 | 74.08% |
| `xgboost_log_target` | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 76.32% |

この結果は、初期D0～D7 revenueが強力なbaselineであることを示しています。単純なearly revenue multiplierだけでも、top 10% revenue captureは72.06％でした。

一方、XGBoostは非線形な活動性、広告露出、IAP、context featureを併せて使用することで、RMSLEとranking性能の両方を改善しました。

---

### 6.2 Feature Engineering Experiment

XGBoostを基準として、feature setを追加比較しました。

| Feature set | MAE | RMSE | RMSLE | Spearman | Top 10% revenue capture |
|---|---:|---:|---:|---:|---:|
| `xgb_current_full` | 5.1163 | 83.4851 | 0.5476 | 0.8031 | 76.32% |
| `xgb_time_bucket_features` | 5.1355 | 84.1380 | 0.5447 | 0.8025 | 76.57% |
| `xgb_velocity_ratio_features` | 5.1225 | 82.5995 | 0.5456 | 0.8032 | 76.60% |
| `xgb_frequency_interaction_features` | 5.0886 | 83.7111 | 0.5430 | 0.8031 | 76.49% |
| `xgb_target_encoding_features` | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 76.41% |

Target encoding feature setはRMSLE基準で最良でした。ただし、top 10% revenue captureだけを見ると、velocity/time bucket系も同程度に機能しました。

そのため、その後のモデル選択ではRMSLEとtop captureを併せて確認しました。

---

### 6.3 Two-stage Model

Zero-heavy targetを反映するため、two-stage XGBoostを実験しました。

| Model | MAE | RMSE | RMSLE | Spearman | Top 10% revenue capture | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_target_encoding_features` | 5.0878 | 83.7934 | 0.5404 | 0.8023 | 76.41% | 7.64 |
| `two_stage_xgb_target_encoding_features` | 5.0390 | 80.2077 | 0.5447 | 0.8051 | 77.66% | 7.76 |

Two-stage modelは、best single-stageのRMSLEよりわずかに悪かった一方、top 10% revenue captureは高くなりました。

つまり、平均的なlog errorではsingle-stageが有利であり、高価値ユーザーを上位に配置するrankingの観点ではtwo-stageが有利でした。

本プロジェクトではUA意思決定の観点からtop-decile captureを重視したため、two-stage系を最終候補として維持しました。

---

### 6.4 Rolling Time Validation

Rolling validationでは、RMSLEの安定性とtop-decile captureの安定性を分けて評価しました。

| Model | RMSLE mean | RMSLE std | Top 10% capture mean | Top 10% capture std | Spearman mean |
|---|---:|---:|---:|---:|---:|
| `single_stage_xgb_target_encoding_features` | 0.5349 | 0.0157 | 78.76% | 2.32% | 0.7961 |
| `single_stage_xgb_velocity_ratio_features` | 0.5400 | 0.0169 | 78.52% | 2.10% | 0.7955 |
| `two_stage_xgb_target_encoding_features` | 0.5409 | 0.0120 | 79.77% | 2.62% | 0.7979 |
| `two_stage_xgb_velocity_ratio_features` | 0.5438 | 0.0132 | 79.16% | 2.06% | 0.7971 |

解釈は明確です。

| 観点 | 最良候補 |
|---|---|
| RMSLE mean | `single_stage_xgb_target_encoding_features` |
| Top 10% capture mean | `two_stage_xgb_target_encoding_features` |

つまり、最終モデルの選択は、単純にRMSLEが最も良いモデルを選ぶ問題ではありませんでした。

UA運用では上位ユーザーの捕捉が重要であるため、top capture objectiveを別途考慮しました。

---

### 6.5 Optuna Tuning and Final Model

Optunaはrolling validationで絞り込んだ候補のみを対象に実施しました。

| 項目 | 設定 |
|---|---|
| Trials per study | 30 |
| Early stopping rounds | 50 |
| Tuning folds | install_day 0-13 → 14-16, 0-16 → 17-19, 0-19 → 20-23 |
| Final holdout | install_day 0-23 → 24-30 |
| Feature set | target encoding features |
| Leakage control | target encodingはfold trainのみでfit |

最終holdoutの結果は次のとおりです。

| Model | Objective | MAE | RMSE | RMSLE | Spearman | Top 10% revenue capture | Top-decile lift |
|---|---|---:|---:|---:|---:|---:|---:|
| `optuna_single_stage_rmsle` | RMSLE | 4.9102 | 80.5268 | 0.5307 | 0.8004 | 76.93% | 7.69 |
| `optuna_two_stage_top_capture` | Top capture | 4.7472 | 76.5951 | 0.5272 | 0.7970 | 78.77% | 7.88 |

最終選択モデルは`optuna_two_stage_top_capture`です。

選択理由は次のとおりです。

| 基準 | 解釈 |
|---|---|
| MAE/RMSE | tuned single-stage modelより低い |
| RMSLE | 0.5272で、tuned single-stage modelよりも低い |
| Top 10% revenue capture | 78.77％で最も高い |
| Top-decile lift | 7.88で最も高い |
| Business fit | UA運用における高価値ユーザーの優先順位付けに適合 |

---

### 6.6 Top-Decile Business Analysis

最終モデルが予測した上位10％のユーザーは、validation rowの10.00％に当たる3,208人です。

この上位10％が、実際のD8～D180 revenueの78.77％を捕捉しました。

上位10％とその他のユーザーの行動差は次のとおりです。

| Feature | Top decile mean | Non-top mean | Ratio |
|---|---:|---:|---:|
| `iap_revenue_d0_d7` | 15.3444 | 0.2520 | 60.90 |
| `revenue_d0_d7` | 17.5543 | 0.5012 | 35.02 |
| `revenue_per_active_day` | 4.3845 | 0.1728 | 25.37 |
| `early_payer_flag` | 0.1428 | 0.0158 | 9.04 |
| `ad_revenue_d0_d7` | 2.2098 | 0.2492 | 8.87 |
| `ad_impression_count` | 574.6328 | 73.4238 | 7.83 |
| `event_count` | 586.4177 | 90.5853 | 6.47 |
| `active_days` | 6.8974 | 3.1880 | 2.16 |

Top-decileのユーザーは、単にイベント数が多いユーザーではありません。

初期IAP revenue、初期総revenue、revenue per active day、early payerの有無に非常に大きな差が見られました。つまりモデルは、「活動量が多いユーザー」だけでなく、「初期monetization qualityが高いユーザー」を上位に配置しています。

---

### 6.7 UA Decision Simulation

最終予測値をcountry/channel segment単位で集約し、synthetic CPIを付与してpredicted ROASに基づく意思決定simulationを作成しました。

重要なのは、synthetic CPIは実際の広告費ではないという点です。このtableは実際の予算執行結果ではなく、LTV予測値をUA decision workflowへ接続する例です。

メインtableでは、`min_users=100`以上のsegmentにのみdecisionを付与しました。

| Country | Channel | Users | Predicted LTV | Actual LTV | Synthetic CPI | Predicted ROAS | Actual ROAS | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NL | `bb16a88d` | 141 | 38.9756 | 61.6187 | 0.8075 | 48.27 | 76.31 | `scale_up` |
| ES | `92247aa9` | 108 | 22.7146 | 77.0867 | 0.8075 | 28.13 | 95.46 | `scale_up` |
| TR | `bb16a88d` | 115 | 18.5414 | 53.4072 | 0.8075 | 22.96 | 66.14 | `scale_up` |
| IT | `bb16a88d` | 128 | 12.4819 | 18.1852 | 0.8075 | 15.46 | 22.52 | `scale_up` |
| PL | `92247aa9` | 301 | 11.2661 | 12.3424 | 0.8075 | 13.95 | 15.28 | `scale_up` |

標本数が小さいsegmentは`insufficient_sample`として分離しました。

例えば、`OTHER + 0a0ae9c4`はpredicted ROASが81.10と非常に高く見えますが、usersが71人しかいないため、scale decisionを付与しませんでした。

この点が重要です。

> 予測ROASが高く見えても、標本数が小さければ直ちにscale-upすべきではない。  
> LTV model outputをbudget decisionへ変換する前に、sample-size guardrailを通す必要がある。

---

## 7. Key Design Decisions

### 7.1 `user_id`ではなくuser-context grainを使用した

当初はユーザー別LTV予測であるため、`user_id`で集約すればよいように見えました。

しかしEDAの結果、`user_id`基準のtarget collision rateは45.31％でした。同一`user_id`内に異なるplatform、country、channel、install_dayが混在し、targetも一貫していませんでした。

そこで、`user_id + platform + country_tier + channel_tier + install_day`をmodeling grainとして選択しました。この判断がなければ、その後のモデル性能が良く見えても、label定義自体が不安定だった可能性があります。

### 7.2 残ったtarget collision groupを平均化せず除外した

Composite grainを使用しても405件のcollision groupが残りました。

target平均で処理することも可能でしたが、その場合、identity問題がモデル学習内に隠れてしまいます。そこで初回のsupervised trainingでは該当groupを除外し、別途logへ記録しました。

### 7.3 Random KFoldではなくtime-based validationを使用した

LTV予測は、将来にインストールするユーザーを予測する問題です。

そのため、random splitで過去と未来を混在させると、実際の運用状況より楽観的な性能が出る可能性があります。そこでinstall_day基準のholdoutとrolling validationを使用しました。

### 7.4 RMSLEとtop-decile captureを併せて評価した

RMSLEは予測全体の誤差を評価する上で有用ですが、UA運用では高価値ユーザーを正しく見つけられるかも重要です。

そこでRMSLE、MAE、RMSE、Spearmanに加え、top 10% revenue captureとtop-decile liftも併せて評価しました。

### 7.5 Two-stage modelを最終候補として維持した

Zero-heavy targetでは、「LTVが0か否か」と「正であればどの程度大きいか」は異なる問題です。

Two-stage modelはこの2つを分けてモデル化します。最終的なOptuna結果でtwo-stage modelはRMSLEとtop captureの双方で強い結果を示したため、最終モデルとして選択しました。

### 7.6 UA simulationにsample-size guardrailを設定した

Segment別のpredicted ROASが高くても、標本数が小さすぎる場合、実際の予算判断には使用しにくくなります。

そこで`min_users=100`という基準を設け、小規模segmentは`insufficient_sample`として分離しました。

### 7.7 最終pipelineと実験pipelineを分離した

`make all`は、最終的に再現可能なpipelineのみを実行します。

baseline、feature ablation、rolling validation、Optunaは`make experiments`で別途再現できるよう分離しました。これにより、最終実行は高速かつ明確になり、実験根拠は別レポートとして維持できます。

---

## 8. Development Notes

本プロジェクトは当初、一般的なKaggleのLTV回帰問題に見えました。

しかしEDAを進める中で、最も重要な問題はモデルアルゴリズムではなく、**何をモデリング単位とするか**であることが分かりました。`user_id`だけで集約するとtarget collisionが大きく、この状態でモデルを学習しても、性能値を解釈することが困難でした。

第一の転換点はmodeling grainの決定でした。`user_id + platform + country_tier + channel_tier + install_day`を使用することで、target collision rateを45.31％から0.25％へ低下させました。

第二の転換点は評価指標でした。early revenue multiplier baselineが既にtop 10% revenue capture 72.06％を記録していました。つまり、単純なモデルにも強いranking signalがありました。そのため、モデル比較をRMSLEだけで行うのは不十分だと判断しました。

第三の転換点はtwo-stage modelingでした。LTV targetは0が多く、long-tailも強くなっています。そこで正のLTVになるかどうかと、正の場合の金額を分けてモデル化し、最終的にtop capture objectiveと高い適合性を示しました。

第四の転換点はbusiness reportでした。モデル結果をsubmission scoreで終わらせず、predicted top decile profileとUA decision simulationまで接続しました。特に、synthetic CPIが実際の広告費ではなく、workflow demonstrationであることを明示しました。

最終的に本プロジェクトは、「モデルを一つ学習したNotebook」ではなく、**データ契約、モデリング単位の検証、モデル選択の根拠、運用レポートを含むML pipeline**として整理されました。

---

## 9. Limitations

本プロジェクトはポートフォリオ向けのproduction-style pipelineであり、実際のゲームUA運用へ直接適用するには、追加データと検証が必要です。

第一に、実際のCPI、キャンペーン予算、creative、bid、国別media costがありません。そのため、UA decision simulationは実際の予算推薦ではなく、workflowの例です。

第二に、test labelがないため、hidden testのMAE/RMSE/RMSLEは算出できません。Test prediction reportは、row count、non-null、non-negative、fallback countなどの提出検証が中心です。

第三に、modeling grainは最も防御的な選択ですが、完全ではありません。残った405件のtarget collision groupは除外しており、実運用ではidentity stitchingやattribution基準の確認が必要です。

第四に、最終モデルはD0～D7の観測データのみを使用します。実際の運用では、D1/D3/D7などの時点ごとに異なるearly prediction modelを運用することも考えられます。

第五に、現在のモデルはXGBoostが中心です。より大規模な運用環境では、calibration、drift monitoring、retraining schedule、online/offline feature consistencyが必要です。

第六に、top-decile captureはUA rankingに有用ですが、budget allocationは実際のCPI、marginal ROAS、campaign saturation、creative fatigueと併せて判断する必要があります。

---

## 10. How to Run

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Raw competition dataはgitに含めていません。

デフォルトのzipパスは次のとおりです。

```text
%USERPROFILE%/Downloads/mobile-game-ltv-forecasting-challenge.zip
```

別のパスを使用する場合は`ZIP_PATH`を指定します。

```bash
make all ZIP_PATH=/path/to/mobile-game-ltv-forecasting-challenge.zip
```

### Run final pipeline

```bash
make all
```

`make all`は最終two-stage LTV pipelineのみを実行します。

```text
validate → features → train → predict → business
```

### Run individual steps

```bash
make validate
make features
make train
make predict
make business
make test
make experiments
make clean
```

| Command | 役割 |
|---|---|
| `make validate` | raw train/test event logの検証 |
| `make features` | modeling grain featureとmodel inputの生成 |
| `make train` | 最終two-stage modelのrefit |
| `make predict` | test predictionとsubmissionの生成 |
| `make business` | model cardとbusiness analysisの生成 |
| `make test` | unit testの実行 |
| `make experiments` | baseline、linear、XGBoost、rolling、Optuna実験の再現 |
| `make clean` | 最終pipeline成果物の整理 |

---

## 11. Project Structure

```text
mobile-game-ltv-pipeline/
├── README.md
├── Makefile
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── train_model_input.parquet
│   │   ├── test_model_input.parquet
│   │   ├── final_model_metrics.csv
│   │   ├── final_model_params.json
│   │   ├── final_holdout_predictions.parquet
│   │   ├── top_decile_analysis.csv
│   │   ├── ua_decision_simulation.csv
│   │   └── model_input_validation.json
│   └── experiments/
│       ├── baseline_metrics.csv
│       ├── linear_model_metrics.csv
│       ├── xgboost_model_metrics.csv
│       ├── two_stage_metrics.csv
│       ├── rolling_validation_metrics.csv
│       ├── optuna_best_metrics.csv
│       ├── submission.csv
│       └── test_predictions.csv
├── models/
│   ├── final_two_stage_stage1.joblib
│   ├── final_two_stage_stage2.joblib
│   └── final_preprocessor.joblib
├── reports/
│   ├── final_model_card.md
│   ├── business_analysis.md
│   ├── model_selection_summary.md
│   ├── diagnostics/
│   ├── eda/
│   └── experiments/
├── scripts/
│   ├── eda_profile.py
│   └── grain_diagnostics.py
├── src/
│   ├── common/
│   ├── experiments/
│   └── pipeline/
└── tests/
```

---

## 12. What This Project Demonstrates

本プロジェクトは、モバイルゲームLTV予測を単純なregression Notebookではなく、再現可能なML pipelineとbusiness-facing model reportとして構成した事例です。

第一に、event-level raw dataから安定したmodeling grainを先に検証しました。

第二に、`user_id`基準のtarget collision問題を発見し、user-context grainを選択することでcollision rateを45.31％から0.25％へ低下させました。

第三に、D0～D7のactivity、ad behavior、IAP、revenue timing、context、target encoding featureを生成しました。

第四に、baseline、linear model、XGBoost、two-stage XGBoost、rolling validation、Optuna tuningを段階的に比較しました。

第五に、random KFoldではなく、install_dayに基づくtime validationとrolling validationを使用しました。

第六に、zero-inflated LTV targetを反映し、positive probabilityとconditional positive valueを分けるtwo-stage modelを最終選択しました。

第七に、RMSLEだけでなく、top 10% revenue capture、top-decile lift、Spearman correlationを併せて評価しました。

第八に、最終モデルはholdoutでRMSLE 0.5272、Top 10% revenue capture 78.77％、Top-decile lift 7.88を記録しました。

第九に、predicted top-decile behaviorとUA decision simulationを作成し、モデル結果をマーケティング意思決定の形式へ変換しました。

最後に、`make all`でraw validationからfeature build、model refit、prediction、business reportまで再現可能なpipelineを構成しました。

本プロジェクトの中心は、単にLTV予測モデルを作成したことではなく、**データgrainの検証、モデル選択根拠、ranking-oriented evaluation、UA意思決定への接続まで含むproduction-style LTV pipelineを設計したこと**です。
