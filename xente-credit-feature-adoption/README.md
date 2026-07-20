# Xente Credit Feature Adoption

返済履歴がない顧客群において、取引行動が既存信用情報に追加的な予測価値を持つかを検証した特徴量採用プロジェクトです。

## 1. Overview

分析の目的は「取引履歴がない顧客は危険か」を示すことではなく、その差が顧客固有の信用リスクなのか、商品・供給者・サービス導線の構造を表す代理変数なのかを確認することです。

## 2. Problem

取引履歴なし群の未返済率は高く見えますが、特定のRetailerやLoan Providerを除くと差が大きく縮小します。表面的な相関をそのまま信用スコアに採用すると、環境依存のシグナルを学習する可能性があります。

## 3. Decision Question

> 取引行動特徴量は、既存情報を超える独立したリスクシグナルとして採用できるか。

## 4. Validation Design

- 顧客単位のStratified Group K-Fold
- Logistic RegressionとLightGBMの比較
- 既存特徴量のみ / 取引特徴量追加 / 特徴量除外の比較
- Permutation Test
- セグメント・Provider・Retailer別の安定性確認

## 5. Key Observation

返済履歴がない顧客のうち取引履歴もない群では未返済率が23.6%でしたが、特定のRetailer・Provider構造を除くと2.4–2.9%まで低下しました。

## 6. Interpretation

取引履歴の有無は強い相関を示しても、独立した信用リスク要因とは限りません。サービス利用経路や供給構造を表す可能性があるため、主要特徴量としての即時採用を見送りました。

## 7. Decision

主要スコア特徴量ではなく、特定セグメントを観察する補助変数として扱い、期間外データで安定性を再検証する方針としました。

## 8. Reproduce

```bash
pip install -r requirements.txt
```

プロジェクト内のNotebookを順に実行し、顧客単位分割と特徴量群比較を再現します。

## 9. Repository Structure

```text
notebooks/              # EDA, validation, adoption decision
outputs/                # Derived tables and figures
data/README.md          # Data provenance and exclusions
```

## 10. Limitations

- データ期間・商品構成が限定されています。
- 観測されない審査ルールや顧客選択の影響を完全には除去できません。
- 時間外・地域外での検証が必要です。

## 11. Key Technologies

Python, LightGBM, Logistic Regression, Group K-Fold, Permutation Test, Feature Ablation

## 12. What This Project Demonstrates

予測力が見える特徴量を無条件で採用せず、代理変数、安定性、追加価値を検証した上で採用判断を保留できることを示します。
