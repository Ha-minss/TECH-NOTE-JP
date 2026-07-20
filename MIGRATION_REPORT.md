# TECH-NOTE-JP Migration Report

## Source Lock

- Repository: `https://github.com/Ha-minss/TECH-NOTE`
- Commit: `dd92cb77c0ea6032515d09ce4244fab886a640b3`
- Original repository is never modified by this package.

## Structural Fixes

1. `amex-credit-risk-decisioning-clean/` を `amex-credit-risk-decisioning/` へ変更
2. `storeops-triage-agent_old/` を除外
3. StoreOpsのlegacy code/docs/tests/old evaluation reportsを除外
4. `recover24/Causal_Inference_SECA/` の重複コピーを除外
5. Root READMEを4カテゴリへ再編
6. Gamelyticsの個人PC絶対パスを `<PATH_TO_DATA>` へ変更
7. GamelyticsのCSV出力を `reports/tables/` へ統一
8. 重複していた `reports/*.csv` を除外

## Translation Policy

### Japanese-localized

- Root README
- 12 project READMEs
- Gamelytics data instructions
- Public-facing project summaries, decisions, limitations, reproduction guidance

### Kept stable

- Python identifiers
- Function/class names
- SQL and YAML/JSON keys
- Model and metric names
- Golden/evaluation utterances whose translation would change test semantics
- Numeric outputs and derived CSV values
