# Data Contract

Only the synthetic 50-case assets are canonical for submission.

Required runtime inputs:

- `data/fixtures/offline_payment_ops_synthetic_50.sqlite3`
- `data/golden/offline_payment_ops_cases_50.json`
- `data/policies/offline_payment_ops/*.md`
- `data/tool_catalog/offline_payment_ops_tools.json`

The SQLite fixture must include `scenarios` and `scenario_stores` so evaluators can map each `fixture_key` such as `SYN-001` to the correct `store_id` such as `STR-SYN-001`.
