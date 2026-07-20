# Recover24 evaluation dataset

This dataset evaluates narrative generation and safety validation, not full-form extraction.

Splits:

- `dev.jsonl`: prompt development; results may be inspected repeatedly.
- `challenge.jsonl`: ambiguous, missing, risky, and deliberately conflicting inputs for validator/fallback development.
- `test.jsonl`: held-out final cases; do not use while tuning prompts.

Rows contain verified `structured_facts`, the source `raw_statement`, `required_fact_ids`, coarse `expected_event_order`, and deterministic `fact_aliases`. They do not contain case-specific banned sentences.

`input_conflicts` declares synthetic disagreement between structured facts and the statement. Current conflict types include:

- `amount_mismatch`
- `police_status_mismatch`
- `freeze_status_mismatch`

The final split contains six cases. Report it as a portfolio engineering evaluation, not statistically conclusive model research.