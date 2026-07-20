# Recover24 Evaluation

Recover24 evaluation is split into four areas:

```text
evaluation/
├─ legacy/          # preserved Gemma narrative evaluation
├─ normalization/   # deterministic canonical + rendering checks
├─ consistency/     # rule-based statement extraction + conflict checks
├─ narrative/       # deterministic required-elements checklist
├─ cli.py           # backward-compatible legacy entrypoint
├─ dataset_loader.py
├─ reporting.py
├─ run_eval.py
├─ run_policy.py
├─ run_all.py       # runs normalization -> consistency -> narrative
└─ scoring.py
```

## Track responsibilities

- `normalization/`
  Verifies that explicit user input becomes the expected canonical values and the expected document-facing display strings.

- `consistency/`
  Extracts statement facts from `raw_statement`, compares them with form facts, and blocks document generation when key fields conflict.

- `narrative/`
  Evaluates only cases that are safe to generate, using a deterministic required-elements checklist.

- `legacy/`
  Preserved Gemma narrative evaluation from the earlier framework. Existing imports such as `evaluation.run_eval` remain available for backward compatibility.

## Commands

```powershell
python -m evaluation.normalization.runner
python -m evaluation.consistency.runner
python -m evaluation.narrative.runner
python -m evaluation.run_all
```

## Current boundaries

- No LLM, Gemma, Colab, or network calls are used in the new deterministic tracks.
- `recover24/` remains the product code path; this redesign work is scoped to `evaluation/`.
- Deterministic checks run before any future judge-based narrative scoring.
