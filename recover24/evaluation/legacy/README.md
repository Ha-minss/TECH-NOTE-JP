# Recover24 LLM Evaluation

Purpose:

> Show that a hybrid workflow is more expressive than a fixed template for complex fraud narratives and safer than unconstrained LLM generation.

The evaluation separates structured form facts, narrative generation, and fact validation.

## Methods

| Method | Description |
|---|---|
| `template` | deterministic safe baseline |
| `gemma` | Gemma generation without blocking |
| `gemma_validator` | Gemma generation plus safety validation |
| `gemma_validator_fallback` | Gemma + validation, fallback to template if unsafe |

## Three-stage workflow

```powershell
$env:RECOVER24_GEMMA_COLAB_URL="https://your-tunnel.trycloudflare.com"

# 1. Prompt development. Repeated runs are allowed.
python -m evaluation.run_eval --mode dev --provider gemma --claim-provider gemma

# 2. Validator/fallback inspection on conflicts and difficult cases.
python -m evaluation.run_eval --mode challenge --provider gemma --claim-provider gemma

# 3. Stop changing prompts and execute the held-out test split once.
python -m evaluation.run_eval --mode final --provider gemma --claim-provider gemma
```

Final mode writes:

```text
evaluation/final_submission/
├─ portfolio_report.md
├─ eval_summary.json
├─ eval_details.jsonl
├─ failure_cases.md
├─ run_manifest.json
└─ FINAL_EVAL_LOCK.json
```

After `FINAL_EVAL_LOCK.json` is created, do not tune prompts from test failures. An intentional replacement requires `--force-final` and should be documented.

## Dry runs

```powershell
python -m evaluation.run_eval --mode dev --provider none --claim-provider none
```

Provider-free methods are labelled `dry_run_*`. The report is marked as non-final, and status/unsupported/order metrics that were not measured appear as `n/a`.

## Metrics

- important fact inclusion
- amount preservation
- status contradiction rate
- unsupported claim case rate
- event order accuracy
- safe output rate
- fallback rate
- latency and LLM call count

There is no case-specific `must_not_claim` list. Status contradictions are derived from normalized structured facts and claims extracted from generated output.