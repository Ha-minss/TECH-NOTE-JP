# Evaluation

The canonical evaluation dataset is `data/golden/offline_payment_ops_cases_50.json` backed by `data/fixtures/offline_payment_ops_synthetic_50.sqlite3`.

Run deterministic evaluation:

```powershell
$env:PYTHONPATH = "src"
python -m storeops.evals.runner
```

The current deterministic benchmark is:

```text
total_cases: 50
passed_cases: 38
state_accuracy: 0.90
cause_accuracy: 0.98
abstention_safety_accuracy: 1.00
unsupported_claim_count: 0
```

Run a live LLM smoke test:

```powershell
$env:PYTHONPATH = "src"
python -m storeops.evals.llm_runner --provider live --fixture-key SYN-001 --output-dir experiments/eval_runs/llm/live_smoke_SYN001
```

The LLM path uses the same fixture DB, policy catalog, read-only tools, evidence builder, reasoner, and safety gate as deterministic evaluation.
