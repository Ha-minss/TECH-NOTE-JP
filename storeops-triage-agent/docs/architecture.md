# Architecture

The submission runtime is a small evidence-first workflow:

1. `storeops.evals.datasets` loads the 50-case golden set.
2. `storeops.infra.database.open_database` opens the canonical SQLite fixture.
3. `OfflinePaymentWorkflow` parses the merchant message, retrieves policy checks, maps them to allowed tool data needs, executes read-only tools, builds evidence, reasons over evidence, and applies safety gates.
4. `storeops.evals.runner` scores deterministic output against the golden set.
5. `storeops.evals.llm_runner` swaps bounded LLM components into parser/planner/clarification/response drafting while keeping the same read-only tool and safety boundary.

Legacy demos, UI code, previous outputs, and older S1-S7 fixtures are preserved in `experiments/` and are not part of the submission runtime.
