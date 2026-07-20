"""CLI entrypoint for deterministic evaluation."""

from __future__ import annotations

import json
import sys

from storeops.evals.runner import run_full_evaluation


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = run_full_evaluation()
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

