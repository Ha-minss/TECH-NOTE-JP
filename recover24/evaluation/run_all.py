"""Run all deterministic evaluation tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.consistency.runner import load_cases as load_consistency_cases
from evaluation.consistency.runner import run as run_consistency
from evaluation.narrative.runner import load_cases as load_narrative_cases
from evaluation.narrative.runner import run as run_narrative
from evaluation.normalization.runner import load_cases as load_normalization_cases
from evaluation.normalization.runner import run as run_normalization


def run_all(*, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[1]
    evaluation_dir = root / "evaluation"
    return {
        "normalization": run_normalization(load_normalization_cases(evaluation_dir / "normalization" / "dataset.jsonl")),
        "consistency": run_consistency(load_consistency_cases(evaluation_dir / "consistency" / "dataset.jsonl"), evaluate_extractor=True),
        "narrative": run_narrative(load_narrative_cases(evaluation_dir / "narrative" / "dataset.jsonl")),
    }


def main() -> None:
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
