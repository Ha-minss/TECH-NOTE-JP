"""Run normalization evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import summarize_results
from .normalizer import normalize_case
from .renderer_check import render_expected_fields


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        result = normalize_case(case["input"], required_fields=case.get("required_fields", []))
        rendered = render_expected_fields(result["canonical"])
        expected_rendered = case.get("expected_rendered", {})
        rows.append(
            {
                "case_id": case["case_id"],
                "result": result,
                "rendered": rendered,
                "canonical_match": result["canonical"] == case["expected_canonical"],
                "rendered_match": all(rendered.get(key) == value for key, value in expected_rendered.items()),
            }
        )

    summary = summarize_results(rows)
    summary["cases"] = len(rows)
    summary["rows"] = rows
    return summary


def main() -> None:
    dataset_path = Path(__file__).with_name("dataset.jsonl")
    print(json.dumps(run(load_cases(dataset_path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
