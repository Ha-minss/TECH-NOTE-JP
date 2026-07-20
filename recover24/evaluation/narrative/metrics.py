"""Metrics for narrative checklist evaluation."""

from __future__ import annotations


def summarize_results(rows: list[dict[str, object]]) -> dict[str, float]:
    eligible = len(rows)
    passed = sum(1 for row in rows if row["result"]["passed"])
    total_required = sum(len(row["required_elements"]) for row in rows)
    total_included = sum(len(row["result"]["included_elements"]) for row in rows)

    return {
        "checklist_accuracy": _ratio(passed, eligible),
        "element_recall": _ratio(total_included, total_required),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
