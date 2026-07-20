"""Metrics for normalization evaluation."""

from __future__ import annotations

from typing import Any


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, float]:
    case_count = len(rows)
    canonical_matches = sum(1 for row in rows if row["canonical_match"])
    rendered_matches = sum(1 for row in rows if row["rendered_match"])
    blocked_cases = sum(1 for row in rows if row["result"]["blocks_on_missing_required"])

    return {
        "canonical_accuracy": _ratio(canonical_matches, case_count),
        "rendered_accuracy": _ratio(rendered_matches, case_count),
        "missing_required_block_rate": _ratio(blocked_cases, case_count),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
