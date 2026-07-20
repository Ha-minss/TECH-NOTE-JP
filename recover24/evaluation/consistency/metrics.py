"""Metrics for deterministic consistency evaluation."""

from __future__ import annotations

from typing import Any


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_expected_conflicts = 0
    total_predicted_conflicts = 0
    total_matching_conflicts = 0
    total_field_scores = 0.0
    blocking_matches = 0
    false_blocks = 0

    for row in rows:
        expected_fields = set(row["expected_conflict_fields"])
        predicted_fields = set(row["predicted_conflict_fields"])

        total_expected_conflicts += len(expected_fields)
        total_predicted_conflicts += len(predicted_fields)
        total_matching_conflicts += len(expected_fields & predicted_fields)

        union = expected_fields | predicted_fields
        total_field_scores += 1.0 if not union else len(expected_fields & predicted_fields) / len(union)

        expected_block = row["expected_can_generate_document"] is False
        predicted_block = row["result"]["can_generate_document"] is False
        if expected_block == predicted_block:
            blocking_matches += 1
        if predicted_block and not expected_block:
            false_blocks += 1

    case_count = len(rows)
    return {
        "conflict_recall": _safe_divide(total_matching_conflicts, total_expected_conflicts),
        "conflict_precision": _safe_divide(total_matching_conflicts, total_predicted_conflicts),
        "blocking_accuracy": _safe_divide(blocking_matches, case_count),
        "false_block_rate": _safe_divide(false_blocks, case_count),
        "field_accuracy": _safe_divide(total_field_scores, case_count),
    }


def summarize_extractor_results(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_expected = 0
    total_predicted = 0
    total_matched = 0
    exact_matches = 0

    for row in rows:
        expected = row["expected_statement_facts"]
        predicted = row["predicted_statement_facts"]
        expected_items = set(expected.items())
        predicted_items = set(predicted.items())

        total_expected += len(expected_items)
        total_predicted += len(predicted_items)
        total_matched += len(expected_items & predicted_items)
        if expected == predicted:
            exact_matches += 1

    case_count = len(rows)
    return {
        "extractor_accuracy": _safe_divide(exact_matches, case_count),
        "field_recall": _safe_divide(total_matched, total_expected),
        "field_precision": _safe_divide(total_matched, total_predicted),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
