"""Aggregate Recover24 narrative-evaluation scores."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)

    summary = {"methods": {}, "num_records": len(records)}
    for method, rows in sorted(by_method.items()):
        summary["methods"][method] = _summarize_method(rows)
    return summary


def _summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_scores = []
    amount_preservation = []
    contradiction_rates = []
    unsupported_rates = []
    order_scores = []
    fallback_flags = []
    safe_flags = []
    latencies = []
    llm_calls = []

    for row in rows:
        validation = row.get("validation", {})
        required = validation.get("required_facts", {})
        if required:
            required_scores.append(required.get("score", 0.0))

        amounts = validation.get("amounts", {})
        amount_errors = amounts.get("errors", [])
        required_amounts = amounts.get("required_amounts_krw", [])
        amount_preservation.append(1.0 if not amount_errors and required_amounts else (1.0 if not required_amounts else 0.0))

        availability = validation.get("metric_availability", {})
        status = validation.get("status_claims", [])
        if availability.get("status_claims", bool(status)):
            contradiction_rates.append(_rate(status, "contradicted"))

        unsupported_claims = validation.get("unsupported_claims", [])
        if availability.get("unsupported_claims", bool(unsupported_claims)):
            unsupported_rates.append(1.0 if unsupported_claims else 0.0)

        order = validation.get("event_order", {})
        if availability.get("event_order", order.get("score") is not None) and order.get("score") is not None:
            order_scores.append(order["score"])

        meta = row.get("meta", {})
        fallback_flags.append(1.0 if meta.get("fallback_used") else 0.0)
        safe_flags.append(1.0 if validation.get("safe_to_use", False) else 0.0)
        latencies.append(float(meta.get("latency_sec", 0.0)))
        llm_calls.append(float(meta.get("llm_calls", 0.0)))

    return {
        "cases": len(rows),
        "important_fact_inclusion": _avg(required_scores),
        "amount_preservation": _avg(amount_preservation),
        "status_contradiction_rate": _avg(contradiction_rates) if contradiction_rates else None,
        "unsupported_claim_case_rate": _avg(unsupported_rates) if unsupported_rates else None,
        "event_order_accuracy": _avg(order_scores) if order_scores else None,
        "safe_output_rate": _avg(safe_flags),
        "fallback_rate": _avg(fallback_flags),
        "avg_latency_sec": _avg(latencies),
        "avg_llm_calls": _avg(llm_calls),
    }


def _rate(items: list[dict[str, Any]], label: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get("label") == label) / len(items)


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0
