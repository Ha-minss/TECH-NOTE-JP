"""Deterministic checklist evaluation for generated narratives."""

from __future__ import annotations

import re
from typing import Any


def evaluate_narrative(
    *,
    canonical_case: dict[str, Any],
    generated_text: str,
    required_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    text = generated_text or ""
    included: list[str] = []
    missing: list[str] = []

    for element in required_elements:
        expected = element.get("expected", [])
        if any(token in text for token in expected):
            included.append(element["id"])
        else:
            missing.append(element["id"])

    factual_errors = _find_factual_errors(canonical_case, text)
    return {
        "included_elements": included,
        "missing_elements": missing,
        "factual_errors": factual_errors,
        "passed": not missing and not factual_errors,
    }


def _find_factual_errors(canonical_case: dict[str, Any], text: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    expected_amount = canonical_case.get("damage_amount_krw")
    amounts = {
        int(match.group(1).replace(",", ""))
        for match in re.finditer(r"(\d[\d,]*)원", text)
    }
    if isinstance(expected_amount, int) and amounts and amounts != {expected_amount}:
        errors.append({"field": "damage_amount_krw", "message": "Narrative includes a mismatched amount."})

    expected_dates = {
        value
        for key, value in canonical_case.items()
        if key.endswith("_date") and isinstance(value, str)
    }
    mentioned_dates = {
        match.group(0)
        for match in re.finditer(r"20\d{2}-\d{2}-\d{2}", text)
    }
    if expected_dates and mentioned_dates and mentioned_dates != expected_dates:
        errors.append({"field": "date", "message": "Narrative includes a mismatched date."})

    return errors
