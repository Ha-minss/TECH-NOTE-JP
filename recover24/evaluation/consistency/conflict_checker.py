"""Deterministic consistency checks between form facts and statement facts."""

from __future__ import annotations

from typing import Any

BLOCKING_FIELDS = {
    "damage_amount_krw",
    "incident_date",
    "transfer_date",
    "police_status",
    "freeze_status",
    "refund_status",
    "recipient_account_known",
}


def check_consistency(
    *,
    form_facts: dict[str, Any],
    statement_facts: dict[str, Any],
) -> dict[str, Any]:
    conflicts: list[dict[str, Any]] = []
    enrichments: list[dict[str, Any]] = []

    for field in sorted(set(form_facts) | set(statement_facts)):
        form_value = form_facts.get(field)
        statement_value = statement_facts.get(field)

        form_known = _is_known(form_value)
        statement_known = _is_known(statement_value)

        if form_known and statement_known:
            if form_value != statement_value:
                conflicts.append(
                    {
                        "field": field,
                        "form_value": form_value,
                        "statement_value": statement_value,
                        "severity": "blocking" if field in BLOCKING_FIELDS else "warning",
                        "message": f"Field '{field}' conflicts between form and statement.",
                    }
                )
            continue

        if (not form_known) and statement_known:
            enrichments.append(
                {
                    "field": field,
                    "statement_value": statement_value,
                    "message": f"Field '{field}' was enriched from the statement.",
                }
            )

    has_blocking_conflict = any(item["severity"] == "blocking" for item in conflicts)
    return {
        "conflicts": conflicts,
        "enrichments": enrichments,
        "can_generate_document": not has_blocking_conflict,
        "requires_human_review": has_blocking_conflict,
    }


def _is_known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "unknown", "null", "none", "n/a"}
    return True
