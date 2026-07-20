"""Rendering checks for normalized values."""

from __future__ import annotations

from typing import Any

from recover24.document_view import build_document_view
from recover24.models import FieldValue, RecoveryCase


UNKNOWN = "unknown"


def render_expected_fields(canonical: dict[str, Any]) -> dict[str, str]:
    case = RecoveryCase.new("EVAL-NORM")

    if "damage_amount_krw" in canonical and isinstance(canonical.get("damage_amount_krw"), int):
        case.transactions[0].amount_krw = FieldValue.answered(canonical["damage_amount_krw"])

    if "incident_date" in canonical:
        if canonical["incident_date"] == UNKNOWN:
            case.investigation.reported_at = FieldValue.unknown()
        elif isinstance(canonical.get("incident_date"), str):
            case.investigation.reported_at = FieldValue.answered(canonical["incident_date"])

    view = build_document_view(case).to_dict()
    return {
        "totalDamageAmountLabel": view["accident"]["totalDamageAmountLabel"],
        "reportedAt": view["investigation"]["reportedAt"],
    }
