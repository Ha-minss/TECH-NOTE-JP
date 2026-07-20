"""Status-value rules for dynamic contradiction detection.

No case-specific banned sentence list lives here. The validator compares an
extracted status claim against the original structured_facts value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_FIELDS = {
    "freeze_status": {
        "allowed": {"not_requested", "requested", "completed", "attempted_but_failed", "unknown"},
        "contradictions": {
            "not_requested": {"requested", "completed", "attempted_but_failed"},
            "requested": {"not_requested", "completed", "attempted_but_failed"},
            "completed": {"not_requested", "attempted_but_failed"},
            "attempted_but_failed": {"requested", "completed"},
            "unknown": {"requested", "completed", "attempted_but_failed", "not_requested"},
        },
    },
    "police_status": {
        "allowed": {"not_reported", "planned", "reported", "in_progress", "closed", "unknown"},
        "contradictions": {
            "not_reported": {"reported", "in_progress", "closed"},
            "planned": {"reported", "in_progress", "closed"},
            "reported": {"not_reported", "closed"},
            "in_progress": {"not_reported", "closed"},
            "closed": {"not_reported", "reported", "in_progress"},
            "unknown": {"not_reported", "planned", "reported", "in_progress", "closed"},
        },
    },
    "refund_status": {
        "allowed": {"not_applied", "planned", "applied", "in_progress", "completed", "unknown"},
        "contradictions": {
            "not_applied": {"applied", "in_progress", "completed"},
            "planned": {"applied", "in_progress", "completed"},
            "applied": {"not_applied", "completed"},
            "in_progress": {"not_applied", "completed"},
            "completed": {"not_applied", "planned", "applied", "in_progress"},
            "unknown": {"not_applied", "planned", "applied", "in_progress", "completed"},
        },
    },
}


@dataclass(frozen=True)
class StatusJudgement:
    field: str
    gold_value: str
    claimed_value: str
    label: str  # supported / contradicted / unsupported / ignored
    evidence_text: str = ""


def judge_status_claim(field: str, claimed_value: str, facts: dict[str, Any], evidence_text: str = "") -> StatusJudgement:
    gold_value = str(facts.get(field, "unknown") or "unknown")
    claimed_value = str(claimed_value or "unknown")
    spec = STATUS_FIELDS.get(field)
    if spec is None:
        return StatusJudgement(field, gold_value, claimed_value, "unsupported", evidence_text)
    if claimed_value not in spec["allowed"]:
        return StatusJudgement(field, gold_value, claimed_value, "unsupported", evidence_text)
    if gold_value == claimed_value:
        return StatusJudgement(field, gold_value, claimed_value, "supported", evidence_text)
    contradictions = spec["contradictions"].get(gold_value, set())
    if claimed_value in contradictions:
        return StatusJudgement(field, gold_value, claimed_value, "contradicted", evidence_text)
    # Example: gold=reported and claim=in_progress. It may be a more specific claim, but
    # if the raw facts did not support it, treat it as unsupported rather than contradicted.
    return StatusJudgement(field, gold_value, claimed_value, "unsupported", evidence_text)


def supported_status_fields_from_facts(facts: dict[str, Any]) -> set[str]:
    return {field for field in STATUS_FIELDS if field in facts and facts.get(field) not in (None, "unknown", "")}
