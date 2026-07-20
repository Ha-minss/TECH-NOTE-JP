from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .models import EvidenceItem, ProviderRecord
from .normalize import days_since, normalize_field_value, values_equal


def source_agreement(evidence: List[EvidenceItem], candidate: EvidenceItem) -> float:
    same_field = [ev for ev in evidence if ev.field == candidate.field]
    if not same_field:
        return 0.0
    counts = Counter(ev.normalized_value for ev in same_field)
    return counts[candidate.normalized_value] / len(same_field)


def source_conflict(evidence: List[EvidenceItem], field: str) -> bool:
    values = {ev.normalized_value for ev in evidence if ev.field == field and ev.normalized_value}
    return len(values) > 1


def entity_match_confidence(record: ProviderRecord, evidence: List[EvidenceItem]) -> float:
    """MVP identity anchor: exact NPI evidence gives strongest confidence."""
    if record.npi:
        for ev in evidence:
            if ev.field == "npi" and ev.normalized_value == normalize_field_value("npi", record.npi):
                return 1.0
    return 0.70


def recency_score(record: ProviderRecord, stale_after_days: int) -> float:
    d = days_since(record.last_verified_date)
    if d is None:
        return 0.50
    if d <= stale_after_days:
        return 0.90
    if d <= stale_after_days * 2:
        return 0.70
    if d <= stale_after_days * 3:
        return 0.55
    return 0.40


def compute_confidence(
    record: ProviderRecord,
    candidate: EvidenceItem,
    all_evidence: List[EvidenceItem],
    config: Dict,
) -> float:
    reliability = candidate.source_confidence
    agreement = source_agreement(all_evidence, candidate)
    entity_match = entity_match_confidence(record, all_evidence)
    recency = recency_score(record, int(config.get("stale_after_days", 365)))
    safety = float(config["field_safety"].get(candidate.field, 0.50))

    score = (
        0.35 * reliability
        + 0.25 * agreement
        + 0.20 * entity_match
        + 0.10 * recency
        + 0.10 * safety
    )
    return round(max(0.0, min(1.0, score)), 4)


def has_meaningful_change(record: ProviderRecord, field: str, evidence_value: str) -> bool:
    old_map = {
        "provider_name": record.provider_name,
        "npi": record.npi,
        "specialty": record.specialty,
        "practice_name": record.practice_name,
        "address": record.address,
        "phone": record.phone,
        "website": record.website or "",
        "active_status": record.active_status,
    }
    old_value = old_map.get(field, "")
    if not old_value and evidence_value:
        return True
    return not values_equal(field, old_value, evidence_value)
