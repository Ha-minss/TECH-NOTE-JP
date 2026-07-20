from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .confidence import compute_confidence, has_meaningful_change, source_conflict
from .models import EvidenceItem, FieldChange, ProviderRecord, Recommendation
from .normalize import display_phone, normalize_active_status
from .utils import now_iso, sha256_text


ACTIONABLE_FIELDS = {"provider_name", "npi", "specialty", "practice_name", "address", "phone", "website", "active_status"}
CONFIRM_ONLY_SOURCES = {
    ("CMS Medicare FFS Public Provider Enrollment", "provider_name"),
    ("CMS Medicare FFS Public Provider Enrollment", "specialty"),
    ("CMS Revoked Medicare Providers and Suppliers", "provider_name"),
    ("CMS Revoked Medicare Providers and Suppliers", "specialty"),
}


def old_value(record: ProviderRecord, field: str) -> str:
    mapping = {
        "provider_name": record.provider_name,
        "npi": record.npi,
        "specialty": record.specialty,
        "practice_name": record.practice_name,
        "address": record.address,
        "phone": record.phone,
        "website": record.website or "",
        "active_status": record.active_status,
    }
    return mapping.get(field, "")


def display_value(field: str, value: str) -> str:
    if field == "phone":
        return display_phone(value)
    return value


def is_confirm_only(ev: EvidenceItem) -> bool:
    if ev.metadata.get("decision_policy") == "confirm_only":
        return True
    return (ev.source_name, ev.field) in CONFIRM_ONLY_SOURCES


def build_field_changes(record: ProviderRecord, evidence: List[EvidenceItem], config: Dict) -> List[FieldChange]:
    by_field_value: Dict[tuple, List[EvidenceItem]] = defaultdict(list)

    for ev in evidence:
        if ev.field not in ACTIONABLE_FIELDS:
            continue
        if is_confirm_only(ev) and not ev.metadata.get("force_review"):
            continue
        if not has_meaningful_change(record, ev.field, ev.value):
            continue
        by_field_value[(ev.field, ev.normalized_value)].append(ev)

    changes: List[FieldChange] = []

    for (field, _norm), items in by_field_value.items():
        best = sorted(items, key=lambda ev: (len(items), ev.source_confidence), reverse=True)[0]
        actionable_for_field = [ev for ev in evidence if ev.field == field and not is_confirm_only(ev)]
        conf = compute_confidence(record, best, evidence, config)
        conflict = source_conflict(actionable_for_field or evidence, field)

        high_risk = field in set(config.get("high_risk_fields", []))
        revoked = field == "active_status" and normalize_active_status(best.value) == "revoked"

        requires_review = conflict or high_risk or revoked or bool(best.metadata.get("force_review"))

        changes.append(FieldChange(
            field=field,
            old_value=old_value(record, field),
            new_value=display_value(field, best.value),
            confidence_score=conf,
            supporting_sources=sorted({ev.source_name for ev in items}),
            evidence_urls=sorted({ev.source_url for ev in items if ev.source_url}),
            evidence_snippets=[ev.evidence_text for ev in items if ev.evidence_text][:5],
            source_conflict=conflict,
            requires_human_review=requires_review,
        ))

    return sorted(changes, key=lambda c: (c.field, c.new_value))


def choose_action(changes: List[FieldChange], config: Dict) -> tuple[str, str, float]:
    if not changes:
        return "no_change", "Record confirmed as accurate against collected evidence.", 0.90

    overall = round(sum(c.confidence_score for c in changes) / len(changes), 4)
    auto_threshold = float(config.get("auto_update_threshold", 0.86))
    review_threshold = float(config.get("human_review_threshold", 0.60))
    allowed = set(config.get("auto_update_allowed_fields", []))

    if any(c.source_conflict for c in changes):
        return "human_review", "Conflicting sources found. Manual verification recommended.", overall

    if any(c.requires_human_review for c in changes):
        return "human_review", "At least one high-risk candidate requires manual verification.", overall

    if all(c.field in allowed and c.confidence_score >= auto_threshold for c in changes):
        return "auto_update", "Updated values were confirmed by reliable source evidence.", overall

    if overall >= review_threshold:
        return "human_review", "Potential update found, but confidence or field risk is not high enough for auto-update.", overall

    return "outreach_required", "Evidence is insufficient for safe update; direct provider/practice verification recommended.", overall


def build_recommendation(record: ProviderRecord, changes: List[FieldChange], config: Dict) -> Recommendation:
    action, reason, overall = choose_action(changes, config)
    audit_id = "AUD_" + sha256_text(f"{record.provider_id}|{record.npi}|{now_iso()}")[:16]
    return Recommendation(
        provider_id=record.provider_id,
        npi=record.npi,
        change_detected=bool(changes),
        changes=changes,
        overall_confidence=overall,
        recommended_action=action,
        reason=reason,
        audit_id=audit_id,
    )