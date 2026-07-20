"""Tests for submission readiness checks."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from recover24.models import (
    EvidenceItem,
    EvidenceStatus,
    FieldValue,
    FraudType,
    RecoveryCase,
    ReportStatus,
    TransactionType,
)
from recover24.questions import EVIDENCE_ITEM_LABELS
from recover24.readiness import ReadinessStatus, evaluate_readiness


def _mark_all_fields_not_applicable(value: Any) -> None:
    if isinstance(value, FieldValue):
        value.value = None
        value.status = value.status.NOT_APPLICABLE
        value.source_text = "test baseline: not applicable"
        return

    if isinstance(value, list):
        for item in value:
            _mark_all_fields_not_applicable(item)
        return

    if isinstance(value, dict):
        for item in value.values():
            _mark_all_fields_not_applicable(item)
        return

    if is_dataclass(value):
        for field in fields(value):
            if field.name in {"case_id", "created_at"}:
                continue
            _mark_all_fields_not_applicable(getattr(value, field.name))


def _complete_ready_case() -> RecoveryCase:
    case = RecoveryCase.new("CASE-READY")
    _mark_all_fields_not_applicable(case)

    # Applicant core
    case.applicant.name = FieldValue.answered("김영희")
    case.applicant.birth_date = FieldValue.answered("1990-01-01")
    case.applicant.mobile_number = FieldValue.answered("010-1234-5678")
    case.applicant.address = FieldValue.answered("서울시 중구")
    case.applicant.sms_consent = FieldValue.answered(True)

    # Exclusion checklist confirmed as not applying.
    for key in case.exclusion.items:
        case.exclusion.items[key] = FieldValue.answered(False)
    case.exclusion.final_has_exclusion = FieldValue.answered(False)

    # Incident and transaction core
    case.incident.first_occurred_at = FieldValue.answered("2026-06-20 14:00")
    case.incident.recognized_at = FieldValue.answered("2026-06-20 14:05")
    case.incident.first_freeze_at = FieldValue.answered("2026-06-20 14:20")
    case.incident.fraud_type = FieldValue.answered(FraudType.OTHER)
    case.incident.overview = FieldValue.answered("전자금융거래 사고 피해 신고")

    tx = case.transactions[0]
    tx.source_bank = FieldValue.answered("국민은행")
    tx.source_account_number = FieldValue.answered("123-456")
    tx.amount_krw = FieldValue.answered(1_000_000)
    tx.destination_bank = FieldValue.answered("카카오뱅크")
    tx.destination_account_number = FieldValue.answered("3333-12-3456789")
    tx.destination_account_holder = FieldValue.answered("김철수")
    tx.holder_type = FieldValue.answered("타인")
    tx.transaction_type = FieldValue.answered(TransactionType.MOBILE_BANKING_TRANSFER)
    tx.transferred_at = FieldValue.answered("2026-06-20 14:00")

    # Statuses that should not need staff review.
    case.relief.status = FieldValue.answered(ReportStatus.COMPLETED)
    case.relief.bank1 = FieldValue.answered("국민은행")
    case.relief.date1 = FieldValue.answered("2026-06-20")
    case.investigation.status = FieldValue.answered(ReportStatus.REPORTED)
    case.investigation.agency = FieldValue.answered("서울중부경찰서")
    case.investigation.reported_at = FieldValue.answered("2026-06-20")

    # Narrative and consent
    case.narrative.incident_circumstances = FieldValue.answered("상대방의 기망으로 송금했습니다.")
    case.narrative.post_action = FieldValue.answered("은행에 지급정지를 요청하고 경찰에 신고했습니다.")
    case.consent.unique_id_collection_agreed = FieldValue.answered(True)
    case.consent.personal_credit_collection_agreed = FieldValue.answered(True)
    case.consent.unique_id_provision_agreed = FieldValue.answered(True)
    case.consent.personal_credit_provision_agreed = FieldValue.answered(True)

    # Direct application, no delegation required.
    case.delegation.proxy_used = FieldValue.answered(False)

    # Confirm every official evidence row. Most are not applicable in this baseline.
    case.evidence = [EvidenceItem(kind=kind, status=EvidenceStatus.NOT_APPLICABLE) for kind in EVIDENCE_ITEM_LABELS]
    case.evidence.append(EvidenceItem(kind="id_card_copy", status=EvidenceStatus.AVAILABLE))

    return case


def test_blank_case_is_not_ready_and_requires_user_action():
    report = evaluate_readiness(RecoveryCase.new("CASE-BLANK"))

    assert report.status == ReadinessStatus.NOT_READY
    assert report.requires_user_action is True
    assert report.can_submit_officially is False
    assert any(issue.code == "document_fields_not_confirmed" for issue in report.issues)


def test_complete_case_is_ready_for_official_submission():
    report = evaluate_readiness(_complete_ready_case())

    assert report.status == ReadinessStatus.READY
    assert report.can_submit_officially is True
    assert report.requires_user_action is False
    assert report.requires_staff_decision is False
    assert report.document_completion_rate == 1.0


def test_missing_core_transaction_field_blocks_user_submission():
    case = _complete_ready_case()
    case.transactions[0].destination_account_number = FieldValue[str]()

    report = evaluate_readiness(case)

    assert report.status == ReadinessStatus.NOT_READY
    assert any(issue.code == "missing_transaction_identity" for issue in report.issues)
    assert any("transactions.0.destination_account_number" in issue.paths for issue in report.blocker_issues)


def test_family_impersonation_without_messenger_capture_routes_to_staff_review():
    case = _complete_ready_case()
    case.incident.fraud_type = FieldValue.answered(FraudType.FAMILY_IMPERSONATION)
    for item in case.evidence:
        if item.kind == "other_evidence":
            item.status = EvidenceStatus.MISSING
            item.note = "카카오톡 대화가 삭제되어 없음"

    report = evaluate_readiness(case)

    assert report.status == ReadinessStatus.NEEDS_REVIEW
    assert report.requires_staff_decision is True
    assert report.requires_user_action is False
    assert any(issue.code == "conditional_evidence_needs_staff_review" for issue in report.review_issues)


def test_missing_required_consent_blocks_submission():
    case = _complete_ready_case()
    case.consent.personal_credit_provision_agreed = FieldValue.answered(False)

    report = evaluate_readiness(case)

    assert report.status == ReadinessStatus.NOT_READY
    assert any(issue.code == "required_consent_not_agreed" for issue in report.blocker_issues)
