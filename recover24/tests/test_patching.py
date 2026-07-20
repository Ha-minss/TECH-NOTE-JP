"""Tests for Patch[] -> RecoveryCase."""

from __future__ import annotations

import pytest

from recover24.models import FieldStatus, FraudType, Patch, RecoveryCase, ReportStatus
from recover24.patching import PatchPathError, PatchTargetError, apply_patch, apply_patches


def test_apply_patches_writes_field_values_without_mutating_original():
    case = RecoveryCase.new("CASE-001")

    patched = apply_patches(
        case,
        [
            Patch(
                path="incident.fraud_type",
                value=FraudType.AUTHORITY_IMPERSONATION,
                source_text="검찰 사칭",
            ),
            Patch(
                path="transactions.0.amount_krw",
                value=1_000_000,
                source_text="100만 원",
            ),
        ],
    )

    assert case.incident.fraud_type.status == FieldStatus.NOT_ASKED
    assert case.transactions[0].amount_krw.status == FieldStatus.NOT_ASKED

    assert patched.incident.fraud_type.status == FieldStatus.ANSWERED
    assert patched.incident.fraud_type.value == FraudType.AUTHORITY_IMPERSONATION
    assert patched.incident.fraud_type.source_text == "검찰 사칭"

    assert patched.transactions[0].amount_krw.status == FieldStatus.ANSWERED
    assert patched.transactions[0].amount_krw.value == 1_000_000
    assert patched.transactions[0].amount_krw.source_text == "100만 원"


def test_apply_patch_supports_unknown_and_not_applicable_statuses():
    case = RecoveryCase.new("CASE-002")

    case = apply_patch(
        case,
        Patch(
            path="investigation.agency",
            value=None,
            status=FieldStatus.UNKNOWN,
            source_text="어느 경찰서인지는 모르겠어요",
        ),
    )

    assert case.investigation.agency.value is None
    assert case.investigation.agency.status == FieldStatus.UNKNOWN
    assert case.investigation.agency.source_text == "어느 경찰서인지는 모르겠어요"

    case = apply_patch(
        case,
        Patch(
            path="relief.bank1",
            value=None,
            status=FieldStatus.NOT_APPLICABLE,
            source_text="피해구제 신청은 아직 안 했어요",
        ),
    )

    assert case.relief.bank1.value is None
    assert case.relief.bank1.status == FieldStatus.NOT_APPLICABLE


def test_apply_patches_expands_transaction_rows_when_needed():
    case = RecoveryCase.new("CASE-003")

    patched = apply_patch(
        case,
        Patch(path="transactions.1.amount_krw", value=300_000, source_text="두 번째로 30만 원"),
    )

    assert len(case.transactions) == 1
    assert len(patched.transactions) == 2
    assert patched.transactions[1].amount_krw.value == 300_000
    assert patched.transactions[1].amount_krw.status == FieldStatus.ANSWERED


def test_rejects_unknown_path():
    case = RecoveryCase.new("CASE-004")

    with pytest.raises(PatchPathError):
        apply_patch(case, Patch(path="victim.money", value=1_000_000))


def test_rejects_non_fieldvalue_target():
    case = RecoveryCase.new("CASE-005")

    with pytest.raises(PatchTargetError):
        apply_patch(case, Patch(path="incident", value="not allowed"))


def test_report_status_patch_is_stored_as_enum_value_from_extraction_layer():
    case = RecoveryCase.new("CASE-006")

    patched = apply_patch(
        case,
        Patch(path="investigation.status", value=ReportStatus.NOT_REPORTED, source_text="경찰에는 아직 안 갔어요"),
    )

    assert patched.investigation.status.value == ReportStatus.NOT_REPORTED
    assert patched.investigation.status.status == FieldStatus.ANSWERED


def test_apply_evidence_status_patch_creates_or_updates_evidence_item():
    from recover24.models import EvidenceStatus

    case = RecoveryCase.new("CASE-EVIDENCE")
    next_case = apply_patches(
        case,
        [
            Patch(
                path="evidence.id_card_copy.status",
                value=EvidenceStatus.AVAILABLE,
                source_text="신분증 사본은 있어요",
            ),
            Patch(
                path="evidence.id_card_copy.note",
                value="주민등록증 앞면 사본 보유",
            ),
        ],
    )

    assert len(next_case.evidence) == 1
    assert next_case.evidence[0].kind == "id_card_copy"
    assert next_case.evidence[0].status == EvidenceStatus.AVAILABLE
    assert next_case.evidence[0].note == "주민등록증 앞면 사본 보유"
    assert next_case.evidence[0].source_text == "신분증 사본은 있어요"
