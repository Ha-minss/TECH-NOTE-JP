"""Tests for RecoveryCase -> OfficialDocumentView."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from recover24.document_view import build_document_view
from recover24.models import (
    EvidenceStatus,
    FieldValue,
    FraudType,
    RecoveryCase,
    ReportStatus,
    TransactionType,
)


def test_builds_core_display_values_from_raw_case():
    case = RecoveryCase.new("CASE-DOC-001")
    case.applicant.name = FieldValue.answered("박동하")
    case.applicant.birth_date = FieldValue.answered("1999-01-01")
    case.applicant.sms_consent = FieldValue.answered(True)
    case.incident.fraud_type = FieldValue.answered(FraudType.AUTHORITY_IMPERSONATION)
    case.transactions[0].amount_krw = FieldValue.answered(1_000_000)
    case.transactions[0].transaction_type = FieldValue.answered(TransactionType.MOBILE_BANKING_TRANSFER)
    case.investigation.status = FieldValue.answered(ReportStatus.NOT_REPORTED)
    case.consent.unique_id_collection_agreed = FieldValue.answered(True)
    case.consent.personal_credit_collection_agreed = FieldValue.answered(False)

    view = build_document_view(case, today=date(2026, 6, 22)).to_dict()

    assert view["applicant"]["name"] == "박동하"
    assert view["applicant"]["birthDate"] == "1999-01-01"
    assert view["checkbox"]["smsConsentYes"] == "☑"
    assert view["checkbox"]["smsConsentNo"] == "☐"
    assert view["accident"]["totalDamageAmountLabel"] == "1,000,000원"
    assert "수사기관 사칭" in view["accident"]["incidentTypeLabel"]
    assert view["transactions_padded"][0]["transactionType"] == "모바일뱅킹"
    assert view["investigation"]["statusNotReported"] == "☑"
    assert view["consent"]["uniqueIdCollectionAgree"] == "☑"
    assert view["consent"]["personalCreditCollectionDisagree"] == "☑"
    assert view["today"] == {"year": 2026, "month": 6, "day": 22}


def test_unknown_and_not_applicable_have_display_policy():
    case = RecoveryCase.new("CASE-DOC-002")
    case.applicant.email = FieldValue.unknown("이메일은 잘 모르겠습니다")
    case.delegation.request_purpose = FieldValue.not_applicable("본인이 직접 신청")

    view = build_document_view(case).to_dict()

    assert view["applicant"]["email"] == "미확인"
    # requestPurpose is intentionally blank for NA so the template default can show.
    assert view["delegation"]["requestPurpose"] == ""


def test_attachment_remarks_cover_all_official_items():
    case = RecoveryCase.new("CASE-DOC-003")
    case.evidence.append(
        __import__("recover24.models", fromlist=["EvidenceItem"]).EvidenceItem(
            kind="id_card_copy",
            status=EvidenceStatus.AVAILABLE,
            note="앞면 사본 보유",
        )
    )
    case.evidence.append(
        __import__("recover24.models", fromlist=["EvidenceItem"]).EvidenceItem(
            kind="police_certificate",
            status=EvidenceStatus.PLANNED,
        )
    )

    remarks = build_document_view(case).to_dict()["attachmentRemarks"]

    expected_keys = {
        "idCardCopy",
        "policeCertificate",
        "idLossEvidence",
        "phoneEvidence",
        "complaintEvidence",
        "investigationDelegation",
        "dataLeakNotice",
        "delayReason",
        "familyProof",
        "signatureCertificate",
        "securitySurvey",
        "otherEvidence",
        "passportOrTravelProof",
    }
    assert set(remarks) == expected_keys
    assert remarks["idCardCopy"] == "보유 - 앞면 사본 보유"
    assert remarks["policeCertificate"] == "추후 제출 예정"


def test_blank_case_document_view_renders_template_without_missing_variables():
    case = RecoveryCase.new("CASE-DOC-004")
    view = build_document_view(case, today=date(2026, 6, 22)).to_dict()

    template_dir = Path(__file__).resolve().parents[1] / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), undefined=StrictUndefined)
    html = env.get_template("recover24_official_report_v1.html").render(**view)

    assert "전자금융거래 사고 피해 신고서" in html
    assert "2026 년 6 월 22 일" in html
