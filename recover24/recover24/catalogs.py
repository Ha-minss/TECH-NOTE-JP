"""Canonical catalogs and display label maps for Recover24 V3.

catalogs.py contains stable code -> Korean label dictionaries.
It must not mutate RecoveryCase, call an LLM, or render HTML.
"""

from __future__ import annotations

from .models import EvidenceStatus, FraudType, ReportStatus, TransactionType


FRAUD_TYPE_LABELS: dict[FraudType, str] = {
    FraudType.AUTHORITY_IMPERSONATION: "검찰·경찰 등 수사기관 사칭 보이스피싱",
    FraudType.FAMILY_IMPERSONATION: "자녀·가족 사칭 메신저피싱",
    FraudType.LOAN_SCAM: "대출빙자 보이스피싱",
    FraudType.SMISHING_MALWARE: "스미싱 및 악성앱 보이스피싱",
    FraudType.INSTITUTION_IMPERSONATION: "금융회사·택배 등 기관 사칭 보이스피싱",
    FraudType.OTHER: "기타 전자금융거래 사고",
}


TRANSACTION_TYPE_LABELS: dict[TransactionType, str] = {
    TransactionType.MOBILE_BANKING_TRANSFER: "모바일뱅킹",
    TransactionType.INTERNET_BANKING_TRANSFER: "인터넷뱅킹",
    TransactionType.PHONE_BANKING_TRANSFER: "폰뱅킹",
    TransactionType.ATM_TRANSFER: "ATM",
    TransactionType.CARD_OR_LOAN: "카드/대출",
    TransactionType.UNKNOWN: "미확인",
}


REPORT_STATUS_LABELS: dict[ReportStatus, str] = {
    ReportStatus.NOT_REPORTED: "미신고",
    ReportStatus.PLANNED: "신청예정",
    ReportStatus.REPORTED: "신고",
    ReportStatus.IN_PROGRESS: "진행중",
    ReportStatus.COMPLETED: "종결",
    ReportStatus.CLOSED: "종결",
    ReportStatus.OTHER: "기타",
    ReportStatus.UNKNOWN: "미확인",
}


EVIDENCE_STATUS_LABELS: dict[EvidenceStatus, str] = {
    EvidenceStatus.AVAILABLE: "보유",
    EvidenceStatus.MISSING: "미제출",
    EvidenceStatus.PLANNED: "추후 제출 예정",
    EvidenceStatus.NOT_APPLICABLE: "해당없음",
    EvidenceStatus.UNKNOWN: "미확인",
    EvidenceStatus.NOT_ASKED: "",
}


EVIDENCE_TEMPLATE_KEYS: dict[str, str] = {
    "id_card_copy": "idCardCopy",
    "police_certificate": "policeCertificate",
    "id_loss_evidence": "idLossEvidence",
    "phone_evidence": "phoneEvidence",
    "complaint_evidence": "complaintEvidence",
    "investigation_delegation": "investigationDelegation",
    "data_leak_notice": "dataLeakNotice",
    "delay_reason": "delayReason",
    "family_proof": "familyProof",
    "signature_certificate": "signatureCertificate",
    "security_survey": "securitySurvey",
    "other_evidence": "otherEvidence",
    "passport_or_travel_proof": "passportOrTravelProof",
}
