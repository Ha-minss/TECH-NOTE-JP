"""Integration smoke test: first statement -> extraction.py -> patching.py -> RecoveryCase."""

from __future__ import annotations

from typing import Any

from recover24.extraction import extract_initial_statement
from recover24.models import FieldStatus, FraudType, RecoveryCase, ReportStatus, TransactionType
from recover24.patching import apply_patches


class FakeProvider:
    def __init__(self, payload: dict[str, Any] | str):
        self.payload = payload
        self.last_prompt: str | None = None

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        self.last_prompt = prompt
        return self.payload


def test_initial_statement_becomes_recovery_case_state():
    provider = FakeProvider(
        {
            "patches": [
                {
                    "path": "incident.fraud_type",
                    "value": "authority_impersonation",
                    "source_text": "검찰 사칭",
                    "confidence": 0.95,
                },
                {
                    "path": "transactions.0.amount_krw",
                    "value": "100만 원",
                    "source_text": "100만 원",
                    "confidence": 0.95,
                },
                {
                    "path": "transactions.0.transaction_type",
                    "value": "mobile_banking_transfer",
                    "source_text": "모바일뱅킹",
                    "confidence": 0.9,
                },
                {
                    "path": "investigation.status",
                    "value": "not_reported",
                    "source_text": "경찰에는 아직 안 갔어요",
                    "confidence": 0.9,
                },
            ]
        }
    )

    patches = extract_initial_statement(
        "검찰 사칭으로 100만 원을 모바일뱅킹으로 보냈고 경찰에는 아직 안 갔어요.",
        provider,
    )
    case = apply_patches(RecoveryCase.new("CASE-INTEGRATION-001"), patches)

    assert case.incident.fraud_type.status == FieldStatus.ANSWERED
    assert case.incident.fraud_type.value == FraudType.AUTHORITY_IMPERSONATION
    assert case.transactions[0].amount_krw.value == 1_000_000
    assert case.transactions[0].transaction_type.value == TransactionType.MOBILE_BANKING_TRANSFER
    assert case.investigation.status.value == ReportStatus.NOT_REPORTED
