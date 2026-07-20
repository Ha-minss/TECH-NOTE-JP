"""Tests for initial statement -> Patch[]."""

from __future__ import annotations

from typing import Any

from recover24.extraction import extract_initial_statement
from recover24.models import FraudType, ReportStatus, TransactionType


class FakeProvider:
    def __init__(self, payload: dict[str, Any] | str):
        self.payload = payload
        self.last_prompt: str | None = None

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        self.last_prompt = prompt
        return self.payload


def test_llm_patch_json_to_normalized_patches():
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
                    "confidence": 0.9,
                },
                {
                    "path": "transactions.0.transaction_type",
                    "value": "mobile_banking_transfer",
                    "source_text": "모바일뱅킹",
                    "confidence": 0.8,
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

    by_path = {patch.path: patch for patch in patches}

    assert by_path["incident.fraud_type"].value == FraudType.AUTHORITY_IMPERSONATION
    assert by_path["transactions.0.amount_krw"].value == 1_000_000
    assert by_path["transactions.0.transaction_type"].value == TransactionType.MOBILE_BANKING_TRANSFER
    assert by_path["investigation.status"].value == ReportStatus.NOT_REPORTED


def test_rejects_unknown_paths_and_bad_enum_values():
    provider = FakeProvider(
        {
            "patches": [
                {"path": "victim.money", "value": 1000000},
                {"path": "incident.fraud_type", "value": "random_fake_type"},
                {"path": "transactions.0.amount_krw", "value": 1000000},
            ]
        }
    )

    patches = extract_initial_statement("100만원 보냈어요", provider)

    assert len(patches) == 1
    assert patches[0].path == "transactions.0.amount_krw"
    assert patches[0].value == 1_000_000


def test_parses_json_inside_markdown_fence():
    provider = FakeProvider(
        """```json
        {"patches":[{"path":"transactions.0.amount_krw","value":"1,000,000원"}]}
        ```"""
    )

    patches = extract_initial_statement("100만원 보냈어요", provider)

    assert patches[0].path == "transactions.0.amount_krw"
    assert patches[0].value == 1_000_000


def test_initial_extraction_allowlist_covers_all_non_consent_model_fields():
    from recover24.extraction import ALLOWED_PATCH_PATHS, BLOCKED_INITIAL_EXTRACTION_PATHS
    from recover24.questions import FIELD_LABELS

    missing = (set(FIELD_LABELS) - BLOCKED_INITIAL_EXTRACTION_PATHS) - ALLOWED_PATCH_PATHS

    assert missing == set()


def test_initial_extraction_allowlist_covers_all_official_evidence_items():
    from recover24.extraction import EVIDENCE_ITEM_KINDS, EVIDENCE_NOTE_PATHS, EVIDENCE_STATUS_PATHS
    from recover24.questions import EVIDENCE_ITEM_LABELS

    assert set(EVIDENCE_ITEM_KINDS) == set(EVIDENCE_ITEM_LABELS)
    assert EVIDENCE_STATUS_PATHS == {f"evidence.{kind}.status" for kind in EVIDENCE_ITEM_LABELS}
    assert EVIDENCE_NOTE_PATHS == {f"evidence.{kind}.note" for kind in EVIDENCE_ITEM_LABELS}


def test_normalizes_evidence_status_patch():
    from recover24.models import EvidenceStatus

    provider = FakeProvider(
        {
            "patches": [
                {
                    "path": "evidence.id_card_copy.status",
                    "value": "available",
                    "source_text": "신분증 사본은 있어요",
                    "confidence": 0.9,
                },
                {
                    "path": "evidence.police_certificate.status",
                    "value": "missing",
                    "source_text": "사건사고사실확인원은 아직 없어요",
                    "confidence": 0.9,
                },
            ]
        }
    )

    patches = extract_initial_statement("신분증 사본은 있는데 경찰 확인원은 아직 없어요", provider)
    by_path = {patch.path: patch for patch in patches}

    assert by_path["evidence.id_card_copy.status"].value == EvidenceStatus.AVAILABLE
    assert by_path["evidence.police_certificate.status"].value == EvidenceStatus.MISSING
