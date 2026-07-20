"""Tests for follow-up answer -> Patch[]."""

from __future__ import annotations

from typing import Any

from recover24.answers import extract_answer_patches
from recover24.models import (
    EvidenceStatus,
    FieldStatus,
    Question,
    QuestionCategory,
    RecoveryCase,
)
from recover24.patching import apply_patches


class FakeProvider:
    def __init__(self, payload: dict[str, Any] | str):
        self.payload = payload
        self.last_prompt: str | None = None

    def generate_json(self, prompt: str) -> dict[str, Any] | str:
        self.last_prompt = prompt
        return self.payload


def test_transaction_answer_extracts_only_question_target_paths():
    question = Question(
        question_id="transaction.primary_details",
        category=QuestionCategory.TRANSACTION,
        prompt="피해 거래 정보 중 수취인 이름과 송금시각만 추가로 알려주세요.",
        target_paths=[
            "transactions.0.destination_account_holder",
            "transactions.0.transferred_at",
        ],
    )
    provider = FakeProvider(
        {
            "patches": [
                {
                    "path": "transactions.0.destination_account_holder",
                    "value": "김철수",
                    "source_text": "김철수에게",
                    "confidence": 0.95,
                },
                {
                    "path": "transactions.0.transferred_at",
                    "value": "어제 오후 2시",
                    "source_text": "어제 오후 2시쯤",
                    "confidence": 0.90,
                },
                {
                    "path": "investigation.status",
                    "value": "not_reported",
                    "source_text": "경찰 신고는 아직 안 했어요",
                    "confidence": 0.90,
                },
            ]
        }
    )

    patches = extract_answer_patches(question, "김철수에게 어제 오후 2시에 보냈고 경찰 신고는 아직 안 했어요.", provider)

    assert [patch.path for patch in patches] == [
        "transactions.0.destination_account_holder",
        "transactions.0.transferred_at",
    ]
    assert patches[0].value == "김철수"
    assert patches[1].value == "어제 오후 2시"
    assert "investigation.status" not in {patch.path for patch in patches}
    assert "transactions.0.destination_account_holder" in (provider.last_prompt or "")
    assert "investigation.status" not in (provider.last_prompt or "")


def test_unknown_answer_creates_unknown_patch_for_target_path():
    question = Question(
        question_id="investigation.status",
        category=QuestionCategory.REPORT_AND_FREEZE,
        prompt="수사기관명을 알려주세요.",
        target_paths=["investigation.agency"],
    )
    provider = FakeProvider(
        {
            "patches": [
                {
                    "path": "investigation.agency",
                    "status": "unknown",
                    "source_text": "어느 경찰서인지는 모르겠어요",
                    "confidence": 0.9,
                }
            ]
        }
    )

    patches = extract_answer_patches(question, "어느 경찰서인지는 모르겠어요.", provider)

    assert len(patches) == 1
    assert patches[0].path == "investigation.agency"
    assert patches[0].value is None
    assert patches[0].status == FieldStatus.UNKNOWN


def test_consent_answer_can_patch_all_required_consent_fields():
    question = Question(
        question_id="consent.required_bundle",
        category=QuestionCategory.CONSENT,
        prompt="필수 개인정보 동의 4개 항목에 모두 동의하시나요?",
        target_paths=[
            "consent.unique_id_collection_agreed",
            "consent.personal_credit_collection_agreed",
            "consent.unique_id_provision_agreed",
            "consent.personal_credit_provision_agreed",
        ],
    )
    provider = FakeProvider(
        {
            "patches": [
                {"path": "consent.unique_id_collection_agreed", "value": True},
                {"path": "consent.personal_credit_collection_agreed", "value": True},
                {"path": "consent.unique_id_provision_agreed", "value": True},
                {"path": "consent.personal_credit_provision_agreed", "value": True},
            ]
        }
    )

    patches = extract_answer_patches(question, "네, 모두 동의합니다.", provider)

    assert {patch.path for patch in patches} == set(question.target_paths)
    assert all(patch.value is True for patch in patches)

    case = apply_patches(RecoveryCase.new("CASE-CONSENT"), patches)
    assert case.consent.unique_id_collection_agreed.value is True
    assert case.consent.personal_credit_collection_agreed.value is True
    assert case.consent.unique_id_provision_agreed.value is True
    assert case.consent.personal_credit_provision_agreed.value is True


def test_evidence_question_allows_official_evidence_status_and_note_paths():
    question = Question(
        question_id="evidence.current_items",
        category=QuestionCategory.EVIDENCE,
        prompt="현재 가지고 있는 증빙자료를 알려주세요.",
        target_paths=["evidence"],
    )
    provider = FakeProvider(
        """```json
        {
          "patches": [
            {"path":"evidence.id_card_copy.status","value":"available","source_text":"신분증 사본 있어요"},
            {"path":"evidence.police_certificate.status","value":"planned","source_text":"경찰 확인원은 추후 제출"},
            {"path":"evidence.id_card_copy.note","value":"주민등록증 앞면 사본 보유"},
            {"path":"transactions.0.amount_krw","value":1000000}
          ]
        }
        ```"""
    )

    patches = extract_answer_patches(question, "신분증 사본은 있고 경찰 확인원은 추후 제출할게요.", provider)
    by_path = {patch.path: patch for patch in patches}

    assert by_path["evidence.id_card_copy.status"].value == EvidenceStatus.AVAILABLE
    assert by_path["evidence.police_certificate.status"].value == EvidenceStatus.PLANNED
    assert by_path["evidence.id_card_copy.note"].value == "주민등록증 앞면 사본 보유"
    assert "transactions.0.amount_krw" not in by_path

    case = apply_patches(RecoveryCase.new("CASE-EVIDENCE-ANSWER"), patches)
    evidence = {item.kind: item for item in case.evidence}
    assert evidence["id_card_copy"].status == EvidenceStatus.AVAILABLE
    assert evidence["id_card_copy"].note == "주민등록증 앞면 사본 보유"
    assert evidence["police_certificate"].status == EvidenceStatus.PLANNED


def test_not_applicable_answer_is_preserved_and_not_reinterpreted():
    question = Question(
        question_id="applicant.corporate",
        category=QuestionCategory.APPLICANT,
        prompt="법인 신청에 해당하면 법인명과 사업자등록번호를 알려주세요.",
        target_paths=["applicant.company_name", "applicant.business_number"],
    )
    provider = FakeProvider(
        {
            "patches": [
                {"path": "applicant.company_name", "status": "not_applicable", "source_text": "개인 신청이라 해당 없어요"},
                {"path": "applicant.business_number", "status": "not_applicable", "source_text": "개인 신청이라 해당 없어요"},
            ]
        }
    )

    patches = extract_answer_patches(question, "개인 신청이라 해당 없어요.", provider)

    assert len(patches) == 2
    assert all(patch.status == FieldStatus.NOT_APPLICABLE for patch in patches)
    assert all(patch.value is None for patch in patches)
