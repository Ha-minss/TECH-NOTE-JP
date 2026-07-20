"""Tests for application-level Recover24 workflow orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from application.run_recover24_case import answer_question, render_case_html, start_case, write_case_artifacts
from recover24.models import Question, QuestionCategory


class FixedProvider:
    def __init__(self, payloads: list[dict[str, Any]]):
        self.payloads = list(payloads)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        assert self.payloads, "No fixed provider payload left"
        return self.payloads.pop(0)


def test_start_case_orchestrates_extraction_patching_questions_and_readiness(tmp_path: Path):
    provider = FixedProvider([
        {
            "patches": [
                {"path": "applicant.name", "value": "김영희", "status": "answered"},
                {"path": "transactions.0.amount_krw", "value": 1000000, "status": "answered"},
            ]
        }
    ])

    result = start_case("김영희가 100만원 피해를 봤어요.", provider, case_id="CASE-APP-001")

    assert result.case.case_id == "CASE-APP-001"
    assert result.case.applicant.name.value == "김영희"
    assert result.case.transactions[0].amount_krw.value == 1_000_000
    assert result.questions
    assert result.readiness.can_render_document is True
    assert result.document_view.to_dict()["applicant"]["name"] == "김영희"


def test_answer_question_applies_scoped_followup_patches():
    start_provider = FixedProvider([{"patches": []}])
    result = start_case("피해를 봤어요.", start_provider, case_id="CASE-APP-002")

    question = Question(
        question_id="transaction.primary_details",
        category=QuestionCategory.TRANSACTION,
        prompt="수취인 이름과 송금시각을 알려주세요.",
        target_paths=["transactions.0.destination_account_holder", "transactions.0.transferred_at"],
    )
    answer_provider = FixedProvider([
        {
            "patches": [
                {"path": "transactions.0.destination_account_holder", "value": "김철수", "status": "answered"},
                {"path": "transactions.0.transferred_at", "value": "2026-06-20 14:00", "status": "answered"},
                {"path": "investigation.status", "value": "not_reported", "status": "answered"},
            ]
        }
    ])

    updated = answer_question(result.case, question, "김철수에게 2026-06-20 14시에 보냈어요.", answer_provider)

    assert updated.case.transactions[0].destination_account_holder.value == "김철수"
    assert updated.case.transactions[0].transferred_at.value == "2026-06-20 14:00"
    # Out-of-scope investigation.status must be discarded by answers.py.
    assert updated.case.investigation.status.value is None


def test_render_and_write_case_artifacts(tmp_path: Path):
    provider = FixedProvider([{"patches": [{"path": "applicant.name", "value": "박동하", "status": "answered"}]}])
    result = start_case("박동하입니다.", provider, case_id="CASE-APP-003")

    html = render_case_html(result.case)
    assert "박동하" in html
    assert "전자금융거래 사고 피해 신고서" in html

    paths = write_case_artifacts(result, output_dir=tmp_path)
    assert paths["case_json"].exists()
    assert paths["readiness_json"].exists()
    assert paths["html"].exists()
