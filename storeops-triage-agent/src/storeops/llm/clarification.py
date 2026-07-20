from __future__ import annotations

from dataclasses import dataclass

from storeops.llm.guardrails import ensure_confidence, filter_clarification_questions
from storeops.llm.schemas import ClarificationOutputSchema


@dataclass(frozen=True)
class ClarificationQuestion:
    field: str
    question: str
    why_needed: str


class ClarificationQuestionGenerator:
    prompt_name = "clarification"

    def __init__(self, *, runtime):
        self.runtime = runtime
        self.last_trace = None

    def generate(self, *, parsed_case) -> list[ClarificationQuestion]:
        missing_fields = list(parsed_case.missing_fields)
        if not missing_fields:
            self.last_trace = None
            return []

        fallback = lambda: self._fallback_questions(missing_fields)
        output, trace = self.runtime.invoke(
            prompt_name=self.prompt_name,
            payload={
                "merchant_message": parsed_case.merchant_message,
                "issue_family": parsed_case.issue_family,
                "known_missing_fields": missing_fields,
            },
            schema=ClarificationOutputSchema,
            fallback=fallback,
            guard=lambda parsed, payload: ensure_confidence(parsed, minimum=0.55),
        )
        self.last_trace = trace
        if isinstance(output, list):
            return output

        filtered = filter_clarification_questions(
            output.questions,
            known_missing_fields=missing_fields,
        )
        if not filtered:
            return self._fallback_questions(missing_fields)
        return [
            ClarificationQuestion(
                field=question.field,
                question=question.question,
                why_needed=question.why_needed,
            )
            for question in filtered
        ]

    @staticmethod
    def _fallback_questions(missing_fields: list[str]) -> list[ClarificationQuestion]:
        templates = {
            "failed_physical_terminal": ClarificationQuestion(
                field="failed_physical_terminal",
                question="오류가 발생한 단말기가 어떤 기기인지 알려주세요.",
                why_needed="문제가 발생한 단말기를 특정해야 시스템 기록과 비교할 수 있습니다.",
            ),
            "visible_error_message": ClarificationQuestion(
                field="visible_error_message",
                question="단말기 화면에 표시된 오류 문구를 알려주세요.",
                why_needed="오류 유형을 구분해야 관련 기록을 더 정확히 확인할 수 있습니다.",
            ),
            "error_time": ClarificationQuestion(
                field="error_time",
                question="오류가 발생한 대략적인 시각을 알려주세요.",
                why_needed="동일 시간대 승인 실패 기록과 비교해야 합니다.",
            ),
            "payment_method": ClarificationQuestion(
                field="payment_method",
                question="어떤 결제 수단에서 실패했는지 알려주세요.",
                why_needed="문제가 특정 결제 수단에 한정되는지 확인해야 합니다.",
            ),
        }
        return [templates[field] for field in missing_fields if field in templates][:2]


__all__ = ["ClarificationQuestion", "ClarificationQuestionGenerator"]
