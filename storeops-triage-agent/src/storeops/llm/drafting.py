from __future__ import annotations

from storeops.llm.guardrails import ensure_confidence, ensure_safe_merchant_response
from storeops.llm.schemas import MerchantResponseDraftSchema


class MerchantResponseDrafter:
    prompt_name = "merchant_response"

    def __init__(self, *, runtime):
        self.runtime = runtime
        self.last_trace = None

    def draft(
        self,
        *,
        state: str,
        primary_cause: str | None,
        confirmed_facts: list[str],
        clarification_questions: list[str],
        fallback_text: str,
    ) -> str:
        output, trace = self.runtime.invoke(
            prompt_name=self.prompt_name,
            payload={
                "state": state,
                "primary_cause": primary_cause,
                "confirmed_facts": confirmed_facts,
                "clarification_questions": clarification_questions,
                "fallback_text": fallback_text,
            },
            schema=MerchantResponseDraftSchema,
            fallback=lambda: fallback_text,
            guard=lambda parsed, payload: (
                ensure_confidence(
                    parsed,
                    minimum=0.5
                    if payload["state"] in {"DEGRADED_REVIEW", "NEEDS_CLARIFICATION"}
                    else 0.55,
                ),
                ensure_safe_merchant_response(parsed, state=payload["state"]),
            ),
        )
        self.last_trace = trace
        if isinstance(output, str):
            return output
        return output.merchant_response


__all__ = ["MerchantResponseDrafter"]
