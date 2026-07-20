from __future__ import annotations

from storeops.core.interfaces import CaseParser
from storeops.core.types import ParsedCase
from storeops.llm.guardrails import ensure_confidence, ensure_issue_family
from storeops.llm.schemas import LLMCaseParserSchema


class LLMCaseParser:
    prompt_name = "case_parser"

    def __init__(self, *, runtime, fallback_parser: CaseParser):
        self.runtime = runtime
        self.fallback_parser = fallback_parser
        self.last_trace = None

    def parse(
        self,
        merchant_message: str,
        *,
        store_id: str,
        case_hint: str | None = None,
    ) -> ParsedCase:
        fallback = lambda: self.fallback_parser.parse(
            merchant_message,
            store_id=store_id,
            case_hint=case_hint,
        )
        output, trace = self.runtime.invoke(
            prompt_name=self.prompt_name,
            payload={
                "merchant_message": merchant_message,
                "store_id": store_id,
                "case_hint": case_hint or "",
            },
            schema=LLMCaseParserSchema,
            fallback=fallback,
            guard=lambda parsed, payload: (
                ensure_confidence(parsed, minimum=0.55),
                ensure_issue_family(parsed.issue_family),
            ),
        )
        self.last_trace = trace
        if isinstance(output, ParsedCase):
            return output
        expanded = " ".join(
            [
                merchant_message,
                case_hint or "",
                output.issue_family,
                " ".join(output.symptoms),
                " ".join(output.context_flags),
            ]
        )
        return ParsedCase(
            store_id=store_id,
            merchant_message=merchant_message,
            issue_family=output.issue_family,
            symptoms=output.symptoms or ["unknown_payment_problem"],
            context_flags=output.context_flags,
            missing_fields=output.missing_fields,
            retrieval_query=expanded,
            planner_query=expanded,
        )


__all__ = ["LLMCaseParser"]
