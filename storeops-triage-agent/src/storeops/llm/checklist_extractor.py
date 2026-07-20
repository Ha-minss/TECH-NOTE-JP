from __future__ import annotations

from storeops.core.policy_checks import PolicyCheck
from storeops.llm.guardrails import ensure_confidence
from storeops.llm.schemas import EvidenceChecklistOutputSchema


class LLMEvidenceChecklistExtractor:
    prompt_name = "checklist_extractor"

    def __init__(self, *, runtime):
        self.runtime = runtime
        self.last_trace = None

    def extract(self, *, parsed_case, query: str, retrieved_policies, tool_catalog) -> list[PolicyCheck]:
        policies = list(retrieved_policies)
        allowed_data_needs = sorted(tool_catalog._by_need.keys())
        # If extraction fails, return no checks; downstream gates handle missing evidence.
        fallback = lambda: []
        output, trace = self.runtime.invoke(
            prompt_name=self.prompt_name,
            payload={
                "query": query,
                "issue_family": getattr(parsed_case, "issue_family", "payment_approval_failure"),
                "retrieved_policy_excerpts": [
                    {
                        "policy_id": getattr(policy, "document_id", ""),
                        "policy_title": getattr(policy, "title", None),
                        "content": getattr(policy, "content", ""),
                    }
                    for policy in policies
                ],
                "allowed_data_needs": allowed_data_needs,
                "tool_catalog": [
                    {
                        "tool_name": tool.tool_name,
                        "description": tool.description,
                        "provides_data_needs": list(tool.provides_data_needs),
                        "read_only": tool.read_only,
                        "stage": tool.stage,
                    }
                    for tool in tool_catalog.tools
                ],
            },
            schema=EvidenceChecklistOutputSchema,
            fallback=fallback,
            guard=lambda parsed, payload: (
                ensure_confidence(parsed, minimum=0.55),
                self._ensure_allowed_data_needs(parsed.policy_checks, payload["allowed_data_needs"]),
            ),
        )
        self.last_trace = trace
        if isinstance(output, list):
            return output
        return [
            PolicyCheck(
                policy_id=item.policy_id,
                policy_title=item.policy_title,
                check_text=item.check_text,
                matched_data_need=item.matched_data_need,
                priority=item.priority,  # type: ignore[arg-type]
                reason=item.reason,
                source_quote=item.source_quote,
            )
            for item in output.policy_checks
        ]

    @staticmethod
    def _ensure_allowed_data_needs(policy_checks, allowed_data_needs: list[str]) -> None:
        allowed = set(allowed_data_needs)
        invalid = [
            item.matched_data_need
            for item in policy_checks
            if item.matched_data_need is not None and item.matched_data_need not in allowed
        ]
        if invalid:
            raise ValueError(f"invalid checklist data_needs: {', '.join(invalid)}")


__all__ = ["LLMEvidenceChecklistExtractor"]