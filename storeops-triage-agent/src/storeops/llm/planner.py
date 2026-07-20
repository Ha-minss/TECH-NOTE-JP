from __future__ import annotations

from storeops.llm.guardrails import ensure_allowed_data_needs, ensure_confidence
from storeops.llm.schemas import PlannerOutputSchema
from storeops.core.planner import DataNeed, DataNeedPriority, PlannedToolCall, PlannerOutput



def _policy_excerpts(policies: list[object], *, limit: int = 1200) -> list[dict[str, str]]:
    excerpts: list[dict[str, str]] = []

    for policy in policies:
        document_id = str(
            getattr(policy, "document_id", "")
            or getattr(policy, "policy_id", "")
            or getattr(policy, "id", "")
        )
        title = str(
            getattr(policy, "title", "")
            or getattr(policy, "name", "")
            or document_id
        )
        content = str(
            getattr(policy, "content", "")
            or getattr(policy, "text", "")
            or getattr(policy, "body", "")
            or getattr(policy, "chunk_text", "")
            or ""
        )

        excerpts.append(
            {
                "document_id": document_id,
                "title": title,
                "content": content[:limit],
            }
        )

    return excerpts


def _tool_catalog_entries(tool_catalog: object, allowed_data_needs: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    for need in allowed_data_needs:
        tool = None
        try:
            tool = tool_catalog.tool_for_data_need(need)
        except Exception:
            tool = None

        entries.append(
            {
                "data_need": str(need),
                "tool_name": str(
                    getattr(tool, "tool_name", "")
                    or getattr(tool, "name", "")
                    or ""
                ),
                "description": str(getattr(tool, "description", "") or ""),
                "stage": str(getattr(tool, "stage", "") or ""),
            }
        )

    return entries


class LLMPlanner:
    prompt_name = "planner"

    def __init__(self, *, runtime, tool_catalog, fallback_planner):
        self.runtime = runtime
        self.tool_catalog = tool_catalog
        self.fallback_planner = fallback_planner
        self.last_trace = None

    def plan(self, *, query: str, retrieved_policies, parsed_case=None) -> PlannerOutput:
        policies = list(retrieved_policies)
        allowed = list(self.tool_catalog._by_need.keys())
        fallback = lambda: self.fallback_planner.plan(
            query=query,
            retrieved_policies=policies,
        )
        output, trace = self.runtime.invoke(
            prompt_name=self.prompt_name,
            payload={
                "query": query,
                "issue_family": getattr(parsed_case, "issue_family", "payment_approval_failure"),
                "allowed_data_needs": allowed,
                "retrieved_policy_ids": [getattr(policy, "document_id") for policy in policies],
                "retrieved_policy_excerpts": _policy_excerpts(policies),
                "tool_catalog_entries": _tool_catalog_entries(self.tool_catalog, allowed),
                "planner_instruction": (
                    "Use retrieved_policy_excerpts and tool_catalog_entries to select data_needs. "
                    "Do not use expected_state, expected_primary_cause, or required_tool_names."
                ),
                "missing_fields": list(getattr(parsed_case, "missing_fields", [])),
            },
            schema=PlannerOutputSchema,
            fallback=fallback,
            guard=lambda parsed, payload: (
                ensure_confidence(parsed, minimum=0.55),
                ensure_allowed_data_needs(
                    parsed.selected_data_needs,
                    allowed=payload["allowed_data_needs"],
                ),
            ),
        )
        self.last_trace = trace
        if isinstance(output, PlannerOutput):
            return output

        data_needs = []
        seen = set()
        for item in output.selected_data_needs:
            if item.name in seen:
                continue
            seen.add(item.name)
            data_needs.append(
                DataNeed(
                    name=item.name,
                    priority=DataNeedPriority(item.priority),
                    reason=item.reason,
                )
            )
        planned_tool_calls = [
            PlannedToolCall(
                tool_name=self.tool_catalog.tool_for_data_need(data_need.name).tool_name,
                data_need=data_need.name,
                reason=data_need.reason,
                required=data_need.priority == DataNeedPriority.REQUIRED,
            )
            for data_need in data_needs
            if self.tool_catalog.tool_for_data_need(data_need.name).stage != "post_assessment"
        ]
        return PlannerOutput(
            case_type=output.case_type,
            data_needs=data_needs,
            planned_tool_calls=planned_tool_calls,
            clarification_candidates=list(output.clarification_candidates),
            forbidden_actions=output.forbidden_actions
            or ["payment_execution", "refund", "payment_cancellation", "config_mutation", "external_handoff_without_approval"],
            retrieved_policy_ids=[getattr(policy, "document_id") for policy in policies],
        )


__all__ = ["LLMPlanner"]
