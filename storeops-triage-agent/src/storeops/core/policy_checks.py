from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from storeops.core.planner import (
    DataNeed,
    DataNeedPriority,
    PlannedToolCall,
    PlannerOutput,
    ToolCatalog,
)


PolicyCheckPriority = Literal["required", "supporting", "optional"]


@dataclass(frozen=True)
class PolicyCheck:
    policy_id: str
    policy_title: str | None
    check_text: str
    matched_data_need: str | None
    priority: PolicyCheckPriority
    reason: str
    source_quote: str | None = None


class EvidencePlanBuilder:
    def build(
        self,
        *,
        policy_checks: list[PolicyCheck],
        tool_catalog: ToolCatalog,
        retrieved_policy_ids: list[str],
    ) -> tuple[PlannerOutput, list[dict[str, object]]]:
        data_needs: list[DataNeed] = []
        tool_calls: list[PlannedToolCall] = []
        trace: list[dict[str, object]] = []
        seen_data_needs: set[str] = set()
        seen_tools: set[str] = set()

        for check in policy_checks:
            trace_item = self._trace_item(check)
            if check.matched_data_need is None:
                trace_item["source"] = "unmatched_policy_check"
                trace_item["tool_name"] = None
                trace.append(trace_item)
                continue
            try:
                tool = tool_catalog.tool_for_data_need(check.matched_data_need)
            except KeyError:
                trace_item["source"] = "unmatched_policy_check"
                trace_item["tool_name"] = None
                trace.append(trace_item)
                continue
            if tool.stage == "post_assessment":
                trace_item["source"] = "unmatched_policy_check"
                trace_item["tool_name"] = None
                trace.append(trace_item)
                continue

            required = check.priority == "required"
            trace_item["source"] = "checklist_extractor"
            trace_item["tool_name"] = tool.tool_name
            trace.append(trace_item)

            if check.matched_data_need not in seen_data_needs:
                seen_data_needs.add(check.matched_data_need)
                data_needs.append(
                    DataNeed(
                        name=check.matched_data_need,
                        priority=DataNeedPriority(check.priority),
                        reason=check.reason,
                    )
                )
            if tool.tool_name not in seen_tools:
                seen_tools.add(tool.tool_name)
                tool_calls.append(
                    PlannedToolCall(
                        tool_name=tool.tool_name,
                        data_need=check.matched_data_need,
                        reason=check.reason,
                        required=required,
                    )
                )
            elif required:
                tool_calls = [
                    PlannedToolCall(
                        tool_name=call.tool_name,
                        data_need=call.data_need,
                        reason=call.reason,
                        required=True if call.tool_name == tool.tool_name else call.required,
                    )
                    for call in tool_calls
                ]

        return (
            PlannerOutput(
                case_type="policy_checklist_evidence_plan",
                data_needs=data_needs,
                planned_tool_calls=tool_calls,
                clarification_candidates=[],
                forbidden_actions=[
                    "payment_execution",
                    "refund",
                    "payment_cancellation",
                    "config_mutation",
                    "external_handoff_without_approval",
                ],
                retrieved_policy_ids=retrieved_policy_ids,
            ),
            trace,
        )

    @staticmethod
    def _trace_item(check: PolicyCheck) -> dict[str, object]:
        return {
            "policy_id": check.policy_id,
            "policy_title": check.policy_title,
            "check_text": check.check_text,
            "matched_data_need": check.matched_data_need,
            "priority": check.priority,
            "reason": check.reason,
            "source_quote": check.source_quote,
        }


__all__ = ["EvidencePlanBuilder", "PolicyCheck", "PolicyCheckPriority"]
