from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from storeops.core.planner import PlannerOutput
from storeops.core.contracts import CaseBrief, CaseState, CauseAssessment, EvidenceRecord, ToolResponse, WorkflowState


@dataclass(frozen=True)
class ParsedCase:
    store_id: str
    merchant_message: str
    issue_family: str
    symptoms: list[str]
    context_flags: list[str]
    missing_fields: list[str]
    retrieval_query: str
    planner_query: str


@dataclass(frozen=True)
class SafetyDecision:
    state: WorkflowState
    cause: CauseAssessment


@dataclass
class WorkflowResult:
    state: CaseState
    parsed_case: ParsedCase
    retrieved_policy_ids: list[str]
    plan: PlannerOutput
    tool_responses: list[ToolResponse]
    evidence: list[EvidenceRecord]
    brief: CaseBrief
    clarification_questions: list[str] = field(default_factory=list)
    drafted_merchant_response: str | None = None
    llm_traces: list[Any] = field(default_factory=list)
    policy_checks: list[Any] = field(default_factory=list)
    policy_check_trace: list[dict[str, Any]] = field(default_factory=list)



__all__ = ["ParsedCase", "SafetyDecision", "WorkflowResult"]

