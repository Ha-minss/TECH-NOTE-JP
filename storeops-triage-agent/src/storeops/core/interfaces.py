from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from storeops.core.types import ParsedCase, SafetyDecision
from storeops.core.planner import PlannerOutput
from storeops.core.contracts import CauseAssessment, EvidenceRecord, ToolResponse


@runtime_checkable
class CaseParser(Protocol):
    def parse(
        self,
        merchant_message: str,
        *,
        store_id: str,
        case_hint: str | None = None,
    ) -> ParsedCase: ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    def execute(
        self,
        *,
        store_id: str,
        plan: PlannerOutput,
    ) -> list[ToolResponse]: ...


@runtime_checkable
class EvidenceBuilder(Protocol):
    def build(
        self,
        *,
        scenario_id: str,
        tool_responses: Iterable[ToolResponse],
    ) -> list[EvidenceRecord]: ...


@runtime_checkable
class EvidenceReasoner(Protocol):
    def reason(
        self,
        *,
        evidence: list[EvidenceRecord],
        parsed_case: ParsedCase,
    ) -> CauseAssessment: ...


@runtime_checkable
class SafetyGateProtocol(Protocol):
    def apply(
        self,
        *,
        parsed_case: ParsedCase,
        planned_required_tools: list[str],
        tool_responses: list[ToolResponse],
        evidence: list[EvidenceRecord],
        cause_assessment: CauseAssessment,
    ) -> SafetyDecision: ...


__all__ = [
    "CaseParser",
    "ToolExecutorProtocol",
    "EvidenceBuilder",
    "EvidenceReasoner",
    "SafetyGateProtocol",
]
