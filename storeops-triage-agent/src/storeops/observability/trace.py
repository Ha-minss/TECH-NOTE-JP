"""Trace record contract for workflow runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ToolTraceResult:
    tool_name: str
    status: str


@dataclass(frozen=True)
class EstimatedCost:
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0


@dataclass(frozen=True)
class LatencyBreakdown:
    total: int = 0
    retrieval: int = 0
    planning: int = 0
    tool_execution: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    case_id: str
    scenario_id: str
    retrieved_policy_ids: list[str]
    planned_tools: list[str]
    tool_results: list[ToolTraceResult]
    evidence_ids: list[str]
    final_state: str
    final_cause: str | None
    latency_ms: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    estimated_cost: EstimatedCost = field(default_factory=EstimatedCost)

    def as_dict(self) -> dict:
        return asdict(self)


def build_trace_record(result) -> TraceRecord:
    return TraceRecord(
        trace_id=result.state.trace_id,
        case_id=result.state.case_id,
        scenario_id=result.state.scenario_id,
        retrieved_policy_ids=list(result.retrieved_policy_ids),
        planned_tools=[call.tool_name for call in result.plan.planned_tool_calls],
        tool_results=[
            ToolTraceResult(tool_name=response.tool_name, status=response.status.value)
            for response in result.tool_responses
        ],
        evidence_ids=[record.evidence_id for record in result.evidence],
        final_state=result.state.current_state.value,
        final_cause=result.brief.cause.primary_cause,
    )


__all__ = [
    "EstimatedCost",
    "LatencyBreakdown",
    "ToolTraceResult",
    "TraceRecord",
    "build_trace_record",
]
