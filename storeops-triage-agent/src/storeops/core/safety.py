"""Core deterministic safety gate implementation."""

from __future__ import annotations

from storeops.core.types import SafetyDecision
from storeops.core.contracts import Assessment, CauseAssessment, ToolStatus, WorkflowState


class SafetyGate:
    """Generic deterministic safety transitions."""

    def apply(
        self,
        *,
        parsed_case,
        planned_required_tools: list[str],
        tool_responses,
        evidence,
        cause_assessment: CauseAssessment | dict,
    ) -> SafetyDecision:
        cause = (
            cause_assessment
            if isinstance(cause_assessment, CauseAssessment)
            else CauseAssessment(**cause_assessment)
        )

        failed_required = [
            response.tool_name
            for response in tool_responses
            if response.tool_name in planned_required_tools
            and response.status in {ToolStatus.ERROR, ToolStatus.PARTIAL}
        ]
        if failed_required:
            return SafetyDecision(
                state=WorkflowState.DEGRADED_REVIEW,
                cause=cause.model_copy(
                    update={
                        "primary_cause": None,
                        "assessment": Assessment.UNAVAILABLE,
                        "supporting_evidence_ids": [],
                        "missing_evidence": sorted(set(cause.missing_evidence + failed_required)),
                    }
                ),
            )

        if cause.contradicting_evidence_ids:
            return SafetyDecision(state=WorkflowState.CONFLICT_REVIEW, cause=cause)

        if cause.assessment is Assessment.LIKELY and not cause.supporting_evidence_ids:
            return SafetyDecision(
                state=WorkflowState.DEGRADED_REVIEW,
                cause=cause.model_copy(
                    update={
                        "primary_cause": None,
                        "assessment": Assessment.UNAVAILABLE,
                        "missing_evidence": sorted(
                            set(cause.missing_evidence + ["supporting_evidence_id"])
                        ),
                    }
                ),
            )

        if cause.primary_cause is None and getattr(parsed_case, "missing_fields", []):
            return SafetyDecision(state=WorkflowState.NEEDS_CLARIFICATION, cause=cause)

        if cause.primary_cause is None:
            return SafetyDecision(state=WorkflowState.DEGRADED_REVIEW, cause=cause)

        return SafetyDecision(state=WorkflowState.READY_FOR_REVIEW, cause=cause)


__all__ = ["SafetyDecision", "SafetyGate"]
