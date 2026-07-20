"""Typed workflow contracts shared across the system."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(StrEnum):
    RECEIVED = "RECEIVED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ROUTE_APPROVED = "ROUTE_APPROVED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    DEGRADED_REVIEW = "DEGRADED_REVIEW"
    CONFLICT_REVIEW = "CONFLICT_REVIEW"
    REJECTED = "REJECTED"
    HANDED_OFF = "HANDED_OFF"


class Assessment(StrEnum):
    LIKELY = "likely"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNAVAILABLE = "unavailable"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_tool: str
    source_record_id: str
    fact_type: str
    normalized_value: Any
    observed_at: datetime
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    sensitivity: str = "internal"


class CauseAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_cause: str | None
    assessment: Assessment
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)


class CaseBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: CauseAssessment
    state: WorkflowState
    operator_actions: list[str]
    recommended_route: str | None = None
    merchant_response: str


class CaseState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    trace_id: str
    scenario_id: str
    store_id: str
    merchant_message: str
    current_state: WorkflowState = WorkflowState.RECEIVED
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    trace_id: str
    store_id: str
    status: ToolStatus
    data: list[dict[str, Any]] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    freshness: str = "current"
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None


__all__ = [
    "Assessment",
    "CaseBrief",
    "CaseState",
    "CauseAssessment",
    "EvidenceRecord",
    "ToolError",
    "ToolResponse",
    "ToolStatus",
    "WorkflowState",
]
