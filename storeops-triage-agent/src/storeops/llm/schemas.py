from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LLMCaseParserSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_family: str
    symptoms: list[str] = Field(default_factory=list)
    context_flags: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float
    reasoning_summary: str = ""


class ClarificationQuestionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    question: str
    why_needed: str


class ClarificationOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[ClarificationQuestionSchema] = Field(default_factory=list)
    confidence: float


class PlannerDataNeedSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: str
    reason: str


class PlannerOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_type: str
    selected_data_needs: list[PlannerDataNeedSchema] = Field(default_factory=list)
    clarification_candidates: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    confidence: float



class PolicyCheckSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    policy_title: str | None = None
    check_text: str
    matched_data_need: str | None = None
    priority: str
    reason: str
    source_quote: str | None = None


class EvidenceChecklistOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_checks: list[PolicyCheckSchema] = Field(default_factory=list)
    confidence: float

class MerchantResponseDraftSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant_response: str
    mentions_uncertainty: bool
    contains_unconfirmed_claim: bool
    confidence: float


__all__ = [
    "ClarificationOutputSchema",
    "ClarificationQuestionSchema",
    "EvidenceChecklistOutputSchema",
    "LLMCaseParserSchema",
    "MerchantResponseDraftSchema",
    "PlannerDataNeedSchema",
    "PlannerOutputSchema",
    "PolicyCheckSchema",
]
