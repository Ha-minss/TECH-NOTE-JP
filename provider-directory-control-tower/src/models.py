from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderRecord(BaseModel):
    provider_id: str
    provider_name: str
    npi: str = ""
    specialty: str = ""
    practice_name: str = ""
    address: str = ""
    phone: str = ""
    website: Optional[str] = None
    last_verified_date: str = ""
    active_status: str = "active"


class EvidenceItem(BaseModel):
    source_name: str
    source_type: str
    source_url: Optional[str] = None
    field: str
    value: str
    normalized_value: str = ""
    evidence_text: str = ""
    source_confidence: float = 0.5
    collected_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FieldChange(BaseModel):
    field: str
    old_value: str
    new_value: str
    confidence_score: float
    supporting_sources: List[str]
    evidence_urls: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    source_conflict: bool = False
    requires_human_review: bool = False


class Recommendation(BaseModel):
    provider_id: str
    npi: str
    change_detected: bool
    changes: List[FieldChange]
    overall_confidence: float
    recommended_action: str
    reason: str
    audit_id: str


class AuditEvent(BaseModel):
    audit_id: str
    provider_id: str
    node: str
    status: str
    detail: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
