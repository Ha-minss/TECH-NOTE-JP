from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class RouterRoute(str, Enum):
    H07_CANDIDATE = "H07_CANDIDATE"
    NEEDS_PRODUCT_VERIFICATION = "NEEDS_PRODUCT_VERIFICATION"
    OTHER_OR_MANUAL_REVIEW = "OTHER_OR_MANUAL_REVIEW"


class RouterResult(BaseModel):
    """Validated output contract for the LLM complaint router.

    The LLM may suggest a route, confidence, and reason.
    It must not decide final financial execution.
    Final rule execution is decided later by deterministic guards.
    """

    route: RouterRoute
    confidence: float = Field(ge=0.0, le=1.0)
    needs_product_verification: bool
    manual_review_required: bool
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def reason_must_be_plain_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reason must not be empty")
        return cleaned

    @field_validator("manual_review_required")
    @classmethod
    def manual_review_for_non_h07_is_allowed(cls, value: bool) -> bool:
        return value


class RouterValidationFailure(BaseModel):
    route: RouterRoute = RouterRoute.OTHER_OR_MANUAL_REVIEW
    confidence: float = 0.0
    needs_product_verification: bool = False
    manual_review_required: bool = True
    reason: str = "Router output failed schema validation."


def parse_router_result(raw: dict[str, Any]) -> RouterResult:
    """Parse and validate LLM router output.

    Raises pydantic.ValidationError if the shape is invalid.
    Use safe_parse_router_result when invalid output should fall back to manual review.
    """

    return RouterResult.model_validate(raw)


def safe_parse_router_result(raw: Any) -> RouterResult:
    """Return a safe manual-review result when LLM output is invalid."""

    try:
        if not isinstance(raw, dict):
            raise TypeError("router output must be a dict")
        return RouterResult.model_validate(raw)
    except (ValidationError, TypeError, ValueError):
        return RouterResult(
            route=RouterRoute.OTHER_OR_MANUAL_REVIEW,
            confidence=0.0,
            needs_product_verification=False,
            manual_review_required=True,
            reason="Router output failed schema validation.",
        )
