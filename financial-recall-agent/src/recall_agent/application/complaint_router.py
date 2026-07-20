from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pydantic import ValidationError

from src.recall_agent.application.router_schema import RouterResult, safe_parse_router_result


@dataclass(frozen=True)
class RouterOutput:
    """Validated router output plus validation metadata.

    LLM routers may produce invalid raw JSON. The evaluator needs to know
    whether schema validation passed, while still receiving a safe fallback
    RouterResult for downstream scoring.
    """

    result: RouterResult
    schema_valid: bool
    raw_output: Any | None = None


class ComplaintRouter(Protocol):
    """Common interface for mock, Gemini, OpenAI, or any future router."""

    name: str

    def route(self, record: Mapping[str, Any]) -> RouterOutput:
        """Route one complaint record into a validated RouterResult."""
        ...


def make_router_output(raw: Any) -> RouterOutput:
    """Validate raw router output and attach schema validity metadata."""

    try:
        result = RouterResult.model_validate(raw)
        return RouterOutput(result=result, schema_valid=True, raw_output=raw)
    except (ValidationError, TypeError, ValueError):
        return RouterOutput(
            result=safe_parse_router_result(raw),
            schema_valid=False,
            raw_output=raw,
        )


def build_llm_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the fields the router is allowed to see.

    Evaluation labels such as expected_router, expected_guard, case_type,
    and internal lookup fields must not be included in router input.
    """

    allowed = record.get("llm_allowed_input_fields") or [
        "channel",
        "product_id_claimed",
        "complaint_text",
    ]
    return {field: record.get(field) for field in allowed}
