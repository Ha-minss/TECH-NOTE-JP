from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from storeops.llm.client import LLMClient
from storeops.llm.models import LLMModelConfig


@dataclass(frozen=True)
class LLMCallTrace:
    prompt_name: str
    model_name: str
    status: str
    latency_ms: int
    used_fallback: bool
    error_message: str | None = None



def _normalize_checklist_output(prompt_name: str, raw: object) -> object:
    """Normalize common LLM envelope mistakes for checklist extraction.

    This does not create policy checks from keywords or expected labels. It only
    wraps structurally valid checklist objects into the schema envelope expected
    by EvidenceChecklistOutputSchema.
    """
    if prompt_name != "checklist_extractor" or not isinstance(raw, dict):
        return raw

    if isinstance(raw.get("policy_checks"), list):
        normalized = dict(raw)
        normalized.setdefault("confidence", 0.8)
        return normalized

    for list_key in ("checks", "checklist", "items"):
        if isinstance(raw.get(list_key), list):
            return {
                "confidence": raw.get("confidence", 0.8),
                "policy_checks": raw[list_key],
            }

    check_keys = {
        "policy_id",
        "policy_title",
        "check_text",
        "matched_data_need",
        "priority",
        "reason",
        "source_quote",
    }

    if any(key in raw for key in check_keys):
        policy_check = {key: raw.get(key) for key in check_keys}
        return {
            "confidence": raw.get("confidence", 0.8),
            "policy_checks": [policy_check],
        }

    return raw


class LLMRuntime:
    def __init__(self, *, client: LLMClient, model_name: str):
        self.client = client
        self.model = LLMModelConfig(model_name=model_name)

    def invoke(
        self,
        *,
        prompt_name: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        fallback,
        guard=None,
    ):
        started = perf_counter()
        try:
            raw = self.client.generate_json(
                prompt_name=prompt_name,
                payload=payload,
                model=self.model,
            )
            raw = _normalize_checklist_output(prompt_name, raw)
            parsed = schema.model_validate(raw)
            if guard is not None:
                guard(parsed, payload)
            trace = LLMCallTrace(
                prompt_name=prompt_name,
                model_name=self.model.model_name,
                status="success",
                latency_ms=int((perf_counter() - started) * 1000),
                used_fallback=False,
            )
            return parsed, trace
        except Exception as exc:  # noqa: BLE001 - fallback boundary
            trace = LLMCallTrace(
                prompt_name=prompt_name,
                model_name=self.model.model_name,
                status="fallback",
                latency_ms=int((perf_counter() - started) * 1000),
                used_fallback=True,
                error_message=str(exc),
            )
            return fallback(), trace


__all__ = ["LLMCallTrace", "LLMRuntime"]
