from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from storeops.llm.models import LLMModelConfig


class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, Any],
        model: LLMModelConfig,
    ) -> dict[str, Any]:
        ...


class ScriptedLLMClient:
    """Small deterministic client used for tests and offline demos."""

    def __init__(self, script: Mapping[str, list[dict[str, Any] | Exception]]):
        self._script = {name: list(items) for name, items in script.items()}

    def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, Any],
        model: LLMModelConfig,
    ) -> dict[str, Any]:
        queue = self._script.get(prompt_name, [])
        if not queue:
            raise KeyError(f"No scripted response for prompt: {prompt_name}")
        item = queue.pop(0)
        self._script[prompt_name] = queue
        if isinstance(item, Exception):
            raise item
        return item


__all__ = ["LLMClient", "ScriptedLLMClient"]
