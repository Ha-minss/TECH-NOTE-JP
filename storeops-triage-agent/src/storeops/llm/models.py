from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModelConfig:
    model_name: str
    temperature: float = 0.0
    timeout_ms: int = 3000


__all__ = ["LLMModelConfig"]
