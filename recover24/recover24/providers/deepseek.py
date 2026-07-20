"""DeepSeek provider for Recover24 evaluation.

Uses DeepSeek's OpenAI-compatible /chat/completions API.
Required env:
- DEEPSEEK_API_KEY

Optional env:
- DEEPSEEK_BASE_URL, default: https://api.deepseek.com
- DEEPSEEK_MODEL, default: deepseek-chat
- DEEPSEEK_TEMPERATURE, default: 0.1
- DEEPSEEK_MAX_TOKENS, default: 1024
"""

from __future__ import annotations

import os
from typing import Any

import requests

from recover24.providers.base import LLMProvider


class DeepSeekProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeekProvider.")

        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature if temperature is not None else float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("DEEPSEEK_MAX_TOKENS", "1024"))

    def generate_json(self, prompt: str) -> str:
        url = self._chat_completions_url()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON-producing assistant. "
                        "Return only valid JSON. Do not include markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response shape: {data}") from exc

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"
