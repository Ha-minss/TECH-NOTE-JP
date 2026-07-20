"""Gemma Colab provider.

Calls a Colab/FastAPI endpoint exposed through Cloudflare Tunnel.

The current Recover24 Colab server expects:
  POST /generate
  {"userText": "..."}

For newer servers that support rawPrompt, this provider sends rawPrompt=True so
V3's own prompt is not wrapped a second time by the server.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from .base import LLMProvider


class GemmaColabProvider(LLMProvider):
    """Provider adapter for Gemma running in Colab/FastAPI."""

    def __init__(self, base_url: str | None = None, timeout_seconds: int = 60) -> None:
        url = base_url or os.getenv("RECOVER24_GEMMA_COLAB_URL", "")
        if not url:
            raise ValueError("GemmaColabProvider requires base_url or RECOVER24_GEMMA_COLAB_URL.")
        self.base_url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str) -> str:
        response = requests.post(
            self._endpoint_url(),
            json={"userText": prompt, "rawPrompt": True},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 422:
            # Backward compatibility for an older Colab cell that expected
            # {"prompt": ...}. The preferred contract is userText/rawPrompt.
            response = requests.post(
                self._endpoint_url(),
                json={"prompt": prompt},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return self._extract_text(response)

    def _extract_text(self, response: requests.Response) -> str:
        try:
            data: Any = response.json()
        except ValueError:
            return response.text

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            if "result" in data:
                result = data["result"]
                if isinstance(result, str):
                    return result
                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False)

            for key in ("text", "response", "generated_text", "output", "answer"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]
                    if isinstance(first.get("text"), str):
                        return first["text"]

        return json.dumps(data, ensure_ascii=False)

    def _endpoint_url(self) -> str:
        if self.base_url.endswith("/generate"):
            return self.base_url
        return f"{self.base_url}/generate"
