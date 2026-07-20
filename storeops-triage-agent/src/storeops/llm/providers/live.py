from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from storeops.llm.models import LLMModelConfig
from storeops.llm.prompt_contracts import prompt_contract_for


@dataclass(frozen=True)
class LiveLLMSettings:
    api_key: str
    model_name: str
    base_url: str
    timeout_seconds: int = 30

    @classmethod
    def from_sources(cls, *, config_path: str | Path | None = None) -> "LiveLLMSettings":
        payload: dict[str, Any] = {}
        candidate = Path(config_path) if config_path else None
        if candidate is not None and candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))

        api_key = os.environ.get("LIVE_LLM_API_KEY") or payload.get("api_key")
        if not api_key:
            raise RuntimeError("Live LLM API key is missing. Set LIVE_LLM_API_KEY or pass --config.")

        model_name = os.environ.get("LIVE_LLM_MODEL") or payload.get("model_name") or "live-json-model"
        base_url = os.environ.get("LIVE_LLM_BASE_URL") or payload.get("base_url")
        if not base_url:
            raise RuntimeError("Live LLM base URL is missing. Set LIVE_LLM_BASE_URL or pass --config.")

        timeout_seconds = int(
            os.environ.get("LIVE_LLM_TIMEOUT_SECONDS")
            or payload.get("timeout_seconds")
            or 30
        )
        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
        )


def _load_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip().lstrip("\ufeff")
    if not text:
        raise RuntimeError("Live LLM returned empty content while a JSON object was required.")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("parsed JSON was not an object")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        try:
            parsed, _ = decoder.raw_decode(candidate)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("raw-decoded JSON was not an object")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    preview = text[:1000].replace("\n", "\\n")
    last_error = errors[-1] if errors else "unknown parse error"
    raise RuntimeError(
        "Live LLM did not return a parseable JSON object. "
        f"last_error={last_error}; raw_response_preview={preview!r}"
    )


class LiveLLMClient:
    """OpenAI-compatible JSON chat-completions adapter for local live demos."""

    def __init__(self, settings: LiveLLMSettings):
        self.settings = settings

    @classmethod
    def from_sources(cls, *, config_path: str | Path | None = None) -> "LiveLLMClient":
        return cls(LiveLLMSettings.from_sources(config_path=config_path))

    def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, Any],
        model: LLMModelConfig,
    ) -> dict[str, Any]:
        selected_model = self.settings.model_name or model.model_name
        user_message = {
            "prompt_name": prompt_name,
            "prompt_contract": prompt_contract_for(prompt_name),
            "payload": payload,
        }
        body = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious workflow AI component. "
                        "Return one valid JSON object only. "
                        "Use only the supplied payload and prompt contract. "
                        "Do not invent tools, policies, causes, or hidden facts."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_message, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Live LLM API error {exc.code}: {error_body[:1000]}") from exc

        choice = raw["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        if not content.strip():
            raw_preview = json.dumps(raw, ensure_ascii=False)[:2000]
            raise RuntimeError(
                "Live LLM returned empty message.content while a JSON object was required. "
                f"finish_reason={choice.get('finish_reason')!r}; raw_response_preview={raw_preview!r}"
            )
        return _load_json_object(content)


__all__ = ["LiveLLMClient", "LiveLLMSettings"]
