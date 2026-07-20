import pytest

pytestmark = pytest.mark.llm
pytest.importorskip("google.genai")

from types import SimpleNamespace
from pathlib import Path

import pytest

from src.recall_agent.infrastructure.gemini_client import (
    GEMINI_MODEL,
    generate_text,
    load_dotenv,
    resolve_api_key,
)


class FakeModels:
    def __init__(self):
        self.request = None

    def generate_content(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text="OK")


def test_generate_text_uses_requested_model():
    models = FakeModels()
    client = SimpleNamespace(models=models)

    result = generate_text("?곌껐 ?뺤씤", api_key="unused", client=client)

    assert result == "OK"
    assert models.request == {
        "model": GEMINI_MODEL,
        "contents": "?곌껐 ?뺤씤",
    }


def test_resolve_api_key_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.recall_agent.infrastructure.gemini_client.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="Gemini API"):
        resolve_api_key(api_key="")


def test_load_dotenv_reads_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / "gemini_client.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")

    load_dotenv(env_file)

    assert resolve_api_key() == "test-key"


