import os
from pathlib import Path
from typing import Any

from google import genai


GEMINI_MODEL = "gemini-3.1-flash-lite"


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_api_key(api_key: str | None = None) -> str:
    load_dotenv()
    resolved = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not resolved:
        raise ValueError("Gemini API 키를 입력하거나 GEMINI_API_KEY 환경변수를 설정하세요.")
    return resolved


def generate_text(
    prompt: str,
    *,
    api_key: str | None = None,
    client: Any | None = None,
) -> str:
    if not prompt.strip():
        raise ValueError("Gemini에 전달할 프롬프트가 비어 있습니다.")

    gemini = client or genai.Client(api_key=resolve_api_key(api_key))
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text or ""


def test_connection(api_key: str | None = None) -> str:
    return generate_text(
        "연결 확인용 요청입니다. 반드시 OK만 출력하세요.",
        api_key=api_key,
    )
