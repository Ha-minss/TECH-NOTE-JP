"""LLM-based statement fact extraction for consistency evaluation.

The LLM only extracts candidate facts from the free-form statement.
It does NOT decide whether there is a conflict. The deterministic
conflict_checker compares these facts against form_facts.
"""

from __future__ import annotations

from typing import Any

from evaluation.llm_json import coerce_json
from evaluation.normalization.normalizer import normalize_date_text
from recover24.providers.base import LLMProvider

from .extractor import extract_statement_facts as extract_statement_facts_rule

STATUS_ALLOWED = {
    "police_status": {"reported", "not_reported", "unknown"},
    "freeze_status": {"not_requested", "requested", "completed", "attempted_but_failed", "unknown"},
    "refund_status": {"not_started", "requested", "completed", "unknown"},
}


def extract_statement_facts_llm(
    raw_statement: str,
    provider: LLMProvider,
    *,
    fallback_to_rule: bool = True,
) -> dict[str, Any]:
    facts, _meta = extract_statement_facts_llm_with_meta(
        raw_statement,
        provider,
        fallback_to_rule=fallback_to_rule,
    )
    return facts


def extract_statement_facts_llm_with_meta(
    raw_statement: str,
    provider: LLMProvider | None,
    *,
    fallback_to_rule: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if provider is None:
        return extract_statement_facts_rule(raw_statement), {
            "extractor": "rule",
            "llm_available": False,
            "fallback_used": False,
            "parse_error": None,
        }

    prompt = _build_prompt(raw_statement)

    try:
        raw = provider.generate_json(prompt)
    except Exception as exc:
        if fallback_to_rule:
            return extract_statement_facts_rule(raw_statement), {
                "extractor": "rule_fallback",
                "llm_available": True,
                "fallback_used": True,
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
        raise

    data = coerce_json(raw)
    if isinstance(data, dict) and isinstance(data.get("statement_facts"), dict):
        data = data["statement_facts"]

    if not isinstance(data, dict):
        if fallback_to_rule:
            return extract_statement_facts_rule(raw_statement), {
                "extractor": "rule_fallback",
                "llm_available": True,
                "fallback_used": True,
                "parse_error": "LLM output was not a JSON object.",
                "raw_output": str(raw)[:500],
            }
        return {}, {
            "extractor": "llm",
            "llm_available": True,
            "fallback_used": False,
            "parse_error": "LLM output was not a JSON object.",
            "raw_output": str(raw)[:500],
        }

    cleaned = _clean_statement_facts(data)
    return cleaned, {
        "extractor": "llm",
        "llm_available": True,
        "fallback_used": False,
        "parse_error": None,
    }


def _build_prompt(raw_statement: str) -> str:
    return f"""
당신은 금융사고 자유진술에서 후보 사실만 추출하는 정보추출기입니다.
충돌 여부를 판단하지 마세요. 폼 값과 비교하지 마세요.
자유진술에 명시된 사실만 JSON으로 추출하세요.
불명확하거나 언급이 없으면 그 필드는 "unknown"으로 두세요.

반환 JSON은 반드시 아래 형식만 사용하세요.
{{
  "statement_facts": {{
    "damage_amount_krw": 30000000 또는 "unknown",
    "incident_date": "YYYY-MM-DD" 또는 "unknown",
    "transfer_date": "YYYY-MM-DD" 또는 "unknown",
    "police_status": "reported" | "not_reported" | "unknown",
    "freeze_status": "not_requested" | "requested" | "completed" | "attempted_but_failed" | "unknown",
    "refund_status": "not_started" | "requested" | "completed" | "unknown",
    "recipient_account_known": true | false | "unknown"
  }}
}}

해석 규칙:
- "경찰에 신고"는 police_status="reported".
- "경찰 미신고", "신고하지 않음"은 police_status="not_reported".
- "지급정지 요청/신청"은 freeze_status="requested".
- "지급정지 완료"는 freeze_status="completed".
- "지급정지 실패", "악성앱 때문에 은행 연락 실패"는 freeze_status="attempted_but_failed".
- "환급 완료"는 refund_status="completed".
- "환급 신청/요청"은 refund_status="requested".
- "환급 전", "아직 환급 안 됨"은 refund_status="not_started".
- "상대 계좌 모름"은 recipient_account_known=false.
- 날짜는 가능하면 YYYY-MM-DD로 정규화하세요.
- 금액은 원화 정수로 변환하세요. 예: 1750만원 → 17500000, 1,791만 5천원 → 17915000.

자유진술:
{raw_statement}
""".strip()


def _clean_statement_facts(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    amount = _normalize_amount(data.get("damage_amount_krw"))
    if amount is not None:
        cleaned["damage_amount_krw"] = amount

    for field in ("incident_date", "transfer_date"):
        value = data.get(field)
        if isinstance(value, str) and value.strip().lower() not in {"", "unknown", "null", "none", "n/a"}:
            normalized = normalize_date_text(value.strip())
            if normalized != "unknown":
                cleaned[field] = normalized

    for field, allowed in STATUS_ALLOWED.items():
        value = _normalize_status(data.get(field))
        if value in allowed and value != "unknown":
            cleaned[field] = value

    recipient_known = data.get("recipient_account_known")
    if isinstance(recipient_known, bool):
        cleaned["recipient_account_known"] = recipient_known
    elif isinstance(recipient_known, str):
        lowered = recipient_known.strip().lower()
        if lowered in {"true", "yes", "known", "확인", "알고있음"}:
            cleaned["recipient_account_known"] = True
        elif lowered in {"false", "no", "모름", "알수없음", "unknown_account"}:
            cleaned["recipient_account_known"] = False

    return cleaned


def _normalize_amount(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, "", "unknown"):
        return None
    if isinstance(value, int):
        return value if value >= 10_000 else None
    if isinstance(value, float):
        return int(value) if value >= 10_000 else None

    if isinstance(value, str):
        text = value.replace(",", "").replace(" ", "").strip().lower()
        if text in {"", "unknown", "null", "none", "n/a"}:
            return None

        try:
            amount = int(float(text))
            return amount if amount >= 10_000 else None
        except ValueError:
            pass

        import re

        mixed = re.search(r"(\d+)만(\d+)천원", text)
        if mixed:
            return int(mixed.group(1)) * 10_000 + int(mixed.group(2)) * 1_000

        man = re.search(r"(\d+(?:\.\d+)?)만원", text)
        if man:
            return int(float(man.group(1)) * 10_000)

        won = re.search(r"(\d+)원", text)
        if won:
            return int(won.group(1))

    return None


def _normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"

    value = value.strip().lower()
    aliases = {
        "not reported": "not_reported",
        "not-requested": "not_requested",
        "not requested": "not_requested",
        "attempted but failed": "attempted_but_failed",
        "attempted-but-failed": "attempted_but_failed",
        "not applied": "not_started",
        "not_applied": "not_started",
        "applied": "requested",
        "in_progress": "requested",
    }
    return aliases.get(value, value)
