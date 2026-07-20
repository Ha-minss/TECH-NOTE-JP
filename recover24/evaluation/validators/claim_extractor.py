"""Claim extraction without case-specific keyword patterns.

This module intentionally avoids Korean keyword lists for statuses such as
"지급정지 완료". For status/action/inference claims, use an LLM provider to
extract normalized claims from the generated text. Deterministic extraction is
limited to numeric KRW values because exact numeric preservation is critical and
can be checked reliably with code.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evaluation.dataset_loader import GoldCase
from recover24.providers.base import LLMProvider


AMOUNT_RE = re.compile(
    r"(?P<num>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>억원|억|천만원|백만원|만원|원)?"
)


def extract_numeric_amounts_krw(text: str) -> list[int]:
    amounts: list[int] = []
    for match in AMOUNT_RE.finditer(text or ""):
        raw = match.group("num").replace(",", "")
        unit = match.group("unit") or ""
        try:
            value = float(raw)
        except ValueError:
            continue
        # Avoid treating years like 2021년 as KRW amounts.
        end = match.end()
        if end < len(text) and text[end:end+1] in {"년", "월", "일", "시", "분"}:
            continue
        if unit in {"억원", "억"}:
            amount = int(value * 100_000_000)
        elif unit == "천만원":
            amount = int(value * 10_000_000)
        elif unit == "백만원":
            amount = int(value * 1_000_000)
        elif unit == "만원":
            amount = int(value * 10_000)
        elif unit == "원":
            amount = int(value)
        else:
            # Bare 7+ digit numbers are likely KRW; smaller bare numbers are often dates/times/counts.
            amount = int(value) if value >= 1_000_000 else 0
        if amount >= 10_000:
            amounts.append(amount)
    return sorted(set(amounts))


def extract_claims_with_provider(text: str, case: GoldCase, provider: LLMProvider | None) -> dict[str, Any]:
    """Extract normalized claims from model output.

    If provider is None, return deterministic numeric claims only. This keeps the
    evaluation pipeline runnable offline. For full status contradiction scoring,
    pass the same Gemma provider or a judge provider.
    """

    base: dict[str, Any] = {
        "amounts_krw": extract_numeric_amounts_krw(text),
        "status_claims": [],
        "unsupported_claims": [],
        "event_order": [],
        "supported_fact_ids": [],
    }
    if provider is None:
        return base

    raw = provider.generate_json(_build_claim_prompt(text, case))
    data = _coerce_json(raw)
    if not isinstance(data, dict):
        return base
    base.update({
        "status_claims": _list_of_dicts(data.get("status_claims")),
        "unsupported_claims": _list_of_dicts(data.get("unsupported_claims")),
        "event_order": [str(item) for item in data.get("event_order", []) if str(item).strip()],
        "supported_fact_ids": [str(item) for item in data.get("supported_fact_ids", []) if str(item).strip()],
    })
    return base


def _build_claim_prompt(text: str, case: GoldCase) -> str:
    return f"""
당신은 금융사고 문서 출력의 사실 주장 추출기입니다.
아래 '출력문'에서 명시적으로 주장한 사실만 추출하세요.
원본 사실을 참고하되, 출력문에 없는 주장을 만들지 마세요.

반환 JSON 형식만 지키세요.

상태 필드와 허용값:
- freeze_status: not_requested, requested, completed, attempted_but_failed, unknown
- police_status: not_reported, planned, reported, in_progress, closed, unknown
- refund_status: not_applied, planned, applied, in_progress, completed, unknown

supported_fact_ids는 출력문이 실제로 언급한 required_fact_ids만 넣으세요.
unsupported_claims에는 원본 structured_facts 또는 raw_statement에 근거가 없는 새로운 완료/판단/결론 주장을 넣으세요.

원본 structured_facts:
{json.dumps(case.structured_facts, ensure_ascii=False, indent=2)}

required_fact_ids:
{json.dumps(case.required_fact_ids, ensure_ascii=False)}

원본 자유진술:
{case.raw_statement}

출력문:
{text}

반환 예:
{{
  "status_claims": [
    {{"field": "freeze_status", "claimed_value": "completed", "evidence_text": "지급정지를 완료했습니다"}}
  ],
  "unsupported_claims": [
    {{"claim": "피해금 환급 완료", "reason": "원본에 환급 완료 사실 없음"}}
  ],
  "event_order": ["contact", "transfer", "recognition", "freeze_attempt"],
  "supported_fact_ids": ["amount_krw", "fraud_type"]
}}
""".strip()


def _coerce_json(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
