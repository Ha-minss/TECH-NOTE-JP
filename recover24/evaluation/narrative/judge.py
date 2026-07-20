"""LLM judge for narrative required-element coverage."""

from __future__ import annotations

import json
from typing import Any

from evaluation.llm_json import coerce_json
from recover24.providers.base import LLMProvider

from .checklist import evaluate_narrative as evaluate_narrative_checklist

ALLOWED_LABELS = {"present", "missing", "contradicted", "unsupported"}


def evaluate_narrative_with_llm(
    *,
    canonical_case: dict[str, Any],
    generated_text: str,
    required_elements: list[dict[str, Any]],
    provider: LLMProvider | None,
    fallback_to_checklist: bool = True,
) -> dict[str, Any]:
    if provider is None:
        result = evaluate_narrative_checklist(
            canonical_case=canonical_case,
            generated_text=generated_text,
            required_elements=required_elements,
        )
        result.update({"judge_available": False, "judge_method": "checklist", "fallback_used": False})
        return result

    prompt = _build_judge_prompt(canonical_case, generated_text, required_elements)

    try:
        raw = provider.generate_json(prompt)
    except Exception as exc:
        if fallback_to_checklist:
            result = evaluate_narrative_checklist(
                canonical_case=canonical_case,
                generated_text=generated_text,
                required_elements=required_elements,
            )
            result.update(
                {
                    "judge_available": False,
                    "judge_method": "checklist_fallback",
                    "fallback_used": True,
                    "judge_error": f"{type(exc).__name__}: {exc}",
                }
            )
            return result
        raise

    data = coerce_json(raw)
    if not isinstance(data, dict):
        if fallback_to_checklist:
            result = evaluate_narrative_checklist(
                canonical_case=canonical_case,
                generated_text=generated_text,
                required_elements=required_elements,
            )
            result.update(
                {
                    "judge_available": False,
                    "judge_method": "checklist_fallback",
                    "fallback_used": True,
                    "judge_error": "LLM judge output was not JSON.",
                    "raw_output": str(raw)[:500],
                }
            )
            return result
        data = {}

    return _coerce_judge_result(data, required_elements)


def _build_judge_prompt(
    canonical_case: dict[str, Any],
    generated_text: str,
    required_elements: list[dict[str, Any]],
) -> str:
    normalized_elements = []
    for element in required_elements:
        normalized_elements.append(
            {
                "id": element.get("id"),
                "description": element.get("description") or element.get("desc") or "",
                "expected_hints": element.get("expected", []),
            }
        )

    return f"""
당신은 금융사고 제출 문서의 사실 보존 평가자입니다.
생성문이 각 required_element를 의미적으로 충족하는지 평가하세요.
키워드가 정확히 같지 않아도 의미가 같으면 present입니다.
원본 canonical_case에 없는 새로운 확정 주장은 unsupported로 표시하세요.
원본 사실과 반대로 말하면 contradicted로 표시하세요.
반드시 JSON만 반환하세요.

라벨:
- present: 생성문에 해당 필수요소가 의미적으로 포함됨
- missing: 생성문에 해당 필수요소가 없음
- contradicted: 생성문이 원본 사실과 반대로 말함
- unsupported: 생성문이 근거 없는 새로운 확정 사실을 말함

반환 형식:
{{
  "elements": [
    {{"id": "amount", "label": "present", "evidence": "생성문 근거 문장"}}
  ],
  "unsupported_claims": [{{"claim": "...", "reason": "..."}}],
  "contradictions": [{{"field": "...", "claim": "...", "gold_value": "..."}}]
}}

canonical_case:
{json.dumps(canonical_case, ensure_ascii=False, indent=2)}

required_elements:
{json.dumps(normalized_elements, ensure_ascii=False, indent=2)}

생성문:
{generated_text}
""".strip()


def _coerce_judge_result(data: dict[str, Any], required_elements: list[dict[str, Any]]) -> dict[str, Any]:
    required_ids = [str(item.get("id")) for item in required_elements]
    by_id: dict[str, dict[str, Any]] = {}

    raw_elements = data.get("elements", [])
    if isinstance(raw_elements, list):
        for item in raw_elements:
            if not isinstance(item, dict):
                continue

            element_id = str(item.get("id", "")).strip()
            if element_id not in required_ids:
                continue

            label = str(item.get("label", "missing")).strip().lower()
            if label not in ALLOWED_LABELS:
                label = "missing"

            by_id[element_id] = {
                "id": element_id,
                "label": label,
                "evidence": str(item.get("evidence", "") or ""),
            }

    element_judgements = []
    included = []
    missing = []
    factual_errors = []

    for element_id in required_ids:
        judgement = by_id.get(element_id, {"id": element_id, "label": "missing", "evidence": ""})
        element_judgements.append(judgement)

        label = judgement["label"]
        if label == "present":
            included.append(element_id)
        else:
            missing.append(element_id)

        if label in {"contradicted", "unsupported"}:
            factual_errors.append(
                {
                    "field": element_id,
                    "label": label,
                    "message": f"LLM judge marked required element {element_id} as {label}.",
                    "evidence": judgement.get("evidence", ""),
                }
            )

    unsupported_claims = _list_of_dicts(data.get("unsupported_claims"))
    contradictions = _list_of_dicts(data.get("contradictions"))

    factual_errors.extend({"type": "unsupported_claim", **item} for item in unsupported_claims)
    factual_errors.extend({"type": "contradiction", **item} for item in contradictions)

    return {
        "included_elements": included,
        "missing_elements": missing,
        "factual_errors": factual_errors,
        "passed": not missing and not factual_errors,
        "judge_available": True,
        "judge_method": "llm",
        "fallback_used": False,
        "element_judgements": element_judgements,
        "unsupported_claims": unsupported_claims,
        "contradictions": contradictions,
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
