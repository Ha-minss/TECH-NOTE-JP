"""Gemma narrative generator for Recover24 evaluation."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from evaluation.dataset_loader import GoldCase
from recover24.providers.base import LLMProvider


class GemmaGenerator:
    name = "gemma"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def generate(self, case: GoldCase) -> dict[str, Any]:
        start = time.perf_counter()
        if self.provider is None:
            # Dry-run fallback: keeps the pipeline executable without Colab.
            outputs = _dry_run_outputs(case)
            llm_calls = 0
        else:
            raw = self.provider.generate_json(_build_prompt(case))
            outputs = _coerce_outputs(raw)
            llm_calls = 1
        return {
            "method": self.name,
            "outputs": outputs,
            "meta": {
                "fallback_used": False,
                "blocked_by_validator": False,
                "latency_sec": round(time.perf_counter() - start, 4),
                "llm_calls": llm_calls,
            },
        }


def _build_prompt(case: GoldCase) -> str:
    return f"""
당신은 은행 전자금융거래 사고 피해 신고서를 작성하는 보조자입니다.
아래 구조화 사실과 피해자 원문 진술만 근거로 세 가지 문장을 작성하세요.

규칙:
- 없는 사실을 추가하지 마세요.
- 금액, 날짜, 은행명, 지급정지/경찰신고/환급 상태를 바꾸지 마세요.
- 구조화 사실과 원문 진술이 충돌하면 단정하지 말고 '확인 필요'라고 쓰세요.
- 공식 문서 문체로 작성하세요.
- 출력은 JSON만 반환하세요.

구조화 사실:
{json.dumps(case.structured_facts, ensure_ascii=False, indent=2)}

피해자 원문 진술:
{case.raw_statement}

반환 형식:
{{
  "incident_circumstances": "사고 발생 경위 문장",
  "post_action": "사고 인지 후 조치 내역 문장",
  "staff_summary": "은행 담당자용 핵심 사건 요약"
}}
""".strip()


def _coerce_outputs(raw: Any) -> dict[str, str]:
    data: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "incident_circumstances": str(data.get("incident_circumstances", "")).strip(),
        "post_action": str(data.get("post_action", "")).strip(),
        "staff_summary": str(data.get("staff_summary", "")).strip(),
    }


def _dry_run_outputs(case: GoldCase) -> dict[str, str]:
    # Dry-run is intentionally not a high-quality model. It allows local tests to run.
    raw = case.raw_statement.strip().replace("\n", " ")
    short = raw[:500]
    return {
        "incident_circumstances": short,
        "post_action": "사고 인지 후 조치 내역은 구조화 사실과 원문 진술을 추가 확인해야 합니다.",
        "staff_summary": f"{case.case_type}: {short[:240]}",
    }
