"""Safe template baseline for Recover24 narrative generation."""

from __future__ import annotations

import time
from typing import Any

from evaluation.dataset_loader import GoldCase
from evaluation.validators.fact_validator import format_amount_krw


class TemplateGenerator:
    name = "template"

    def generate(self, case: GoldCase) -> dict[str, Any]:
        start = time.perf_counter()
        facts = case.structured_facts
        case_type = facts.get("fraud_type") or case.case_type
        amount = format_amount_krw(facts.get("amount_krw")) if facts.get("amount_krw") else "피해금액 미확인"
        source_bank = facts.get("source_bank") or facts.get("source_bank_name") or "출금은행 미확인"
        dest_bank = facts.get("destination_bank") or facts.get("destination_account_type") or "수취기관 미확인"
        method = facts.get("transfer_method") or "거래수단 미확인"
        contact = facts.get("contact_channel") or "접촉경로 미확인"

        incident = (
            f"신청인은 {contact}을 통해 {case_type} 유형의 피해를 입은 것으로 확인됩니다. "
            f"확인된 구조화 정보에 따르면 {source_bank}에서 {dest_bank}로 {amount} 상당의 피해가 발생하였으며, "
            f"거래수단은 {method}로 기록되어 있습니다."
        )
        post_action = _post_action_from_status(facts)
        summary = (
            f"피해유형: {case_type}\n"
            f"피해금액: {amount}\n"
            f"거래/수법: {source_bank} → {dest_bank}, {method}\n"
            f"조치상태: 지급정지={facts.get('freeze_status', 'unknown')}, "
            f"수사기관 신고={facts.get('police_status', 'unknown')}, 환급={facts.get('refund_status', 'unknown')}"
        )
        return {
            "method": self.name,
            "outputs": {
                "incident_circumstances": incident,
                "post_action": post_action,
                "staff_summary": summary,
            },
            "meta": {
                "fallback_used": False,
                "blocked_by_validator": False,
                "latency_sec": round(time.perf_counter() - start, 4),
                "llm_calls": 0,
            },
        }


def _post_action_from_status(facts: dict[str, Any]) -> str:
    parts: list[str] = []
    freeze = facts.get("freeze_status", "unknown")
    police = facts.get("police_status", "unknown")
    refund = facts.get("refund_status", "unknown")

    if freeze == "completed":
        parts.append("금융기관에 지급정지 또는 거래제한을 요청한 것으로 기록되어 있습니다.")
    elif freeze == "attempted_but_failed":
        parts.append("지급정지 또는 거래제한을 시도하였으나 완료하지 못한 것으로 기록되어 있습니다.")
    elif freeze == "not_requested":
        parts.append("지급정지 요청은 아직 확인되지 않았습니다.")
    else:
        parts.append("지급정지 요청 여부는 추가 확인이 필요합니다.")

    if police in {"reported", "in_progress"}:
        parts.append("수사기관 신고 사실이 기록되어 있습니다.")
    elif police == "not_reported":
        parts.append("수사기관 신고는 아직 이루어지지 않은 것으로 기록되어 있습니다.")
    else:
        parts.append("수사기관 신고 여부는 추가 확인이 필요합니다.")

    if refund in {"applied", "in_progress"}:
        parts.append("피해구제 또는 환급 신청이 진행 중인 것으로 기록되어 있습니다.")
    elif refund == "completed":
        parts.append("피해구제 또는 환급 절차가 완료된 것으로 기록되어 있습니다.")
    elif refund in {"not_applied", "planned"}:
        parts.append("피해구제 신청은 미신청 또는 예정 상태로 기록되어 있습니다.")
    else:
        parts.append("피해구제 신청 여부는 추가 확인이 필요합니다.")

    return " ".join(parts)
