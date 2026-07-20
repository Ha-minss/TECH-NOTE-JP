"""Rule-based statement fact extraction for consistency evaluation."""

from __future__ import annotations

import re
from typing import Any

from evaluation.normalization.normalizer import normalize_date_text


def extract_statement_facts(raw_statement: str) -> dict[str, Any]:
    text = (raw_statement or "").strip()
    facts: dict[str, Any] = {}

    amount = _extract_amount(text)
    if amount is not None:
        facts["damage_amount_krw"] = amount

    incident_date, transfer_date = _extract_dates(text)
    if incident_date is not None:
        facts["incident_date"] = incident_date
    if transfer_date is not None:
        facts["transfer_date"] = transfer_date

    police_status = _extract_police_status(text)
    if police_status is not None:
        facts["police_status"] = police_status

    freeze_status = _extract_freeze_status(text)
    if freeze_status is not None:
        facts["freeze_status"] = freeze_status

    refund_status = _extract_refund_status(text)
    if refund_status is not None:
        facts["refund_status"] = refund_status

    recipient_known = _extract_recipient_account_known(text)
    if recipient_known is not None:
        facts["recipient_account_known"] = recipient_known

    return facts


def _extract_amount(text: str) -> int | None:
    compact = text.replace(",", "").replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)만원", compact)
    if match:
        return int(float(match.group(1)) * 10_000)

    match = re.search(r"(\d[\d,]*)원", text)
    if match:
        return int(match.group(1).replace(",", ""))

    return None


def _extract_dates(text: str) -> tuple[str | None, str | None]:
    matches = re.findall(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}|20\d{2}년\s*\d{1,2}월\s*\d{1,2}일", text)
    normalized = [normalize_date_text(item) for item in matches]
    normalized = [item for item in normalized if item != "unknown"]
    if not normalized:
        return None, None
    if len(normalized) == 1:
        return normalized[0], None
    return normalized[0], normalized[1]


def _extract_police_status(text: str) -> str | None:
    if "미신고" in text or "신고하지" in text:
        return "not_reported"
    if "경찰" in text and "신고" in text:
        return "reported"
    return None


def _extract_freeze_status(text: str) -> str | None:
    if "지급정지" not in text:
        return None
    if "실패" in text or "되지 않았" in text:
        return "attempted_but_failed"
    if "요청" in text or "신청" in text:
        return "requested"
    if "완료" in text:
        return "completed"
    return None


def _extract_refund_status(text: str) -> str | None:
    if "환급" not in text:
        return None

    refund_context = next((part for part in re.split(r"[,.]\s*|그리고|하지만", text) if "환급" in part), text)
    if "완료" in refund_context:
        return "completed"
    if "신청" in refund_context or "요청" in refund_context:
        return "requested"
    if "아직" in refund_context or "전" in refund_context:
        return "not_started"
    return None


def _extract_recipient_account_known(text: str) -> bool | None:
    if "계좌" not in text:
        return None
    if "모릅" in text or "모름" in text or "알 수 없" in text:
        return False
    if "알고" in text or "확인" in text:
        return True
    return None
