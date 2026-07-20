"""Deterministic normalization evaluation helpers."""

from __future__ import annotations

import re
from typing import Any

from recover24.form_patches import normalize_krw_amount

UNKNOWN = "unknown"

_STATUS_ALIASES = {
    "police_status": {
        "reported": "reported",
        "신고": "reported",
        "신고함": "reported",
        "경찰 신고": "reported",
        "not_reported": "not_reported",
        "미신고": "not_reported",
        "신고 안함": "not_reported",
        "planned": "planned",
        "신고 예정": "planned",
    },
    "freeze_status": {
        "requested": "requested",
        "지급정지 요청": "requested",
        "requested_freeze": "requested",
        "attempted_but_failed": "attempted_but_failed",
        "시도했으나 실패": "attempted_but_failed",
        "failed": "attempted_but_failed",
        "completed": "completed",
        "완료": "completed",
    },
    "refund_status": {
        "not_started": "not_started",
        "미진행": "not_started",
        "not started": "not_started",
        "requested": "requested",
        "신청": "requested",
        "completed": "completed",
        "완료": "completed",
    },
}


def normalize_case(raw_input: dict[str, Any], required_fields: list[str] | None = None) -> dict[str, Any]:
    canonical: dict[str, Any] = {}

    if "damage_amount" in raw_input or "damage_amount_krw" in raw_input:
        canonical["damage_amount_krw"] = _normalize_amount(raw_input.get("damage_amount", raw_input.get("damage_amount_krw")))
    if "incident_date" in raw_input:
        canonical["incident_date"] = _normalize_date(raw_input.get("incident_date"))
    if "transfer_date" in raw_input:
        canonical["transfer_date"] = _normalize_date(raw_input.get("transfer_date"))
    if "police_status" in raw_input:
        canonical["police_status"] = _normalize_status("police_status", raw_input.get("police_status"))
    if "freeze_status" in raw_input:
        canonical["freeze_status"] = _normalize_status("freeze_status", raw_input.get("freeze_status"))
    if "refund_status" in raw_input:
        canonical["refund_status"] = _normalize_status("refund_status", raw_input.get("refund_status"))
    if "recipient_account_known" in raw_input:
        canonical["recipient_account_known"] = _normalize_bool(raw_input.get("recipient_account_known"))

    required = required_fields or []
    missing_required_fields = [
        field
        for field in required
        if canonical.get(field, UNKNOWN) == UNKNOWN
    ]

    return {
        "canonical": canonical,
        "missing_required_fields": missing_required_fields,
        "blocks_on_missing_required": bool(missing_required_fields),
    }


def normalize_date_text(value: Any) -> str:
    return _normalize_date(value)


def _normalize_amount(value: Any) -> int | str:
    if _is_unknown(value):
        return UNKNOWN
    return normalize_krw_amount(value)


def _normalize_date(value: Any) -> str:
    if _is_unknown(value):
        return UNKNOWN

    text = str(value).strip()
    match = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})", text)
    if not match:
        return UNKNOWN

    year, month, day = (int(group) for group in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return UNKNOWN
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_status(field: str, value: Any) -> str:
    if _is_unknown(value):
        return UNKNOWN

    text = str(value).strip().lower()
    aliases = _STATUS_ALIASES[field]
    return aliases.get(text, aliases.get(str(value).strip(), UNKNOWN))


def _normalize_bool(value: Any) -> bool | str:
    if _is_unknown(value):
        return UNKNOWN
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "known", "확인", "알고 있음"}:
        return True
    if text in {"false", "no", "n", "0", "unknown_account", "모름", "알 수 없음"}:
        return False
    return UNKNOWN


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "none", "null", "n/a"}
    return False
