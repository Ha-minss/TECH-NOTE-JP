"""Build Patch[] from structured UI/form inputs.

This module is deliberately *not* a Korean natural-language parser. It only
converts explicit widget values (text inputs, radio buttons, selectboxes,
checkboxes) into Patch objects that can enter the existing V3 pipeline:

    form widget -> Patch[] -> patching.py -> RecoveryCase -> document_view/html

Free-form narrative and ambiguous meaning still belong in answers.py/LLM.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .field_policy import blank_status_for_path
from .models import EvidenceStatus, FieldStatus, FraudType, Patch, ReportStatus, TransactionType

UNKNOWN_SENTINELS = {"__unknown__", "unknown", "모름", "미확인", "모르겠음", "모르겠습니다"}
NOT_APPLICABLE_SENTINELS = {"__not_applicable__", "not_applicable", "해당 없음", "해당없음", "없음", "없다", "없습니다", "-"}
SKIP_SENTINELS = {"", None, "__skip__", "선택 안 함", "미응답"}

BOOLEAN_PATH_PREFIXES = (
    "exclusion.items.",
    "optional_reports.",
    "survey.",
    "consent.",
)
BOOLEAN_PATHS = {
    "applicant.sms_consent",
    "exclusion.final_has_exclusion",
    "delegation.proxy_used",

    # Survey yes/no fields whose names do not end with a boolean-looking suffix.
    "survey.id_copy_stored_before_incident",
    "survey.account_password_stored",
    "survey.id_loss_reported_before_incident",
    "survey.phone_lock_enabled",
    "survey.id_copy_stored_digitally",
}

FRAUD_TYPE_PATHS = {"incident.fraud_type"}
REPORT_STATUS_PATHS = {"relief.status", "investigation.status"}
TRANSACTION_TYPE_PATHS = {"transactions.0.transaction_type"}
AMOUNT_SUFFIXES = ("amount_krw",)


def build_form_patches(values: Mapping[str, Any], *, source_text: str = "form") -> list[Patch]:
    """Convert explicit path->value widget outputs into Patch objects.

    Empty/skip values are ignored for required fields. For fields declared
    optional in field_policy.py, an empty submission becomes NOT_APPLICABLE or
    UNKNOWN so the app does not ask the same optional blank again. The caller
    can also pass sentinel strings for UNKNOWN / NOT_APPLICABLE.
    """

    patches: list[Patch] = []
    for path, raw_value in values.items():
        patch = patch_from_form_value(path, raw_value, source_text=source_text)
        if patch is not None:
            patches.append(patch)
    return patches


def patch_from_form_value(path: str, raw_value: Any, *, source_text: str = "form") -> Patch | None:
    """Create one Patch from a structured widget value."""

    if raw_value in SKIP_SENTINELS:
        blank_status = blank_status_for_path(path)
        if blank_status is not None:
            return Patch(path=path, value=None, status=blank_status, source_text=source_text)
        return None

    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if cleaned in SKIP_SENTINELS:
            blank_status = blank_status_for_path(path)
            if blank_status is not None:
                return Patch(path=path, value=None, status=blank_status, source_text=source_text)
            return None
        if cleaned in UNKNOWN_SENTINELS:
            return Patch(path=path, value=None, status=FieldStatus.UNKNOWN, source_text=source_text)
        if cleaned in NOT_APPLICABLE_SENTINELS:
            return Patch(path=path, value=None, status=FieldStatus.NOT_APPLICABLE, source_text=source_text)
        raw_value = cleaned

    normalized = normalize_form_value(path, raw_value)
    return Patch(path=path, value=normalized, status=FieldStatus.ANSWERED, source_text=source_text)


def build_boolean_choice_patches(choices: Mapping[str, str], *, source_text: str = "form") -> list[Patch]:
    """Convert yes/no/unknown/not_applicable radio choices into Patch[]."""

    mapping: dict[str, Any] = {
        "예": True,
        "아니오": False,
        "해당": True,
        "해당 없음": "__not_applicable__",
        "해당없음": "__not_applicable__",
        "모름": "__unknown__",
        "미확인": "__unknown__",
        "해당 없음/대상 아님": "__not_applicable__",
        "선택 안 함": "__skip__",
    }
    values = {path: mapping.get(choice, choice) for path, choice in choices.items()}
    return build_form_patches(values, source_text=source_text)


def build_evidence_patches(
    statuses: Mapping[str, str | EvidenceStatus],
    notes: Mapping[str, str] | None = None,
    *,
    source_text: str = "form:evidence",
) -> list[Patch]:
    """Convert attachment status selectboxes into evidence.<kind> Patch[]."""

    patches: list[Patch] = []
    notes = notes or {}
    for kind, raw_status in statuses.items():
        if raw_status in SKIP_SENTINELS:
            continue
        status = normalize_evidence_status(raw_status)
        patches.append(
            Patch(
                path=f"evidence.{kind}.status",
                value=status,
                status=FieldStatus.ANSWERED,
                source_text=source_text,
            )
        )
        note = (notes.get(kind) or "").strip()
        if note:
            patches.append(
                Patch(
                    path=f"evidence.{kind}.note",
                    value=note,
                    status=FieldStatus.ANSWERED,
                    source_text=source_text,
                )
            )
    return patches


def normalize_form_value(path: str, value: Any) -> Any:
    """Normalize explicit widget values to the expected domain type."""

    if path in FRAUD_TYPE_PATHS:
        return value if isinstance(value, FraudType) else FraudType(str(value))

    if path in REPORT_STATUS_PATHS:
        return value if isinstance(value, ReportStatus) else ReportStatus(str(value))

    if path in TRANSACTION_TYPE_PATHS:
        return value if isinstance(value, TransactionType) else TransactionType(str(value))

    if path.endswith(AMOUNT_SUFFIXES):
        return normalize_krw_amount(value)

    if is_boolean_path(path):
        return normalize_boolean(value)

    return value


def normalize_evidence_status(value: str | EvidenceStatus) -> EvidenceStatus:
    if isinstance(value, EvidenceStatus):
        return value

    value_text = str(value).strip()
    label_map = {
        "보유": EvidenceStatus.AVAILABLE,
        "없음": EvidenceStatus.MISSING,
        "추후 제출": EvidenceStatus.PLANNED,
        "추후제출": EvidenceStatus.PLANNED,
        "해당 없음": EvidenceStatus.NOT_APPLICABLE,
        "해당없음": EvidenceStatus.NOT_APPLICABLE,
        "모름": EvidenceStatus.UNKNOWN,
        "미확인": EvidenceStatus.UNKNOWN,
        "미응답": EvidenceStatus.NOT_ASKED,
    }
    if value_text in label_map:
        return label_map[value_text]
    return EvidenceStatus(value_text)


def normalize_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "예", "네", "동의", "동의함", "해당", "있음", "보유"}:
        return True
    if text in {"false", "0", "no", "n", "아니오", "아니요", "미동의", "동의안함", "해당없음", "해당 없음", "없음"}:
        return False
    raise ValueError(f"Cannot normalize boolean value: {value!r}")


def normalize_krw_amount(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean cannot be a KRW amount")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("Empty amount")

    match = re.search(r"(\d+(?:\.\d+)?)\s*만\s*원", text)
    if match:
        return int(float(match.group(1)) * 10_000)

    match = re.search(r"(\d+(?:\.\d+)?)\s*원", text)
    if match:
        return int(float(match.group(1)))

    return int(float(text))


def is_boolean_path(path: str) -> bool:
    if path in BOOLEAN_PATHS:
        return True
    if path.startswith("exclusion.items."):
        return True
    bool_suffixes = (
        "_reported",
        "_clicked",
        "_installed",
        "_id_card",
        "_personal_info",
        "_device",
        "_account_password",
        "_security_media",
        "_other_financial_info",
        "_used",
        "_lent",
        "_digitally",
        "_enabled",
        "_agreed",
    )
    return path.endswith(bool_suffixes)
