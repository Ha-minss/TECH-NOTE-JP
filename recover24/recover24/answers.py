"""LLM-first follow-up answer extraction for Recover24 V3.

Input: the current Question and the user's follow-up answer.
Output: Patch[] describing facts to write into RecoveryCase.

Rules:
- answers.py reads only the answer to the current question.
- answers.py may write only to Question.target_paths.
- If the question target is the special path "evidence", answers.py may write
  official evidence status/note paths.
- answers.py returns Patch[] only.
- answers.py never mutates RecoveryCase.
- answers.py never asks follow-up questions.
- answers.py never generates HTML.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    EvidenceStatus,
    FieldStatus,
    FraudType,
    Patch,
    Question,
    ReportStatus,
    TransactionType,
)
from .providers.base import LLMProvider
from .questions import EVIDENCE_ITEM_LABELS, FIELD_LABELS


EVIDENCE_STATUS_PATHS: set[str] = {f"evidence.{kind}.status" for kind in EVIDENCE_ITEM_LABELS}
EVIDENCE_NOTE_PATHS: set[str] = {f"evidence.{kind}.note" for kind in EVIDENCE_ITEM_LABELS}
EVIDENCE_ANSWER_PATHS: set[str] = EVIDENCE_STATUS_PATHS | EVIDENCE_NOTE_PATHS

FIELD_STATUS_VALUES = {item.value for item in FieldStatus}

REPORT_STATUS_PATHS = {
    "relief.status",
    "investigation.status",
}

FRAUD_TYPE_PATHS = {
    "incident.fraud_type",
}

TRANSACTION_TYPE_PATHS = {
    "transactions.0.transaction_type",
}

AMOUNT_PATHS = {
    "transactions.0.amount_krw",
}

# Boolean paths include every checkbox/yes-no raw fact, including consent.
BOOLEAN_PATHS = {
    "applicant.sms_consent",
    *(f"exclusion.items.exclude_{i}" for i in range(1, 18)),
    "exclusion.final_has_exclusion",
    "optional_reports.id_loss_reported",
    "optional_reports.phone_loss_reported",
    "optional_reports.identity_theft_phone_reported",
    "survey.smishing_link_clicked",
    "survey.malicious_app_installed",
    "survey.provided_id_card",
    "survey.provided_personal_info",
    "survey.provided_device",
    "survey.provided_account_password",
    "survey.provided_security_media",
    "survey.provided_other_financial_info",
    "survey.internet_banking_used",
    "survey.phone_banking_used",
    "survey.open_banking_used",
    "survey.id_lent",
    "survey.id_copy_stored_digitally",
    "survey.phone_lent",
    "survey.id_loss_reported_before_incident",
    "survey.id_copy_stored_before_incident",
    "survey.account_password_stored",
    "survey.phone_lock_enabled",
    "delegation.proxy_used",
    "consent.unique_id_collection_agreed",
    "consent.personal_credit_collection_agreed",
    "consent.unique_id_provision_agreed",
    "consent.personal_credit_provision_agreed",
}


def extract_answer_patches(question: Question, answer_text: str, provider: LLMProvider) -> list[Patch]:
    """Convert a follow-up answer into normalized Patch objects.

    Unlike initial extraction, this function is scoped to the current question.
    The LLM can only return patches for the question's target paths. This prevents
    a transaction answer from unexpectedly modifying investigation, consent, or
    other unrelated areas.
    """

    clean_text = (answer_text or "").strip()
    if not clean_text:
        return []

    allowed_paths = _allowed_paths_for_question(question)
    if not allowed_paths:
        return []

    raw_response = provider.generate_json(_build_answer_prompt(question, clean_text, allowed_paths))
    data = _coerce_llm_json(raw_response)

    patches: list[Patch] = []
    for item in _iter_patch_items(data):
        patch = _patch_from_llm_item(item, allowed_paths)
        if patch is not None:
            patches.append(patch)

    return _dedupe_patches_keep_last(patches)


def _allowed_paths_for_question(question: Question) -> set[str]:
    """Return the exact paths answers.py may patch for this question."""

    allowed: set[str] = set()
    for path in question.target_paths:
        if path == "evidence":
            allowed.update(EVIDENCE_ANSWER_PATHS)
        elif path in FIELD_LABELS:
            allowed.add(path)
    return allowed


def _build_answer_prompt(question: Question, answer_text: str, allowed_paths: set[str]) -> str:
    """Build a strict prompt for scoped follow-up answer extraction."""

    allowed_path_lines = "\n".join(f"- {path}: {_label_for_path(path)}" for path in sorted(allowed_paths))
    target_path_lines = "\n".join(f"- {path}" for path in question.target_paths)

    return f"""
You are Recover24 V3's follow-up answer extraction engine.

Your job:
Read the user's Korean answer to the current question and return JSON patches for Recover24's RecoveryCase.

Return JSON only. No markdown. No explanation.

Current question id: {question.question_id}
Current question category: {question.category.value}
Current question prompt:
{question.prompt}

Question target paths:
{target_path_lines}

Hard rules:
- Extract only facts explicitly stated by the user in this answer.
- Use only the allowed patch paths below.
- Never output a path outside the allowed patch paths.
- Do not patch unrelated fields even if the user mentions them.
- If the user says they do not know, use status "unknown" for the relevant target path.
- If the user says it does not apply, use status "not_applicable" for the relevant target path.
- If information is absent, omit that patch.
- Do not ask questions.
- Do not write HTML.
- Use enum values exactly as provided.
- Store KRW amounts as integers, e.g. 100만원 -> 1000000.

Allowed field status values:
- answered
- unknown
- not_applicable

Allowed fraud_type values:
- authority_impersonation
- family_impersonation
- loan_scam
- smishing_malware
- institution_impersonation
- other

Allowed report status values:
- not_reported
- planned
- reported
- in_progress
- completed
- closed
- other
- unknown

Allowed transaction_type values:
- mobile_banking_transfer
- internet_banking_transfer
- phone_banking_transfer
- atm_transfer
- card_or_loan
- unknown

Allowed evidence status values for evidence.<kind>.status:
- available
- missing
- planned
- not_applicable
- unknown
- not_asked

Allowed patch paths for this answer:
{allowed_path_lines}

Output format:
{{
  "patches": [
    {{
      "path": "transactions.0.destination_account_holder",
      "value": "김철수",
      "status": "answered",
      "source_text": "김철수에게",
      "confidence": 0.95
    }}
  ]
}}

User answer:
{answer_text}
""".strip()


def _label_for_path(path: str) -> str:
    if path in FIELD_LABELS:
        return FIELD_LABELS[path]
    if path.startswith("evidence."):
        parts = path.split(".")
        if len(parts) == 3:
            kind, attribute = parts[1], parts[2]
            label = EVIDENCE_ITEM_LABELS.get(kind, kind)
            suffix = "보유 상태" if attribute == "status" else "비고/메모"
            return f"{label} {suffix}"
    return path


def _iter_patch_items(data: Any) -> list[dict[str, Any]]:
    """Return patch item dicts from supported LLM response shapes."""

    if isinstance(data, dict):
        patches = data.get("patches", [])
        if isinstance(patches, list):
            return [item for item in patches if isinstance(item, dict)]
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def _coerce_llm_json(raw: Any) -> Any:
    """Coerce provider output into Python JSON-compatible data."""

    if isinstance(raw, (dict, list)):
        return raw

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    if not isinstance(raw, str):
        return {}

    text = raw.strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_object = _extract_first_json_object(text)
    if json_object is None:
        return {}

    try:
        return json.loads(json_object)
    except json.JSONDecodeError:
        return {}


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from a text response."""

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _patch_from_llm_item(item: dict[str, Any], allowed_paths: set[str]) -> Patch | None:
    """Validate one LLM patch item and convert it to Patch."""

    path = item.get("path")
    if not isinstance(path, str) or path not in allowed_paths:
        return None

    status = _normalize_status(item.get("status", FieldStatus.ANSWERED.value))
    if status is None or status == FieldStatus.NOT_ASKED:
        return None

    raw_value = item.get("value")

    if status in {FieldStatus.UNKNOWN, FieldStatus.NOT_APPLICABLE}:
        value = None
    else:
        value = _normalize_value(path, raw_value)
        if value is None:
            return None
        status = FieldStatus.ANSWERED

    return Patch(
        path=path,
        value=value,
        status=status,
        source_text=_normalize_source_text(item.get("source_text")),
        confidence=_normalize_confidence(item.get("confidence")),
    )


def _normalize_status(value: Any) -> FieldStatus | None:
    if isinstance(value, FieldStatus):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in FIELD_STATUS_VALUES:
            return FieldStatus(normalized)
    return None


def _normalize_value(path: str, value: Any) -> Any:
    """Normalize LLM values into the raw types expected by models.py."""

    if value is None:
        return None

    if path in FRAUD_TYPE_PATHS:
        return _normalize_enum(value, FraudType)

    if path in REPORT_STATUS_PATHS:
        return _normalize_enum(value, ReportStatus)

    if path in TRANSACTION_TYPE_PATHS:
        return _normalize_enum(value, TransactionType)

    if path in EVIDENCE_STATUS_PATHS:
        return _normalize_enum(value, EvidenceStatus)

    if path in AMOUNT_PATHS:
        return _normalize_amount_krw(value)

    if path in BOOLEAN_PATHS:
        return _normalize_bool(value)

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)

    return None


def _normalize_enum(value: Any, enum_cls: type[FraudType] | type[ReportStatus] | type[TransactionType] | type[EvidenceStatus]) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        try:
            return enum_cls(normalized)
        except ValueError:
            return None
    return None


def _normalize_amount_krw(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        return _parse_korean_amount(value)
    return None


def _parse_korean_amount(text: str) -> int | None:
    clean = text.strip().replace(",", "").replace(" ", "")
    if not clean:
        return None

    # 1억2천만, 1억2000만원, 2억 etc.
    total = 0
    rest = clean

    eok_match = re.search(r"(\d+)억", rest)
    if eok_match:
        total += int(eok_match.group(1)) * 100_000_000
        rest = rest[eok_match.end() :]

    cheonman_match = re.search(r"(\d+)천만", rest)
    if cheonman_match:
        total += int(cheonman_match.group(1)) * 10_000_000
        rest = rest[cheonman_match.end() :]

    man_match = re.search(r"(\d+)만", rest)
    if man_match:
        total += int(man_match.group(1)) * 10_000
        rest = rest[man_match.end() :]

    if total > 0:
        return total

    won_match = re.search(r"(\d+)원?", clean)
    if won_match:
        return int(won_match.group(1))

    return None


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower().replace(" ", "")
    true_values = {
        "true", "yes", "y", "1",
        "예", "네", "맞음", "있음", "했다", "했음", "동의", "동의함", "모두동의", "전체동의",
    }
    false_values = {
        "false", "no", "n", "0",
        "아니오", "아니요", "아님", "없음", "안함", "안했음", "미동의", "동의안함", "거부",
    }

    if normalized in true_values:
        return True
    if normalized in false_values:
        return False
    return None


def _normalize_source_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 1.0)
    if isinstance(value, str):
        try:
            return min(max(float(value), 0.0), 1.0)
        except ValueError:
            return None
    return None


def _dedupe_patches_keep_last(patches: list[Patch]) -> list[Patch]:
    """Deduplicate by path while keeping the last patch for each path."""

    by_path: dict[str, Patch] = {}
    order: list[str] = []

    for patch in patches:
        if patch.path not in by_path:
            order.append(patch.path)
        by_path[patch.path] = patch

    return [by_path[path] for path in order]
