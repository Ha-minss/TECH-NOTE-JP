"""LLM-first initial statement extraction for Recover24 V3.

Input: the victim's first free-form Korean statement.
Output: Patch[] describing facts to write into RecoveryCase.

Rules:
- LLM reads the natural language statement.
- extraction.py validates and normalizes the LLM JSON.
- extraction.py returns Patch[] only.
- extraction.py never mutates RecoveryCase.
- extraction.py never asks follow-up questions.
- extraction.py never generates HTML.
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
    ReportStatus,
    TransactionType,
)
from .providers.base import LLMProvider


# The LLM is allowed to write only to these RecoveryCase paths.
# This is the safety gate that prevents the model from inventing schema names.
# Consent fields are intentionally blocked from initial extraction.
# They must be answered explicitly in the consent step handled by answers.py.
BLOCKED_INITIAL_EXTRACTION_PATHS: set[str] = {
    "consent.unique_id_collection_agreed",
    "consent.personal_credit_collection_agreed",
    "consent.unique_id_provision_agreed",
    "consent.personal_credit_provision_agreed",
}

EVIDENCE_ITEM_KINDS: tuple[str, ...] = (
    "id_card_copy",
    "police_certificate",
    "id_loss_evidence",
    "phone_evidence",
    "complaint_evidence",
    "investigation_delegation",
    "data_leak_notice",
    "delay_reason",
    "family_proof",
    "signature_certificate",
    "security_survey",
    "other_evidence",
    "passport_or_travel_proof",
)

EVIDENCE_STATUS_PATHS: set[str] = {f"evidence.{kind}.status" for kind in EVIDENCE_ITEM_KINDS}
EVIDENCE_NOTE_PATHS: set[str] = {f"evidence.{kind}.note" for kind in EVIDENCE_ITEM_KINDS}

# The LLM is allowed to write only to these RecoveryCase paths.
# This is the safety gate that prevents the model from inventing schema names.
# It covers every user-answerable FieldValue in models.py except explicit consent,
# plus official evidence attachment status/note paths used for Page 7.
ALLOWED_PATCH_PATHS: set[str] = {
    # Page 1 applicant facts.
    "applicant.name",
    "applicant.birth_date",
    "applicant.customer_number",
    "applicant.company_name",
    "applicant.business_number",
    "applicant.phone_number",
    "applicant.mobile_number",
    "applicant.email",
    "applicant.address",
    "applicant.memo",
    "applicant.sms_consent",

    # Page 1-2 exclusion checklist.
    *(f"exclusion.items.exclude_{i}" for i in range(1, 18)),
    "exclusion.final_has_exclusion",

    # Page 2 incident summary.
    "incident.first_occurred_at",
    "incident.recognized_at",
    "incident.first_freeze_at",
    "incident.fraud_type",
    "incident.overview",

    # Page 2 first transaction row. V3 starts with index 0 and patching.py can expand later.
    "transactions.0.source_bank",
    "transactions.0.source_account_number",
    "transactions.0.amount_krw",
    "transactions.0.destination_bank",
    "transactions.0.destination_account_number",
    "transactions.0.destination_account_holder",
    "transactions.0.holder_type",
    "transactions.0.transaction_type",
    "transactions.0.transferred_at",

    # Page 2 optional reports.
    "optional_reports.id_loss_reported",
    "optional_reports.id_loss_reported_date",
    "optional_reports.phone_loss_reported",
    "optional_reports.phone_loss_reported_date",
    "optional_reports.identity_theft_phone_reported",
    "optional_reports.identity_theft_phone_reported_date",
    "optional_reports.other",

    # Page 2 relief and investigation.
    "relief.status",
    "relief.bank1",
    "relief.date1",
    "relief.bank2",
    "relief.date2",
    "relief.bank3",
    "relief.date3",
    "investigation.status",
    "investigation.agency",
    "investigation.reported_at",

    # Page 3 full security survey facts.
    "survey.transfer_actor",
    "survey.smishing_link_clicked",
    "survey.smishing_link_clicked_other_text",
    "survey.malicious_app_installed",
    "survey.malicious_app_installed_other_text",
    "survey.provided_id_card",
    "survey.provided_personal_info",
    "survey.provided_device",
    "survey.provided_account_password",
    "survey.provided_security_media",
    "survey.provided_other_financial_info",
    "survey.provided_other_financial_info_text",
    "survey.internet_banking_used",
    "survey.internet_banking_frequency",
    "survey.phone_banking_used",
    "survey.phone_banking_frequency",
    "survey.open_banking_used",
    "survey.open_banking_frequency",
    "survey.id_lent",
    "survey.id_copy_stored_digitally",
    "survey.id_physical_storage_method",
    "survey.phone_lent",
    "survey.phone_lock_method",
    "survey.security_media_storage_method",
    "survey.id_loss_reported_before_incident",
    "survey.id_copy_stored_before_incident",
    "survey.id_copy_stored_before_incident_other_text",
    "survey.account_password_stored",
    "survey.account_password_stored_other_text",
    "survey.phone_lock_enabled",
    "survey.phone_lock_enabled_other_text",
    "survey.personal_info_leak_suspicion_details",

    # Page 4 narrative drafts. The LLM may draft only if the user already gave enough facts.
    "narrative.incident_circumstances",
    "narrative.post_action",

    # Page 7 attachment/evidence statuses and notes.
    *EVIDENCE_STATUS_PATHS,
    *EVIDENCE_NOTE_PATHS,

    # Page 8 delegation/proxy facts.
    "delegation.proxy_used",
    "delegation.agent_name",
    "delegation.agent_birth_date",
    "delegation.agent_phone_number",
    "delegation.agent_mobile_number",
    "delegation.agent_email",
    "delegation.agent_address",
    "delegation.agent_memo",
    "delegation.request_purpose",
}

FRAUD_TYPE_VALUES = {item.value for item in FraudType}
REPORT_STATUS_VALUES = {item.value for item in ReportStatus}
TRANSACTION_TYPE_VALUES = {item.value for item in TransactionType}
EVIDENCE_STATUS_VALUES = {item.value for item in EvidenceStatus}
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
}

# Consent fields are intentionally absent from ALLOWED_PATCH_PATHS.
# Required consent must be explicit and is handled later by answers.py.


def extract_initial_statement(text: str, provider: LLMProvider) -> list[Patch]:
    """Convert the first victim statement into normalized Patch objects.

    The model is responsible for understanding Korean free text.
    This function is responsible for keeping the system safe:
    path allow-list, enum validation, type normalization, and Patch construction.
    """

    clean_text = (text or "").strip()
    if not clean_text:
        return []

    raw_response = provider.generate_json(_build_extraction_prompt(clean_text))
    data = _coerce_llm_json(raw_response)

    patches: list[Patch] = []
    for item in _iter_patch_items(data):
        patch = _patch_from_llm_item(item)
        if patch is not None:
            patches.append(patch)

    return _dedupe_patches_keep_last(patches)


def _build_extraction_prompt(text: str) -> str:
    """Build a strict LLM prompt for initial incident extraction."""

    allowed_paths = "\n".join(f"- {path}" for path in sorted(ALLOWED_PATCH_PATHS))

    return f"""
You are Recover24 V3's initial incident extraction engine.

Your job:
Read the user's first Korean victim statement and return JSON patches for Recover24's RecoveryCase.

Return JSON only. No markdown. No explanation.

Hard rules:
- Extract only facts explicitly stated or strongly implied by the user.
- Do not invent banks, account numbers, dates, agencies, names, consent, or evidence.
- Do not ask questions.
- Do not write HTML.
- Do not summarize unless writing incident.overview or narrative fields from already stated facts.
- Omit absent information. Do not output null patches for information that is simply missing.
- Use only the allowed patch paths below.
- Use enum values exactly as provided.
- Store KRW amounts as integers, e.g. 100만원 -> 1000000.
- Consent fields must not be extracted from the first statement.

Allowed fraud_type values:
- authority_impersonation: 검찰/경찰/금감원/수사기관 사칭
- family_impersonation: 자녀/가족/지인 사칭 메신저피싱
- loan_scam: 대출빙자/저금리/대환대출
- smishing_malware: 스미싱/악성앱/원격제어 앱
- institution_impersonation: 은행/카드사/택배/기관 사칭
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

Allowed evidence item kinds:
- id_card_copy
- police_certificate
- id_loss_evidence
- phone_evidence
- complaint_evidence
- investigation_delegation
- data_leak_notice
- delay_reason
- family_proof
- signature_certificate
- security_survey
- other_evidence
- passport_or_travel_proof

Allowed patch paths:
{allowed_paths}

Output format:
{{
  "patches": [
    {{
      "path": "incident.fraud_type",
      "value": "authority_impersonation",
      "status": "answered",
      "source_text": "검찰이라고 전화가 왔어요",
      "confidence": 0.95
    }}
  ]
}}

Examples:
Input: "검찰 사칭 전화를 받고 모바일뱅킹으로 100만 원을 보냈어요. 아직 경찰 신고는 못 했어요."
Output:
{{
  "patches": [
    {{"path": "incident.fraud_type", "value": "authority_impersonation", "status": "answered", "source_text": "검찰 사칭", "confidence": 0.95}},
    {{"path": "transactions.0.amount_krw", "value": 1000000, "status": "answered", "source_text": "100만 원", "confidence": 0.95}},
    {{"path": "transactions.0.transaction_type", "value": "mobile_banking_transfer", "status": "answered", "source_text": "모바일뱅킹", "confidence": 0.90}},
    {{"path": "investigation.status", "value": "not_reported", "status": "answered", "source_text": "아직 경찰 신고는 못 했어요", "confidence": 0.90}}
  ]
}}

User statement:
{text}
""".strip()


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
    """Coerce provider output into Python JSON-compatible data.

    Providers may return a dict, a JSON string, or text containing a JSON object.
    This function is intentionally defensive because local LLMs often wrap JSON in text.
    """

    if isinstance(raw, (dict, list)):
        return raw

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    if not isinstance(raw, str):
        return {}

    text = raw.strip()
    if not text:
        return {}

    # Remove common markdown fences if the model ignored the instruction.
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


def _patch_from_llm_item(item: dict[str, Any]) -> Patch | None:
    """Validate one LLM patch item and convert it to Patch."""

    path = item.get("path")
    if not isinstance(path, str) or path not in ALLOWED_PATCH_PATHS:
        return None

    status = _normalize_status(item.get("status", FieldStatus.ANSWERED.value))
    if status is None:
        return None

    raw_value = item.get("value")

    # Initial extraction usually omits absent fields.
    # But if the user explicitly said "I don't know" or "not applicable", keep that state.
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

    # String-like fields: keep raw Korean facts as user-friendly strings.
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, (int, float, bool)):
        return str(value)

    return None


def _normalize_enum(
    value: Any,
    enum_cls: type[FraudType] | type[ReportStatus] | type[TransactionType] | type[EvidenceStatus],
) -> Any:
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    try:
        return enum_cls(normalized)
    except ValueError:
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
    """Parse common KRW amount expressions.

    Supports examples like:
    - 100만원
    - 100만 원
    - 1,000,000원
    - 1000000원
    - 1억 2천만 원

    This is not the main extractor. It is a normalizer for LLM-provided amount text.
    """

    value = text.strip().replace(",", "").replace(" ", "")
    if not value:
        return None

    # Plain won: 1000000원
    plain_won = re.search(r"(\d+)원$", value)
    if plain_won and "만" not in value and "억" not in value and "천" not in value:
        return int(plain_won.group(1))

    total = 0

    # 1억
    eok = re.search(r"(\d+)억", value)
    if eok:
        total += int(eok.group(1)) * 100_000_000

    # 2천만 / 2천만원 means 20,000,000
    cheon_man = re.search(r"(\d+)천만", value)
    if cheon_man:
        total += int(cheon_man.group(1)) * 10_000_000

    # 100만
    man = re.search(r"(\d+)만", value)
    if man:
        # Avoid double-counting the 만 part inside 천만.
        if not re.search(r"\d+천만", value):
            total += int(man.group(1)) * 10_000

    if total > 0:
        return total

    # Bare digits from model, e.g. "1000000"
    if re.fullmatch(r"\d+", value):
        return int(value)

    return None


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "예", "네", "맞음", "있음", "했다", "동의", "동의함"}:
            return True
        if normalized in {"false", "no", "n", "0", "아니오", "아니요", "없음", "안함", "안 했음", "미동의"}:
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
        confidence = float(value)
    elif isinstance(value, str):
        try:
            confidence = float(value.strip())
        except ValueError:
            return None
    else:
        return None

    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def _dedupe_patches_keep_last(patches: list[Patch]) -> list[Patch]:
    """If the model emits the same path multiple times, keep the last valid patch."""

    by_path: dict[str, Patch] = {}
    order: list[str] = []

    for patch in patches:
        if patch.path not in by_path:
            order.append(patch.path)
        by_path[patch.path] = patch

    return [by_path[path] for path in order]
