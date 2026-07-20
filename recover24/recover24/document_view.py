"""Build official HTML-ready document view for Recover24 V3.

Input: RecoveryCase raw facts.
Output: a dict whose keys match templates/recover24_official_report_v1.html.

Rules:
- Do not mutate RecoveryCase.
- Do not call an LLM.
- Do not infer new facts from text.
- Do not render HTML here; html_renderer.py owns template rendering.
- Convert raw values into display labels, money strings, and checkbox marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any, Iterable

from .catalogs import (
    EVIDENCE_STATUS_LABELS,
    EVIDENCE_TEMPLATE_KEYS,
    FRAUD_TYPE_LABELS,
    REPORT_STATUS_LABELS,
    TRANSACTION_TYPE_LABELS,
)
from .models import (
    EvidenceItem,
    EvidenceStatus,
    FieldStatus,
    FieldValue,
    FraudType,
    RecoveryCase,
    ReportStatus,
    Transaction,
    TransactionType,
)

CHECKED = "☑"
UNCHECKED = "☐"
UNKNOWN_LABEL = "미확인"
NOT_APPLICABLE_LABEL = "해당없음"


@dataclass(frozen=True, slots=True)
class OfficialDocumentView:
    """Small wrapper around the HTML context dict.

    Keeping a class here gives callers an explicit type while still letting
    html_renderer.py receive a normal Jinja-compatible dictionary via to_dict().
    """

    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.context


def build_document_view(case: RecoveryCase, today: date | None = None) -> OfficialDocumentView:
    """Build every display value required by the official HTML template."""

    today_value = today or date.today()
    context: dict[str, Any] = {
        "applicant": _build_applicant_view(case),
        "checkbox": _build_checkbox_view(case),
        "accident": _build_accident_view(case),
        "transactions_padded": _build_transactions_padded(case),
        "optionalReports": _build_optional_reports_view(case),
        "relief": _build_relief_view(case),
        "investigation": _build_investigation_view(case),
        "survey": _build_survey_view(case),
        "providedAccountPassword": _checked_if_true(case.survey.provided_account_password),
        "narrative": _build_narrative_view(case),
        "consent": _build_consent_view(case),
        "attachmentRemarks": _build_attachment_remarks_view(case),
        "delegation": _build_delegation_view(case),
        "showDelegation": _should_show_delegation(case),
        "today": {"year": today_value.year, "month": today_value.month, "day": today_value.day},
    }
    return OfficialDocumentView(context=context)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_applicant_view(case: RecoveryCase) -> dict[str, str]:
    applicant = case.applicant
    return {
        "name": _display_text(applicant.name),
        "birthDate": _display_text(applicant.birth_date),
        "customerNumber": _display_text(applicant.customer_number, not_applicable=""),
        "companyName": _display_text(applicant.company_name),
        "businessNumber": _display_text(applicant.business_number),
        "phoneNumber": _display_text(applicant.phone_number, not_applicable=""),
        "mobileNumber": _display_text(applicant.mobile_number),
        "email": _display_text(applicant.email, not_applicable=""),
        "address": _display_text(applicant.address),
        "memo": _display_text(applicant.memo, not_applicable=""),
    }


def _build_checkbox_view(case: RecoveryCase) -> dict[str, str]:
    view: dict[str, str] = {}

    sms_yes, sms_no = _yes_no(case.applicant.sms_consent)
    view["smsConsentYes"] = sms_yes
    view["smsConsentNo"] = sms_no

    for i in range(1, 18):
        field_value = case.exclusion.items.get(f"exclude_{i}")
        view[f"exclude{i}"] = _checked_if_true(field_value)

    final_yes, final_no = _yes_no(case.exclusion.final_has_exclusion)
    # HTML wording: 해당없음 / 해당됨. True means at least one exclusion applies.
    view["finalExcludeNo"] = final_no
    view["finalExcludeYes"] = final_yes
    return view


def _build_accident_view(case: RecoveryCase) -> dict[str, str]:
    incident = case.incident
    return {
        "firstIncidentAt": _display_datetime_text(incident.first_occurred_at),
        "firstDiscoveredAt": _display_datetime_text(incident.recognized_at),
        "firstFreezeAt": _display_datetime_text(incident.first_freeze_at),
        "totalDamageAmountLabel": _money(_total_damage_amount(case)),
        "incidentTypeLabel": _fraud_type_label(incident.fraud_type),
        "incidentOverview": _auto_incident_overview(case) or _display_text(incident.overview),
    }


def _build_transactions_padded(case: RecoveryCase, minimum_rows: int = 3) -> list[dict[str, str]]:
    rows = [_build_transaction_view(tx) for tx in case.transactions]
    while len(rows) < minimum_rows:
        rows.append(_empty_transaction_view())
    return rows


def _build_transaction_view(tx: Transaction) -> dict[str, str]:
    return {
        "sourceBank": _display_text(tx.source_bank),
        "sourceAccountNumber": _display_text(tx.source_account_number),
        "amountLabel": _money(_field_raw_value(tx.amount_krw)),
        "destinationBank": _display_text(tx.destination_bank),
        "destinationAccountNumber": _display_text(tx.destination_account_number),
        "destinationAccountHolder": _display_text(tx.destination_account_holder),
        "holderType": _display_text(tx.holder_type),
        "transactionType": _transaction_type_label(tx.transaction_type),
    }


def _empty_transaction_view() -> dict[str, str]:
    return {
        "sourceBank": "",
        "sourceAccountNumber": "",
        "amountLabel": "",
        "destinationBank": "",
        "destinationAccountNumber": "",
        "destinationAccountHolder": "",
        "holderType": "",
        "transactionType": "",
    }


def _build_optional_reports_view(case: RecoveryCase) -> dict[str, str]:
    optional = case.optional_reports
    id_yes, id_no = _yes_no_optional(optional.id_loss_reported)
    phone_yes, phone_no = _yes_no_optional(optional.phone_loss_reported)
    identity_yes, identity_no = _yes_no_optional(optional.identity_theft_phone_reported)
    return {
        "idLossReportedYes": id_yes,
        "idLossReportedNo": id_no,
        "idLossReportedDate": _optional_date_for_bool(optional.id_loss_reported, optional.id_loss_reported_date),
        "phoneLossReportedYes": phone_yes,
        "phoneLossReportedNo": phone_no,
        "phoneLossReportedDate": _optional_date_for_bool(optional.phone_loss_reported, optional.phone_loss_reported_date),
        "identityTheftPhoneReportedYes": identity_yes,
        "identityTheftPhoneReportedNo": identity_no,
        "identityTheftPhoneReportedDate": _optional_date_for_bool(optional.identity_theft_phone_reported, optional.identity_theft_phone_reported_date),
        "other": _display_text(optional.other, not_applicable=""),
    }


def _build_relief_view(case: RecoveryCase) -> dict[str, str]:
    status = _field_raw_value(case.relief.status)
    return {
        "statusPlanned": _checked_if(status == ReportStatus.PLANNED),
        "statusInProgress": _checked_if(status == ReportStatus.IN_PROGRESS),
        "statusCompleted": _checked_if(status in {ReportStatus.COMPLETED, ReportStatus.CLOSED}),
        "statusOther": _checked_if(status in {ReportStatus.OTHER, ReportStatus.UNKNOWN, ReportStatus.NOT_REPORTED}),
        "statusLabel": _report_status_label(case.relief.status),
        "bank1": _display_text(case.relief.bank1),
        "date1": _display_date_text(case.relief.date1),
        "bank2": _display_text(case.relief.bank2, not_applicable=""),
        "date2": _display_date_text(case.relief.date2, not_applicable=""),
        "bank3": _display_text(case.relief.bank3, not_applicable=""),
        "date3": _display_date_text(case.relief.date3, not_applicable=""),
        "rows": _relief_rows(case),
    }


def _build_investigation_view(case: RecoveryCase) -> dict[str, str]:
    status = _field_raw_value(case.investigation.status)
    return {
        "statusNotReported": _checked_if(status == ReportStatus.NOT_REPORTED),
        "statusReported": _checked_if(status == ReportStatus.REPORTED),
        "statusInProgress": _checked_if(status == ReportStatus.IN_PROGRESS),
        "statusClosed": _checked_if(status in {ReportStatus.CLOSED, ReportStatus.COMPLETED}),
        "statusOther": _checked_if(status in {ReportStatus.OTHER, ReportStatus.UNKNOWN, ReportStatus.PLANNED}),
        "statusLabel": _report_status_label(case.investigation.status),
        "agency": _display_text(case.investigation.agency),
        "reportedAt": _display_date_text(case.investigation.reported_at),
    }


def _build_survey_view(case: RecoveryCase) -> dict[str, str]:
    survey = case.survey
    transfer_actor = _display_text(survey.transfer_actor).replace(" ", "")

    smishing_yes, smishing_no, smishing_other = _yes_no_other(
        survey.smishing_link_clicked, survey.smishing_link_clicked_other_text
    )
    app_yes, app_no, app_other = _yes_no_other(
        survey.malicious_app_installed, survey.malicious_app_installed_other_text
    )
    id_loss_yes, id_loss_no, id_loss_na = _yes_no_na(survey.id_loss_reported_before_incident)
    id_copy_yes, id_copy_no, id_copy_other = _yes_no_other(
        survey.id_copy_stored_before_incident, survey.id_copy_stored_before_incident_other_text
    )
    account_pw_yes, account_pw_no, account_pw_other = _yes_no_other(
        survey.account_password_stored, survey.account_password_stored_other_text
    )
    phone_lock_yes, phone_lock_no, phone_lock_other = _yes_no_other(
        survey.phone_lock_enabled, survey.phone_lock_enabled_other_text
    )
    id_lent_yes, id_lent_no = _yes_no(survey.id_lent)
    id_copy_digital_yes, id_copy_digital_no = _yes_no(survey.id_copy_stored_digitally)
    phone_lent_yes, phone_lent_no = _yes_no(survey.phone_lent)

    return {
        "transferActorSelf": _checked_if(any(token in transfer_actor for token in ["본인", "손님본인", "self"])),
        "transferActorFamily": _checked_if(any(token in transfer_actor for token in ["가족", "family"])),
        "transferActorAcquaintance": _checked_if(any(token in transfer_actor for token in ["지인", "acquaintance"])),
        "transferActorUnknownThirdParty": _checked_if(any(token in transfer_actor for token in ["불상", "제3자", "unknown"])),
        "smishingLinkClickedYes": smishing_yes,
        "smishingLinkClickedNo": smishing_no,
        "smishingLinkClickedOther": smishing_other,
        "smishingLinkClickedOtherText": _display_text(survey.smishing_link_clicked_other_text, not_applicable=""),
        "maliciousAppInstalledYes": app_yes,
        "maliciousAppInstalledNo": app_no,
        "maliciousAppInstalledOther": app_other,
        "maliciousAppInstalledOtherText": _display_text(survey.malicious_app_installed_other_text, not_applicable=""),
        "providedIdCard": _checked_if_true(survey.provided_id_card),
        "providedPersonalInfo": _checked_if_true(survey.provided_personal_info),
        "providedDevice": _checked_if_true(survey.provided_device),
        "providedSecurityMedia": _checked_if_true(survey.provided_security_media),
        "providedOtherFinancialInfo": _checked_if_true(survey.provided_other_financial_info),
        "providedOtherFinancialInfoText": _display_text(survey.provided_other_financial_info_text, not_applicable=""),
        "internetBankingUsed": _ox(survey.internet_banking_used),
        "internetBankingFrequency": _frequency_display(survey.internet_banking_used, survey.internet_banking_frequency),
        "phoneBankingUsed": _ox(survey.phone_banking_used),
        "phoneBankingFrequency": _frequency_display(survey.phone_banking_used, survey.phone_banking_frequency),
        "openBankingUsed": _ox(survey.open_banking_used),
        "openBankingFrequency": _frequency_display(survey.open_banking_used, survey.open_banking_frequency),
        "idLentYes": id_lent_yes,
        "idLentNo": id_lent_no,
        "idCopyStoredDigitallyYes": id_copy_digital_yes,
        "idCopyStoredDigitallyNo": id_copy_digital_no,
        "idPhysicalStorageMethod": _display_text(survey.id_physical_storage_method),
        "phoneLentYes": phone_lent_yes,
        "phoneLentNo": phone_lent_no,
        "phoneLockMethod": _display_text(survey.phone_lock_method),
        "securityMediaStorageMethod": _display_text(survey.security_media_storage_method),
        "idLossReportedYes": id_loss_yes,
        "idLossReportedNo": id_loss_no,
        "idLossReportedNotApplicable": id_loss_na,
        "idCopyStoredBeforeIncidentYes": id_copy_yes,
        "idCopyStoredBeforeIncidentNo": id_copy_no,
        "idCopyStoredBeforeIncidentOther": id_copy_other,
        "idCopyStoredBeforeIncidentOtherText": _display_text(survey.id_copy_stored_before_incident_other_text, not_applicable=""),
        "accountPasswordStoredYes": account_pw_yes,
        "accountPasswordStoredNo": account_pw_no,
        "accountPasswordStoredOther": account_pw_other,
        "accountPasswordStoredOtherText": _display_text(survey.account_password_stored_other_text, not_applicable=""),
        "phoneLockEnabledYes": phone_lock_yes,
        "phoneLockEnabledNo": phone_lock_no,
        "phoneLockEnabledOther": phone_lock_other,
        "phoneLockEnabledOtherText": _display_text(survey.phone_lock_enabled_other_text, not_applicable=""),
        "personalInfoLeakSuspicionDetails": _display_text(survey.personal_info_leak_suspicion_details, not_applicable=""),
    }


def _build_narrative_view(case: RecoveryCase) -> dict[str, str]:
    incident_text = _display_text(case.narrative.incident_circumstances)
    post_text = _display_text(case.narrative.post_action)
    if not post_text or post_text == incident_text:
        post_text = _auto_post_action(case) or post_text
    return {
        "incidentNarrativeDraft": incident_text,
        "postActionNarrativeDraft": post_text,
    }


def _build_consent_view(case: RecoveryCase) -> dict[str, str]:
    unique_collection_no, unique_collection_yes = _disagree_agree(case.consent.unique_id_collection_agreed)
    personal_collection_no, personal_collection_yes = _disagree_agree(case.consent.personal_credit_collection_agreed)
    unique_provision_no, unique_provision_yes = _disagree_agree(case.consent.unique_id_provision_agreed)
    personal_provision_no, personal_provision_yes = _disagree_agree(case.consent.personal_credit_provision_agreed)
    return {
        "uniqueIdCollectionDisagree": unique_collection_no,
        "uniqueIdCollectionAgree": unique_collection_yes,
        "personalCreditCollectionDisagree": personal_collection_no,
        "personalCreditCollectionAgree": personal_collection_yes,
        "uniqueIdProvisionDisagree": unique_provision_no,
        "uniqueIdProvisionAgree": unique_provision_yes,
        "personalCreditProvisionDisagree": personal_provision_no,
        "personalCreditProvisionAgree": personal_provision_yes,
    }


def _build_attachment_remarks_view(case: RecoveryCase) -> dict[str, str]:
    evidence_by_kind = {item.kind: item for item in case.evidence}
    remarks: dict[str, str] = {}
    for kind, template_key in EVIDENCE_TEMPLATE_KEYS.items():
        remarks[template_key] = _evidence_remark(evidence_by_kind.get(kind))
    return remarks


def _build_delegation_view(case: RecoveryCase) -> dict[str, str]:
    delegation = case.delegation
    return {
        "agentName": _display_text(delegation.agent_name),
        "agentBirthDate": _display_text(delegation.agent_birth_date),
        "agentPhoneNumber": _display_text(delegation.agent_phone_number),
        "agentMobileNumber": _display_text(delegation.agent_mobile_number),
        "agentEmail": _display_text(delegation.agent_email),
        "agentAddress": _display_text(delegation.agent_address),
        "agentMemo": _display_text(delegation.agent_memo),
        "requestPurpose": _display_text(delegation.request_purpose, unknown="", not_applicable=""),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------




def _should_show_delegation(case: RecoveryCase) -> bool:
    return case.delegation.proxy_used.status == FieldStatus.ANSWERED and case.delegation.proxy_used.value is True


def _relief_rows(case: RecoveryCase) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx in range(1, 4):
        bank = _display_text(getattr(case.relief, f"bank{idx}"), unknown=UNKNOWN_LABEL, not_applicable="")
        date_value = _display_date_text(getattr(case.relief, f"date{idx}"), unknown=UNKNOWN_LABEL, not_applicable="")
        if bank or date_value:
            rows.append({"index": idx, "bank": bank or UNKNOWN_LABEL, "date": date_value or UNKNOWN_LABEL})
    if not rows:
        rows.append({"index": 1, "bank": "", "date": ""})
    return rows


def _auto_incident_overview(case: RecoveryCase) -> str:
    tx = case.transactions[0] if case.transactions else None
    if tx is None:
        return ""
    amount = _money(_field_raw_value(tx.amount_krw))
    source = _display_text(tx.source_bank, unknown="", not_applicable="")
    dest = _display_text(tx.destination_bank, unknown="", not_applicable="")
    fraud = _fraud_type_label(case.incident.fraud_type)
    if not any([amount, source, dest, fraud]):
        return ""
    pieces = []
    if fraud:
        pieces.append(f"{fraud} 피해로")
    if source or dest or amount:
        transfer = ""
        if source and dest:
            transfer = f"{source}에서 {dest}로"
        elif source:
            transfer = f"{source}에서"
        elif dest:
            transfer = f"{dest} 계좌로"
        if amount:
            transfer = f"{transfer} {amount}을 송금".strip()
        else:
            transfer = f"{transfer} 송금".strip()
        pieces.append(transfer)
    if not pieces:
        return ""
    return " ".join(pieces) + "한 사건입니다."


def _auto_post_action(case: RecoveryCase) -> str:
    actions: list[str] = []
    recognized = _display_datetime_text(case.incident.recognized_at, unknown="", not_applicable="")
    freeze = _display_datetime_text(case.incident.first_freeze_at, unknown="", not_applicable="")
    if recognized:
        actions.append(f"신청인은 {recognized}경 사고 사실을 인지하였습니다." if not recognized.endswith("경") else f"신청인은 {recognized} 사고 사실을 인지하였습니다.")
    if freeze:
        actions.append(f"이후 {freeze} 금융기관에 계좌 지급정지 또는 거래제한을 요청하였습니다." if not freeze.endswith("경") else f"이후 {freeze} 금융기관에 계좌 지급정지 또는 거래제한을 요청하였습니다.")
    inv_status = _field_raw_value(case.investigation.status)
    agency = _display_text(case.investigation.agency, unknown="", not_applicable="")
    reported = _display_date_text(case.investigation.reported_at, unknown="", not_applicable="")
    if inv_status in {ReportStatus.REPORTED, ReportStatus.IN_PROGRESS, ReportStatus.COMPLETED, ReportStatus.CLOSED}:
        if agency and reported:
            actions.append(f"또한 {reported} {agency}에 피해 사실을 신고하였습니다.")
        elif agency:
            actions.append(f"또한 {agency}에 피해 사실을 신고하였습니다.")
        else:
            actions.append("또한 수사기관에 피해 사실을 신고하였습니다.")
    relief_status = _field_raw_value(case.relief.status)
    if relief_status == ReportStatus.PLANNED:
        actions.append("피해구제 환급 신청은 신청 예정입니다.")
    elif relief_status in {ReportStatus.REPORTED, ReportStatus.IN_PROGRESS}:
        actions.append("피해구제 환급 신청을 접수하였거나 진행 중입니다.")
    return " ".join(actions)


def _optional_date_for_bool(flag: FieldValue[bool], date_field: FieldValue[str]) -> str:
    if flag.status == FieldStatus.ANSWERED and flag.value is True:
        return _display_date_text(date_field)
    if flag.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if flag.status in {FieldStatus.ANSWERED, FieldStatus.NOT_APPLICABLE}:
        return "-"
    return ""


def _frequency_display(used: FieldValue[bool], frequency: FieldValue[str]) -> str:
    if used.status == FieldStatus.ANSWERED and used.value is True:
        value = _display_text(frequency, unknown=UNKNOWN_LABEL, not_applicable=UNKNOWN_LABEL)
        return value if value else UNKNOWN_LABEL
    if used.status == FieldStatus.ANSWERED and used.value is False:
        return "-"
    if used.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if used.status == FieldStatus.NOT_APPLICABLE:
        return "-"
    return ""

def _field_raw_value(field: FieldValue[Any]) -> Any:
    if field.status == FieldStatus.ANSWERED:
        return field.value
    return None


def _display_text(
    field: FieldValue[Any],
    *,
    default: str = "",
    unknown: str = UNKNOWN_LABEL,
    not_applicable: str = NOT_APPLICABLE_LABEL,
) -> str:
    if field.status == FieldStatus.ANSWERED:
        if field.value is None:
            return default
        value = field.value
        if hasattr(value, "value"):
            value = value.value
        return str(value)
    if field.status == FieldStatus.UNKNOWN:
        return unknown
    if field.status == FieldStatus.NOT_APPLICABLE:
        return not_applicable
    return default



def _display_datetime_text(
    field: FieldValue[Any],
    *,
    default: str = "",
    unknown: str = UNKNOWN_LABEL,
    not_applicable: str = NOT_APPLICABLE_LABEL,
) -> str:
    base = _display_text(field, default=default, unknown=unknown, not_applicable=not_applicable)
    if base in {default, unknown, not_applicable, ""}:
        return base
    return _normalize_date_time_for_document(base, allow_time=True)


def _display_date_text(
    field: FieldValue[Any],
    *,
    default: str = "",
    unknown: str = UNKNOWN_LABEL,
    not_applicable: str = NOT_APPLICABLE_LABEL,
) -> str:
    base = _display_text(field, default=default, unknown=unknown, not_applicable=not_applicable)
    if base in {default, unknown, not_applicable, ""}:
        return base
    return _normalize_date_time_for_document(base, allow_time=False)


def _normalize_date_time_for_document(text: str, *, allow_time: bool) -> str:
    value = str(text).strip()
    compact = re.sub(r"\s+", "", value)
    if any(token in compact for token in ("몰라", "모름", "모르겠", "미확인")):
        return UNKNOWN_LABEL
    if any(token in compact for token in ("해당없", "없음", "아직요청전", "미신고")):
        return NOT_APPLICABLE_LABEL if "해당" in compact else UNKNOWN_LABEL

    if re.search(r"\b\d{1,2}:\d{3,}\b", value):
        return UNKNOWN_LABEL

    # 2026-06-13 / 2026년 6월 13일 / 2026.6.13 plus optional 15:33 or 15시 33분
    pattern = re.compile(
        r"(?P<y>20\d{2})\s*(?:년|[-./])\s*(?P<m>\d{1,2})\s*(?:월|[-./])\s*(?P<d>\d{1,2})\s*(?:일)?"
        r"(?:\s*(?P<h>\d{1,2})\s*(?:시|:)?\s*(?P<mi>\d{0,2})?\s*(?:분)?\s*(?P<approx>경|쯤|정도|\?)?)?"
    )
    match = pattern.search(value)
    if not match:
        return value

    y = int(match.group("y"))
    m = int(match.group("m"))
    d = int(match.group("d"))
    h_raw = match.group("h")
    mi_raw = match.group("mi")
    approx = match.group("approx") or ("경" if "경" in value or "쯤" in value else "")

    if not (1 <= m <= 12 and 1 <= d <= 31):
        return UNKNOWN_LABEL

    if not allow_time or h_raw is None:
        return f"{y:04d}년 {m:02d}월 {d:02d}일"

    h = int(h_raw)
    if not (0 <= h <= 23):
        return UNKNOWN_LABEL
    if mi_raw in {None, ""}:
        mi = 0
    else:
        if len(mi_raw) > 2:
            return UNKNOWN_LABEL
        mi = int(mi_raw)
    if not (0 <= mi <= 59):
        return UNKNOWN_LABEL
    return f"{y:04d}년 {m:02d}월 {d:02d}일 {h:02d}시 {mi:02d}분경"

def _money(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}원"
    return ""


def _total_damage_amount(case: RecoveryCase) -> int | None:
    amounts = [tx.amount_krw.value for tx in case.transactions if tx.amount_krw.has_answer]
    amounts = [amount for amount in amounts if isinstance(amount, int) and not isinstance(amount, bool)]
    if not amounts:
        return None
    return sum(amounts)


def _fraud_type_label(field: FieldValue[FraudType]) -> str:
    if field.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if field.status == FieldStatus.NOT_APPLICABLE:
        return NOT_APPLICABLE_LABEL
    value = _field_raw_value(field)
    return FRAUD_TYPE_LABELS.get(value, "") if isinstance(value, FraudType) else ""


def _transaction_type_label(field: FieldValue[TransactionType]) -> str:
    if field.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if field.status == FieldStatus.NOT_APPLICABLE:
        return NOT_APPLICABLE_LABEL
    value = _field_raw_value(field)
    return TRANSACTION_TYPE_LABELS.get(value, "") if isinstance(value, TransactionType) else ""


def _report_status_label(field: FieldValue[ReportStatus]) -> str:
    if field.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if field.status == FieldStatus.NOT_APPLICABLE:
        return NOT_APPLICABLE_LABEL
    value = _field_raw_value(field)
    return REPORT_STATUS_LABELS.get(value, "") if isinstance(value, ReportStatus) else ""


def _checked_if(condition: bool) -> str:
    return CHECKED if condition else UNCHECKED


def _checked_if_true(field: FieldValue[bool] | None) -> str:
    return CHECKED if isinstance(field, FieldValue) and field.status == FieldStatus.ANSWERED and field.value is True else UNCHECKED


def _yes_no(field: FieldValue[bool]) -> tuple[str, str]:
    if field.status != FieldStatus.ANSWERED:
        return UNCHECKED, UNCHECKED
    return _checked_if(field.value is True), _checked_if(field.value is False)


def _yes_no_optional(field: FieldValue[bool]) -> tuple[str, str]:
    if field.status == FieldStatus.NOT_APPLICABLE:
        return UNCHECKED, CHECKED
    return _yes_no(field)


def _yes_no_na(field: FieldValue[bool]) -> tuple[str, str, str]:
    if field.status == FieldStatus.NOT_APPLICABLE:
        return UNCHECKED, UNCHECKED, CHECKED
    yes, no = _yes_no(field)
    return yes, no, UNCHECKED


def _yes_no_other(field: FieldValue[bool], other_text: FieldValue[str]) -> tuple[str, str, str]:
    if other_text.has_answer:
        return UNCHECKED, UNCHECKED, CHECKED
    yes, no = _yes_no(field)
    return yes, no, UNCHECKED


def _disagree_agree(field: FieldValue[bool]) -> tuple[str, str]:
    if field.status != FieldStatus.ANSWERED:
        return UNCHECKED, UNCHECKED
    return _checked_if(field.value is False), _checked_if(field.value is True)


def _ox(field: FieldValue[bool]) -> str:
    if field.status == FieldStatus.ANSWERED:
        return "O" if field.value is True else "X"
    if field.status == FieldStatus.UNKNOWN:
        return UNKNOWN_LABEL
    if field.status == FieldStatus.NOT_APPLICABLE:
        return NOT_APPLICABLE_LABEL
    return ""


def _evidence_remark(item: EvidenceItem | None) -> str:
    if item is None:
        return UNKNOWN_LABEL
    label = EVIDENCE_STATUS_LABELS.get(item.status, "")
    if item.note:
        return f"{label} - {item.note}" if label else item.note
    return label
