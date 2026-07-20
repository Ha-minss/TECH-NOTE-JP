"""Question generation for Recover24 V3.

Input: RecoveryCase.
Output: Question[] for missing raw facts.

questions.py decides *what to ask next* from RecoveryCase state.
It does not parse user answers, call an LLM, mutate RecoveryCase, or render HTML.

Core rules:
- Only FieldStatus.NOT_ASKED fields are question targets.
- Fields already answered, unknown, or not_applicable are not asked again.
- Question prompts are built from missing field labels, not from hard-coded
  combinations of every possible missing-field case.
- Every user-answerable FieldValue path in models.py must have a label here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable

from .models import (
    EvidenceStatus,
    FieldStatus,
    FieldValue,
    Question,
    QuestionCategory,
    QuestionInputType,
    RecoveryCase,
    ReportStatus,
)


@dataclass(frozen=True, slots=True)
class QuestionGroup:
    """A reusable question rule for a coherent block of missing fields."""

    question_id: str
    category: QuestionCategory
    candidate_paths: tuple[str, ...]
    labels: dict[str, str]
    intro: str
    single_intro: str | None = None
    required: bool = True
    input_type: QuestionInputType = QuestionInputType.FORM


# ---------------------------------------------------------------------------
# Field labels: this is the official dictionary used by questions.py.
# If models.py adds a user-answerable FieldValue path, it should be added here.
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, str] = {
    # Page 1: applicant
    "applicant.name": "성명",
    "applicant.birth_date": "생년월일",
    "applicant.customer_number": "고객번호",
    "applicant.company_name": "법인명",
    "applicant.business_number": "사업자등록번호",
    "applicant.phone_number": "전화번호",
    "applicant.mobile_number": "휴대전화번호",
    "applicant.email": "이메일",
    "applicant.address": "주소",
    "applicant.memo": "기타 메모",
    "applicant.sms_consent": "SMS 수신 동의 여부",

    # Page 1-2: exclusion checklist
    "exclusion.items.exclude_1": "이용자 본인이 직접 지급지시한 금융거래 해당 여부",
    "exclusion.items.exclude_2": "동거가족 또는 지인에 의한 거래 해당 여부",
    "exclusion.items.exclude_3": "접근매체 양도/양수/질권 설정 관련 거래 해당 여부",
    "exclusion.items.exclude_4": "법인 이용자의 기관/피용자로서 법인을 위한 거래 해당 여부",
    "exclusion.items.exclude_5": "재화 공급을 가장한 상거래 해당 여부",
    "exclusion.items.exclude_6": "용역 제공을 가장한 거래 해당 여부",
    "exclusion.items.exclude_7": "불법적이거나 비정상적인 재화/용역 관련 거래 해당 여부",
    "exclusion.items.exclude_8": "간편송금업체를 통한 금융거래 해당 여부",
    "exclusion.items.exclude_9": "영업점 창구를 통한 대면 금융거래 해당 여부",
    "exclusion.items.exclude_10": "피해 예방 안내에도 정상 거래 주장으로 발생한 피해 해당 여부",
    "exclusion.items.exclude_11": "이전 피해구제/전자금융거래 사고 배상 신청 여부",
    "exclusion.items.exclude_12": "카드/현금서비스/카드론 등 여신전문금융업 관련 거래 해당 여부",
    "exclusion.items.exclude_13": "전자금융사고로 보기 어려운 경우 해당 여부",
    "exclusion.items.exclude_14": "소송 등 법적 분쟁 진행/판결 확정 여부",
    "exclusion.items.exclude_15": "이용자와 은행 사이 화해/합의 성립 여부",
    "exclusion.items.exclude_16": "수사기관 등이 전자금융사고가 아니라고 판단한 여부",
    "exclusion.items.exclude_17": "기타 피해 신고 제외대상 해당 여부",
    "exclusion.final_has_exclusion": "피해 신고 제외대상 최종 해당 여부",

    # Page 2: incident
    "incident.first_occurred_at": "최초 사고발생 일시",
    "incident.recognized_at": "최초 사고인지 일시",
    "incident.first_freeze_at": "최초 계좌 지급정지/거래제한 요청 일시",
    "incident.fraud_type": "피해유형",
    "incident.overview": "사건개요",

    # Page 2: transaction row 0
    "transactions.0.source_bank": "출금은행",
    "transactions.0.source_account_number": "출금계좌번호",
    "transactions.0.amount_krw": "출금/피해금액",
    "transactions.0.destination_bank": "입금은행",
    "transactions.0.destination_account_number": "입금/상대계좌번호",
    "transactions.0.destination_account_holder": "수취인/예금주 이름",
    "transactions.0.holder_type": "본인/타인 여부",
    "transactions.0.transaction_type": "거래유형",
    "transactions.0.transferred_at": "송금시각",

    # Page 2: optional reports
    "optional_reports.id_loss_reported": "신분증 분실 신고 여부",
    "optional_reports.id_loss_reported_date": "신분증 분실 신고일",
    "optional_reports.phone_loss_reported": "휴대전화 분실 신고 여부",
    "optional_reports.phone_loss_reported_date": "휴대전화 분실 신고일",
    "optional_reports.identity_theft_phone_reported": "명의도용 휴대전화 개설 신고 여부",
    "optional_reports.identity_theft_phone_reported_date": "명의도용 휴대전화 개설 신고일",
    "optional_reports.other": "선택기재 기타 사항",

    # Page 2: relief and investigation
    "relief.status": "피해구제 신청 상태",
    "relief.bank1": "피해구제 신청은행 1",
    "relief.date1": "피해구제 신청일 1",
    "relief.bank2": "피해구제 신청은행 2",
    "relief.date2": "피해구제 신청일 2",
    "relief.bank3": "피해구제 신청은행 3",
    "relief.date3": "피해구제 신청일 3",
    "investigation.status": "경찰/수사기관 신고 상태",
    "investigation.agency": "수사기관명",
    "investigation.reported_at": "경찰/수사기관 신고일",

    # Page 3: survey
    "survey.transfer_actor": "전자금융거래 실행자",
    "survey.smishing_link_clicked": "스미싱/문자/카톡 링크 클릭 여부",
    "survey.smishing_link_clicked_other_text": "링크 클릭 관련 기타 내용",
    "survey.malicious_app_installed": "악성앱 또는 원격제어 앱 설치 여부",
    "survey.malicious_app_installed_other_text": "악성앱 설치 관련 기타 내용",
    "survey.provided_id_card": "신분증 제공 여부",
    "survey.provided_personal_info": "개인정보 제공 여부",
    "survey.provided_device": "전자적 장치 제공 여부",
    "survey.provided_account_password": "계좌번호/계좌 비밀번호 제공 여부",
    "survey.provided_security_media": "OTP/보안카드/인증서 제공 여부",
    "survey.provided_other_financial_info": "기타 전자금융 관련정보 제공 여부",
    "survey.provided_other_financial_info_text": "기타 제공한 전자금융 관련정보 내용",
    "survey.internet_banking_used": "인터넷뱅킹 사용 여부",
    "survey.internet_banking_frequency": "인터넷뱅킹 사용 빈도",
    "survey.phone_banking_used": "폰뱅킹 사용 여부",
    "survey.phone_banking_frequency": "폰뱅킹 사용 빈도",
    "survey.open_banking_used": "오픈뱅킹 사용 여부",
    "survey.open_banking_frequency": "오픈뱅킹 사용 빈도",
    "survey.id_lent": "가족/지인에게 신분증 대여 경험 여부",
    "survey.id_copy_stored_digitally": "신분증 사본 디지털 저장 여부",
    "survey.id_physical_storage_method": "신분증 실물 보관 방식",
    "survey.phone_lent": "가족/지인에게 휴대전화 제공/대여 경험 여부",
    "survey.phone_lock_method": "평상시 휴대전화 잠금 방식",
    "survey.security_media_storage_method": "보안카드 등 접근매체 보관 방식",
    "survey.id_loss_reported_before_incident": "사고 이전 신분증 분실신고 여부",
    "survey.id_copy_stored_before_incident": "사고 이전 신분증 사본 저장 여부",
    "survey.id_copy_stored_before_incident_other_text": "사고 이전 신분증 사본 저장 관련 기타 내용",
    "survey.account_password_stored": "사고 이전 계좌번호/비밀번호 저장 여부",
    "survey.account_password_stored_other_text": "계좌번호/비밀번호 저장 관련 기타 내용",
    "survey.phone_lock_enabled": "사고 이전 휴대전화 잠금 설정 여부",
    "survey.phone_lock_enabled_other_text": "휴대전화 잠금 설정 관련 기타 내용",
    "survey.personal_info_leak_suspicion_details": "개인정보 유출 의심 정황 및 기타 참고사항",

    # Page 4: narrative
    "narrative.incident_circumstances": "전자금융거래 사고 발생 경위 및 사유",
    "narrative.post_action": "사고 인지 후 조치 내역",

    # Page 5-6: required consent
    "consent.unique_id_collection_agreed": "고유식별정보 수집·이용 동의",
    "consent.personal_credit_collection_agreed": "개인(신용)정보 수집·이용 동의",
    "consent.unique_id_provision_agreed": "고유식별정보 제공 동의",
    "consent.personal_credit_provision_agreed": "개인(신용)정보 제공 동의",

    # Page 8: delegation
    "delegation.proxy_used": "본인 직접 신청 여부 또는 대리인 신청 여부",
    "delegation.agent_name": "대리인 성명",
    "delegation.agent_birth_date": "대리인 생년월일",
    "delegation.agent_phone_number": "대리인 전화번호",
    "delegation.agent_mobile_number": "대리인 휴대전화번호",
    "delegation.agent_email": "대리인 이메일",
    "delegation.agent_address": "대리인 주소",
    "delegation.agent_memo": "대리인 기타 메모",
    "delegation.request_purpose": "위임 신청취지/요구사항",
}


EVIDENCE_ITEM_LABELS: dict[str, str] = {
    "id_card_copy": "신분증 사본",
    "police_certificate": "사건사고사실확인원 및 수사기관 자료",
    "id_loss_evidence": "신분증 분실/도난 신고 및 재발급 증빙",
    "phone_evidence": "휴대전화 개통/명의도용/분실신고 관련 서류",
    "complaint_evidence": "수사기관 고소 또는 금융당국 민원 접수 내역",
    "investigation_delegation": "수사자료 열람·등사 위임장 또는 동의서",
    "data_leak_notice": "개인정보 유출 통지 내역",
    "delay_reason": "신청 지연 경위서",
    "family_proof": "가족·동거인 확인 서류",
    "signature_certificate": "본인서명사실확인서",
    "security_survey": "전자금융거래 사용 경험 및 보안 준수 조사 양식",
    "other_evidence": "기타 피해 확인 증빙자료",
    "passport_or_travel_proof": "출입국사실증명원 또는 여권 사본",
}


# ---------------------------------------------------------------------------
# Question groups: grouped for UX, but backed by complete field labels.
# ---------------------------------------------------------------------------

APPLICANT_BASIC_GROUP = QuestionGroup(
    question_id="applicant.basic",
    category=QuestionCategory.APPLICANT,
    candidate_paths=(
        "applicant.name",
        "applicant.birth_date",
        "applicant.mobile_number",
        "applicant.address",
    ),
    labels=FIELD_LABELS,
    intro="신청서 작성을 위해 기본정보를 확인하겠습니다.",
    input_type=QuestionInputType.FORM,
)

APPLICANT_ADDITIONAL_GROUP = QuestionGroup(
    question_id="applicant.additional",
    category=QuestionCategory.APPLICANT,
    candidate_paths=(
        "applicant.customer_number",
        "applicant.phone_number",
        "applicant.email",
        "applicant.memo",
        "applicant.sms_consent",
    ),
    labels=FIELD_LABELS,
    intro="신청인 추가 정보를 확인하겠습니다.",
    input_type=QuestionInputType.FORM,
)

APPLICANT_CORPORATE_GROUP = QuestionGroup(
    question_id="applicant.corporate",
    category=QuestionCategory.APPLICANT,
    candidate_paths=(
        "applicant.company_name",
        "applicant.business_number",
    ),
    labels=FIELD_LABELS,
    intro="법인 신청에 해당하는 경우 법인 정보를 확인해야 합니다. 개인 신청이면 해당 없다고 알려주세요.",
    input_type=QuestionInputType.FORM,
)

EXCLUSION_PRIMARY_GROUP = QuestionGroup(
    question_id="exclusion.primary_checklist",
    category=QuestionCategory.EXCLUSION,
    candidate_paths=tuple(f"exclusion.items.exclude_{i}" for i in range(1, 15)),
    labels=FIELD_LABELS,
    intro="피해 신고 제외대상 확인 항목입니다. 해당되는 항목만 체크해 주세요.",
    input_type=QuestionInputType.CHECKBOX,
)

EXCLUSION_SECONDARY_GROUP = QuestionGroup(
    question_id="exclusion.secondary_checklist",
    category=QuestionCategory.EXCLUSION,
    candidate_paths=(
        "exclusion.items.exclude_15",
        "exclusion.items.exclude_16",
        "exclusion.items.exclude_17",
        "exclusion.final_has_exclusion",
    ),
    labels=FIELD_LABELS,
    intro="추가 제외대상과 최종 제외대상 해당 여부를 확인해 주세요.",
    input_type=QuestionInputType.CHECKBOX,
)

INCIDENT_TIMELINE_GROUP = QuestionGroup(
    question_id="incident.timeline",
    category=QuestionCategory.INCIDENT,
    candidate_paths=(
        "incident.first_occurred_at",
        "incident.recognized_at",
        "incident.first_freeze_at",
    ),
    labels=FIELD_LABELS,
    intro="사고 경과를 시간순으로 정리해야 합니다.",
    input_type=QuestionInputType.FORM,
)

INCIDENT_SUMMARY_GROUP = QuestionGroup(
    question_id="incident.summary",
    category=QuestionCategory.INCIDENT,
    candidate_paths=(
        "incident.fraud_type",
        "incident.overview",
    ),
    labels=FIELD_LABELS,
    intro="신고서의 사고 정보란을 채우기 위해 사건 핵심 내용을 확인해야 합니다.",
    input_type=QuestionInputType.LLM_TEXT,
)

TRANSACTION_PRIMARY_GROUP = QuestionGroup(
    question_id="transaction.primary_details",
    category=QuestionCategory.TRANSACTION,
    candidate_paths=(
        "transactions.0.source_bank",
        "transactions.0.source_account_number",
        "transactions.0.amount_krw",
        "transactions.0.destination_bank",
        "transactions.0.destination_account_number",
        "transactions.0.destination_account_holder",
        "transactions.0.holder_type",
        "transactions.0.transaction_type",
        "transactions.0.transferred_at",
    ),
    labels=FIELD_LABELS,
    intro="피해 거래 정보를 정확히 특정해야 합니다.",
    input_type=QuestionInputType.TABLE,
)

OPTIONAL_REPORTS_GROUP = QuestionGroup(
    question_id="optional_reports.loss_and_identity",
    category=QuestionCategory.REPORT_AND_FREEZE,
    candidate_paths=(
        "optional_reports.id_loss_reported",
        "optional_reports.id_loss_reported_date",
        "optional_reports.phone_loss_reported",
        "optional_reports.phone_loss_reported_date",
        "optional_reports.identity_theft_phone_reported",
        "optional_reports.identity_theft_phone_reported_date",
        "optional_reports.other",
    ),
    labels=FIELD_LABELS,
    intro="신분증 분실, 휴대전화 분실, 명의도용 휴대전화 개설 신고 여부를 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

RELIEF_GROUP = QuestionGroup(
    question_id="relief.application",
    category=QuestionCategory.REPORT_AND_FREEZE,
    candidate_paths=(
        "relief.status",
        "relief.bank1",
        "relief.date1",
        "relief.bank2",
        "relief.date2",
        "relief.bank3",
        "relief.date3",
    ),
    labels=FIELD_LABELS,
    intro="피해구제 환급 신청 상태와 신청은행/신청일을 확인해야 합니다. 여러 은행에 신청한 경우 함께 알려주세요.",
    input_type=QuestionInputType.FORM,
)

INVESTIGATION_GROUP = QuestionGroup(
    question_id="investigation.status",
    category=QuestionCategory.REPORT_AND_FREEZE,
    candidate_paths=(
        "investigation.status",
        "investigation.agency",
        "investigation.reported_at",
    ),
    labels=FIELD_LABELS,
    intro="경찰 또는 수사기관 신고 상태를 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_TRANSFER_GROUP = QuestionGroup(
    question_id="survey.transfer_actor",
    category=QuestionCategory.SURVEY,
    candidate_paths=("survey.transfer_actor",),
    labels=FIELD_LABELS,
    intro="이번 사고에서 전자금융거래를 실제로 실행한 사람을 확인합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_APP_GROUP = QuestionGroup(
    question_id="survey.app_and_smishing",
    category=QuestionCategory.SURVEY,
    candidate_paths=(
        "survey.smishing_link_clicked",
        "survey.smishing_link_clicked_other_text",
        "survey.malicious_app_installed",
        "survey.malicious_app_installed_other_text",
    ),
    labels=FIELD_LABELS,
    intro="스미싱 링크 클릭과 악성앱/원격제어 앱 설치 여부를 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_PROVIDED_INFO_GROUP = QuestionGroup(
    question_id="survey.provided_information",
    category=QuestionCategory.SURVEY,
    candidate_paths=(
        "survey.provided_id_card",
        "survey.provided_personal_info",
        "survey.provided_device",
        "survey.provided_account_password",
        "survey.provided_security_media",
        "survey.provided_other_financial_info",
        "survey.provided_other_financial_info_text",
    ),
    labels=FIELD_LABELS,
    intro="상대방에게 제공한 개인정보와 전자금융 관련 정보를 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_BANKING_USAGE_GROUP = QuestionGroup(
    question_id="survey.banking_usage",
    category=QuestionCategory.SURVEY,
    candidate_paths=(
        "survey.internet_banking_used",
        "survey.internet_banking_frequency",
        "survey.phone_banking_used",
        "survey.phone_banking_frequency",
        "survey.open_banking_used",
        "survey.open_banking_frequency",
    ),
    labels=FIELD_LABELS,
    intro="평소 전자금융거래 사용 경험과 사용 빈도를 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_ID_PHONE_STORAGE_GROUP = QuestionGroup(
    question_id="survey.id_phone_storage",
    category=QuestionCategory.SURVEY,
    candidate_paths=(
        "survey.id_lent",
        "survey.id_copy_stored_digitally",
        "survey.id_physical_storage_method",
        "survey.phone_lent",
        "survey.phone_lock_method",
        "survey.security_media_storage_method",
    ),
    labels=FIELD_LABELS,
    intro="신분증, 휴대전화, 보안매체 보관 방식과 대여 경험을 확인해야 합니다.",
    input_type=QuestionInputType.FORM,
)

SURVEY_PRE_INCIDENT_GROUP = QuestionGroup(
    question_id="survey.pre_incident_security",
    category=QuestionCategory.SURVEY,
    candidate_paths=(
        "survey.id_loss_reported_before_incident",
        "survey.id_copy_stored_before_incident",
        "survey.id_copy_stored_before_incident_other_text",
        "survey.account_password_stored",
        "survey.account_password_stored_other_text",
        "survey.phone_lock_enabled",
        "survey.phone_lock_enabled_other_text",
        "survey.personal_info_leak_suspicion_details",
    ),
    labels=FIELD_LABELS,
    intro="사고 이전 보안 상태와 개인정보 유출 의심 정황을 확인합니다.",
    input_type=QuestionInputType.FORM,
)

NARRATIVE_GROUP = QuestionGroup(
    question_id="narrative.drafts",
    category=QuestionCategory.NARRATIVE,
    candidate_paths=(
        "narrative.incident_circumstances",
        "narrative.post_action",
    ),
    labels=FIELD_LABELS,
    intro="신고서에 들어갈 사고 경위와 사고 인지 후 조치 내역을 정리해야 합니다.",
    input_type=QuestionInputType.LLM_TEXT,
)

DELEGATION_PROXY_GROUP = QuestionGroup(
    question_id="delegation.proxy_used",
    category=QuestionCategory.DELEGATION,
    candidate_paths=("delegation.proxy_used",),
    labels=FIELD_LABELS,
    intro="신청 방식을 확인해야 합니다.",
    single_intro="본인이 직접 신청하시나요, 아니면 가족이나 대리인이 대신 신청하나요?",
    input_type=QuestionInputType.FORM,
)

DELEGATION_AGENT_GROUP = QuestionGroup(
    question_id="delegation.agent_details",
    category=QuestionCategory.DELEGATION,
    candidate_paths=(
        "delegation.agent_name",
        "delegation.agent_birth_date",
        "delegation.agent_phone_number",
        "delegation.agent_mobile_number",
        "delegation.agent_email",
        "delegation.agent_address",
        "delegation.agent_memo",
        "delegation.request_purpose",
    ),
    labels=FIELD_LABELS,
    intro="대리인이 신청하는 경우 위임장에 들어갈 대리인 정보를 확인해야 합니다. 본인이 직접 신청하면 해당 없음이라고 알려주세요.",
    input_type=QuestionInputType.FORM,
)

CONSENT_REQUIRED_GROUP = QuestionGroup(
    question_id="consent.required_bundle",
    category=QuestionCategory.CONSENT,
    candidate_paths=(
        "consent.unique_id_collection_agreed",
        "consent.personal_credit_collection_agreed",
        "consent.unique_id_provision_agreed",
        "consent.personal_credit_provision_agreed",
    ),
    labels=FIELD_LABELS,
    intro="전자금융거래 사고 피해 신고 접수와 조사를 위해 필수 개인정보 동의가 필요합니다.",
    input_type=QuestionInputType.CHECKBOX,
)


QUESTION_GROUPS: tuple[QuestionGroup, ...] = (
    APPLICANT_BASIC_GROUP,
    APPLICANT_ADDITIONAL_GROUP,
    APPLICANT_CORPORATE_GROUP,
    EXCLUSION_PRIMARY_GROUP,
    EXCLUSION_SECONDARY_GROUP,
    INCIDENT_TIMELINE_GROUP,
    INCIDENT_SUMMARY_GROUP,
    TRANSACTION_PRIMARY_GROUP,
    OPTIONAL_REPORTS_GROUP,
    RELIEF_GROUP,
    INVESTIGATION_GROUP,
    SURVEY_TRANSFER_GROUP,
    SURVEY_APP_GROUP,
    SURVEY_PROVIDED_INFO_GROUP,
    SURVEY_BANKING_USAGE_GROUP,
    SURVEY_ID_PHONE_STORAGE_GROUP,
    SURVEY_PRE_INCIDENT_GROUP,
    NARRATIVE_GROUP,
    DELEGATION_PROXY_GROUP,
    DELEGATION_AGENT_GROUP,
    # Consent intentionally comes last. It must be explicit and should not be inferred.
    CONSENT_REQUIRED_GROUP,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_next_questions(case: RecoveryCase, limit: int = 3) -> list[Question]:
    """Build the next user-facing questions from current RecoveryCase state.

    This function asks only for fields whose FieldValue.status is NOT_ASKED.
    It returns grouped questions so the user is not asked one tiny field at a time.
    """

    if limit <= 0:
        return []

    questions: list[Question] = []

    for group in QUESTION_GROUPS:
        if group.question_id == "delegation.agent_details" and not _should_ask_delegation_details(case):
            continue

        question = _question_from_group(case, group)
        if question is not None:
            questions.append(question)
            if len(questions) >= limit:
                return questions[:limit]

    evidence_question = _build_evidence_question(case)
    if evidence_question is not None and len(questions) < limit:
        questions.append(evidence_question)

    return questions[:limit]


def all_question_label_paths() -> set[str]:
    """Return every model FieldValue path known by questions.py."""

    return set(FIELD_LABELS)


def evidence_item_labels() -> dict[str, str]:
    """Return official evidence item labels used in the evidence prompt."""

    return dict(EVIDENCE_ITEM_LABELS)


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------


def _question_from_group(case: RecoveryCase, group: QuestionGroup) -> Question | None:
    missing_paths = [path for path in group.candidate_paths if _needs_question(case, path)]
    if not missing_paths:
        return None

    labels = [group.labels[path] for path in missing_paths]

    if group.single_intro and len(missing_paths) == 1:
        prompt = group.single_intro
    elif group.question_id == "consent.required_bundle":
        prompt = _build_consent_prompt(labels)
    else:
        prompt = _build_missing_fields_prompt(group.intro, labels)

    return Question(
        question_id=group.question_id,
        category=group.category,
        prompt=prompt,
        target_paths=missing_paths,
        required=group.required,
        input_type=group.input_type,
    )


def _build_evidence_question(case: RecoveryCase) -> Question | None:
    """Ask about all official evidence items if no captured evidence status exists yet."""

    if case.evidence and all(item.status != EvidenceStatus.NOT_ASKED for item in case.evidence):
        return None

    evidence_list = _join_korean_list(EVIDENCE_ITEM_LABELS.values())
    return Question(
        question_id="evidence.current_items",
        category=QuestionCategory.EVIDENCE,
        prompt=(
            "첨부서류란을 채우기 위해 현재 가지고 있는 증빙자료를 확인해야 합니다. "
            f"대상 서류는 {evidence_list}입니다. "
            "가지고 있는 자료, 없는 자료, 추후 제출 가능한 자료를 아는 범위에서 알려주세요."
        ),
        target_paths=["evidence"],
        required=True,
        input_type=QuestionInputType.EVIDENCE,
    )


def _should_ask_delegation_details(case: RecoveryCase) -> bool:
    proxy = case.delegation.proxy_used
    if proxy.status == FieldStatus.ANSWERED and proxy.value is False:
        return False
    return True


# ---------------------------------------------------------------------------
# Field access and prompt helpers
# ---------------------------------------------------------------------------


def _needs_question(case: RecoveryCase, path: str) -> bool:
    field_value = _get_field_value(case, path)
    return isinstance(field_value, FieldValue) and field_value.status == FieldStatus.NOT_ASKED


def _get_field_value(root: Any, path: str) -> FieldValue[Any] | None:
    current: Any = root
    for part in path.split("."):
        if isinstance(current, list):
            if not part.isdigit():
                return None
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue

        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return None
            continue

        if not hasattr(current, part):
            return None
        current = getattr(current, part)

    return current if isinstance(current, FieldValue) else None


def _build_missing_fields_prompt(intro: str, labels: list[str]) -> str:
    joined = _join_korean_list(labels)

    if len(labels) == 1:
        return f"{intro} {joined}을/를 알려주세요."
    if len(labels) <= 3:
        return f"{intro} {joined}만 추가로 알려주세요."
    return f"{intro} {joined}을/를 아는 범위에서 알려주세요."


def _build_consent_prompt(labels: list[str]) -> str:
    numbered = "\n".join(f"{index}. {label}" for index, label in enumerate(labels, start=1))
    return (
        "전자금융거래 사고 피해 신고 접수와 조사를 위해 아래 필수 개인정보 동의가 필요합니다.\n\n"
        f"{numbered}\n\n"
        "위 항목에 모두 동의하시나요? 일부만 동의하는 경우 번호별로 알려주세요."
    )


def _join_korean_list(labels: Iterable[str]) -> str:
    items = [label for label in labels if label]
    if not items:
        return "필요한 정보"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}와 {items[1]}"
    return ", ".join(items[:-1]) + f", {items[-1]}"


# ---------------------------------------------------------------------------
# Coverage helper used by tests.
# ---------------------------------------------------------------------------


def collect_field_value_paths(case: RecoveryCase) -> set[str]:
    """Collect all FieldValue paths from a RecoveryCase instance.

    This is intentionally public-ish for tests because label coverage is part of
    the V3 contract: every user-answerable model field should be questionable.
    """

    paths: set[str] = set()

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, FieldValue):
            paths.add(prefix.rstrip("."))
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}{index}.")
            return

        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}{key}.")
            return

        if is_dataclass(value):
            for field in fields(value):
                if field.name in {"case_id", "created_at"}:
                    continue
                walk(getattr(value, field.name), f"{prefix}{field.name}.")

    walk(case, "")
    return paths
