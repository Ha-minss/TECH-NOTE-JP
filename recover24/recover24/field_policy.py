"""Field-level UX and validation policy for Recover24 V3.

This module answers questions that models.py intentionally does not answer:
- Is a blank value acceptable for this field?
- What should a blank value mean?
- What placeholder/help text should be shown in the Streamlit UI?
- Which fields can safely receive a raw-text fallback if the LLM returns no patches?

It keeps the official case model clean while preventing the UI from treating every
HTML cell as a mandatory customer question.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import FieldStatus


@dataclass(frozen=True, slots=True)
class QuestionGuidance:
    help: str
    example: str
    empty_policy: str = "모르면 ‘미확인’, 해당 없으면 ‘해당없음’을 선택하거나 비워두세요."


# Fields where an empty structured-form submission should mark the field as
# not-applicable instead of leaving it NOT_ASKED forever.
BLANK_AS_NOT_APPLICABLE_PATHS: set[str] = {
    # Applicant optional fields
    "applicant.customer_number",
    "applicant.company_name",
    "applicant.business_number",
    "applicant.phone_number",
    "applicant.email",
    "applicant.memo",
    "applicant.sms_consent",

    # Optional report details
    "optional_reports.id_loss_reported_date",
    "optional_reports.phone_loss_reported_date",
    "optional_reports.identity_theft_phone_reported_date",
    "optional_reports.other",

    # Multi-bank relief rows. Row 1 date can also be unknown/not-yet-fixed when status is planned.
    "relief.bank2",
    "relief.date2",
    "relief.bank3",
    "relief.date3",

    # Survey conditional/detail text
    "survey.smishing_link_clicked_other_text",
    "survey.malicious_app_installed_other_text",
    "survey.provided_other_financial_info_text",
    "survey.internet_banking_frequency",
    "survey.phone_banking_frequency",
    "survey.open_banking_frequency",
    "survey.id_physical_storage_method",
    "survey.phone_lock_method",
    "survey.security_media_storage_method",
    "survey.id_copy_stored_before_incident_other_text",
    "survey.account_password_stored_other_text",
    "survey.phone_lock_enabled_other_text",
    "survey.personal_info_leak_suspicion_details",

    # Delegation details are optional unless proxy_used=True; when displayed and blank, don't repeat forever.
    "delegation.agent_name",
    "delegation.agent_birth_date",
    "delegation.agent_phone_number",
    "delegation.agent_mobile_number",
    "delegation.agent_email",
    "delegation.agent_address",
    "delegation.agent_memo",
    "delegation.request_purpose",
}

# Blank means the user does not know / date is not fixed yet, but the question
# should not block the flow repeatedly.
BLANK_AS_UNKNOWN_PATHS: set[str] = {
    "relief.date1",
    "investigation.reported_at",
}

TEXT_FALLBACK_PATHS: set[str] = {
    "incident.overview",
    "narrative.incident_circumstances",
    "narrative.post_action",
    "applicant.memo",
    "optional_reports.other",
    "survey.personal_info_leak_suspicion_details",
    "delegation.request_purpose",
}

UNKNOWN_ON_LLM_FAILURE_PREFIXES: tuple[str, ...] = (
    "incident.",
    "transactions.",
    "relief.",
    "investigation.",
    "survey.",
)

DATE_LIKE_PATHS: set[str] = {
    "applicant.birth_date",
    "incident.first_occurred_at",
    "incident.recognized_at",
    "incident.first_freeze_at",
    "transactions.0.transferred_at",
    "optional_reports.id_loss_reported_date",
    "optional_reports.phone_loss_reported_date",
    "optional_reports.identity_theft_phone_reported_date",
    "relief.date1",
    "relief.date2",
    "relief.date3",
    "investigation.reported_at",
    "delegation.agent_birth_date",
}

PLACEHOLDERS: dict[str, str] = {
    'applicant.name': '예: 김민수',
    'applicant.birth_date': '예: 1994년 4월 2일',
    'applicant.customer_number': '예: 고객번호를 모르면 비워두세요',
    'applicant.company_name': '예: 개인 신청이면 비워두세요',
    'applicant.business_number': '예: 개인 신청이면 비워두세요',
    'applicant.phone_number': '예: 02-1234-5678',
    'applicant.mobile_number': '예: 010-1234-5678',
    'applicant.email': '예: minsu.kim@example.com',
    'applicant.address': '예: 서울특별시 동대문구 신설동 33가길 12',
    'applicant.memo': '예: 연락 가능 시간은 평일 오후입니다',
    'incident.first_occurred_at': '예: 2026년 6월 14일 오후 6시 11분경',
    'incident.recognized_at': '예: 2026년 6월 14일 오후 7시 33분경',
    'incident.first_freeze_at': '예: 2026년 6월 14일 오후 8시 42분경',
    'incident.overview': '예: 검찰청 직원을 사칭한 전화에 속아 1,500만원을 송금한 피해',
    'transactions.0.source_bank': '예: 우리은행',
    'transactions.0.source_account_number': '예: 1002-123-456789',
    'transactions.0.destination_bank': '예: 농협은행',
    'transactions.0.destination_account_number': '예: 356-1234-5678-90',
    'transactions.0.destination_account_holder': '예: 김철수',
    'transactions.0.holder_type': '예: 타인',
    'transactions.0.transferred_at': '예: 2026년 6월 14일 오후 6시 20분경',
    'optional_reports.id_loss_reported_date': '예: 2026년 6월 14일',
    'optional_reports.phone_loss_reported_date': '예: 2026년 6월 14일',
    'optional_reports.identity_theft_phone_reported_date': '예: 2026년 6월 15일',
    'optional_reports.other': '예: 명의도용 방지서비스 가입 예정',
    'relief.bank1': '예: 우리은행',
    'relief.date1': '예: 2026년 6월 15일',
    'relief.bank2': '예: 농협은행',
    'relief.date2': '예: 2026년 6월 15일',
    'relief.bank3': '예: 추가 신청은행이 있을 때만 입력',
    'relief.date3': '예: 추가 신청일이 있을 때만 입력',
    'investigation.agency': '예: 서울중부경찰서',
    'investigation.reported_at': '예: 2026년 6월 14일',
    'survey.transfer_actor': '예: 손님 본인',
    'survey.internet_banking_frequency': '예: 주 1회 정도',
    'survey.phone_banking_frequency': '예: 거의 사용하지 않음',
    'survey.open_banking_frequency': '예: 월 1~2회 정도',
    'survey.id_physical_storage_method': '예: 지갑에 보관',
    'survey.phone_lock_method': '예: 비밀번호와 지문인식 사용',
    'survey.security_media_storage_method': '예: 자택 서랍에 보관',
    'survey.personal_info_leak_suspicion_details': '예: 신분증 사진과 인증번호를 상대방에게 보냈습니다',
    'narrative.incident_circumstances': '예: 검찰청 직원을 사칭한 사람에게 전화를 받고 송금했습니다',
    'narrative.post_action': '예: 은행 고객센터에 지급정지를 요청하고 경찰서에 신고했습니다',
    'delegation.agent_name': '예: 이수진',
    'delegation.agent_birth_date': '예: 1975년 8월 10일',
    'delegation.agent_phone_number': '예: 02-2345-6789',
    'delegation.agent_mobile_number': '예: 010-9876-5432',
    'delegation.agent_email': '예: agent@example.com',
    'delegation.agent_address': '예: 서울특별시 중구 세종대로 110',
    'delegation.agent_memo': '예: 가족 대리 신청',
    'delegation.request_purpose': '예: 전자금융거래 사고 피해 신고 및 관련 서류 제출 위임',
}

QUESTION_GUIDANCE: dict[str, QuestionGuidance] = {
    'applicant.basic': QuestionGuidance(
        help='주민등록상 정보와 연락 가능한 번호를 기준으로 작성해 주세요. 이름, 생년월일, 휴대전화번호, 주소는 본인 확인에 필요한 기본 정보입니다.',
        example='성명: 김민수\n생년월일: 1994년 4월 2일\n휴대전화번호: 010-1234-5678\n주소: 서울특별시 동대문구 신설동 33가길 12',
        empty_policy='성명, 생년월일, 휴대전화번호, 주소는 가능한 정확히 입력해 주세요. 정확하지 않은 부분은 담당자 검토가 필요합니다.',
    ),
    'applicant.additional': QuestionGuidance(
        help='연락·식별을 돕는 추가 정보입니다. 일반전화, 고객번호, 이메일, 기타 메모는 없으면 비워도 됩니다.',
        example='고객번호: 미확인\n일반전화: 비워둠\n이메일: minsu.kim@example.com\n기타 메모: 평일 오후 통화 가능\nSMS 수신 동의: 동의함',
        empty_policy='일반전화나 고객번호가 없으면 비워두세요. 필수 연락처는 휴대전화번호를 기준으로 확인합니다.',
    ),
    'applicant.corporate': QuestionGuidance(
        help='법인 명의 신청인 경우에만 작성합니다. 개인 피해자는 법인명과 사업자등록번호를 비워두면 됩니다.',
        example='개인 신청: 법인명 비워둠, 사업자등록번호 비워둠\n법인 신청: 주식회사 예시, 123-45-67890',
        empty_policy='개인 피해자라면 비워두면 해당없음으로 처리됩니다.',
    ),
    'exclusion.primary_checklist': QuestionGuidance(
        help='아래 항목 중 이번 사고에 해당되는 내용이 있으면 체크해 주세요. 해당되는 항목이 없으면 ‘모든 항목 해당없음’을 선택해 주세요.',
        example='예시: 보이스피싱범의 거짓말에 속아 송금했고, 가족·지인 거래나 상거래 분쟁이 아니라면 개별 제외대상은 체크하지 않고 ‘모든 항목 해당없음’을 선택합니다.',
        empty_policy='확실하지 않은 항목은 담당자 검토가 필요합니다. 해당되는 항목이 없다는 점이 분명하면 ‘모든 항목 해당없음’을 선택하세요.',
    ),
    'exclusion.secondary_checklist': QuestionGuidance(
        help='추가 제외대상과 최종 제외대상 해당 여부를 확인합니다. 소송, 합의, 수사기관 판단 등 이미 진행된 절차가 있는지 확인해 주세요.',
        example='소송이나 합의가 없고, 수사기관에서 전자금융사고가 아니라고 판단한 사실도 없다면 해당 항목은 체크하지 않습니다.',
        empty_policy='확실하지 않으면 미확인으로 두고 담당자에게 확인하세요.',
    ),
    'incident.timeline': QuestionGuidance(
        help='사고가 발생한 시점, 피해 사실을 알게 된 시점, 은행에 지급정지나 거래제한을 요청한 시점을 확인합니다. 정확한 시간이 기억나지 않으면 ‘경’ 또는 ‘쯤’으로 작성해도 됩니다.',
        example='최초 사고발생: 2026년 6월 14일 오후 6시 11분경\n최초 사고인지: 2026년 6월 14일 오후 7시 33분경\n지급정지 요청: 2026년 6월 14일 오후 8시 42분경',
        empty_policy='정확한 시간을 모르면 ‘미확인’으로 입력하세요. 아직 지급정지를 요청하지 않았다면 ‘아직 요청 전’으로 남겨도 됩니다.',
    ),
    'incident.summary': QuestionGuidance(
        help='피해유형과 사건개요를 정리합니다. 누가 사칭했는지, 어떤 이유로 송금하게 되었는지, 얼마를 송금했는지 중심으로 적어 주세요.',
        example='검찰청 직원을 사칭한 사람에게 전화를 받고, 제 계좌가 범죄에 연루되었다는 말을 들었습니다. 상대방의 지시에 따라 우리은행 계좌에서 농협은행 계좌로 1,500만원을 송금했습니다.',
        empty_policy='사건개요가 아직 정리되지 않았으면 아는 범위만 작성해도 됩니다.',
    ),
    'transaction.primary_details': QuestionGuidance(
        help='이체확인증, 거래내역 화면, 문자 알림 등을 기준으로 피해 거래를 특정해 주세요. 모르는 계좌번호나 예금주는 추측하지 말고 ‘미확인’으로 남겨 주세요.',
        example='출금은행: 우리은행\n출금계좌번호: 1002-123-456789\n피해금액: 1,500만원\n입금은행: 농협은행\n입금계좌번호: 356-1234-5678-90\n예금주: 김철수\n본인/타인: 타인\n거래유형: 모바일뱅킹\n송금시각: 2026년 6월 14일 오후 6시 20분경',
        empty_policy='피해금액은 가능한 정확히 입력해 주세요. 계좌번호나 예금주를 모르면 미확인으로 남겨도 됩니다.',
    ),
    'optional_reports.loss_and_identity': QuestionGuidance(
        help='신분증 분실, 휴대전화 분실, 명의도용 휴대전화 개설 신고 여부를 확인합니다. ‘예’를 선택한 항목만 신고일을 입력합니다.',
        example='신분증 분실 신고: 아니오\n휴대전화 분실 신고: 아니오\n명의도용 휴대전화 개설 신고: 미확인\n기타: 명의도용 방지서비스 가입 예정',
        empty_policy='해당 없는 항목은 아니오 또는 해당없음으로 선택하세요. 기타 사항은 없으면 비워두세요.',
    ),
    'relief.application': QuestionGuidance(
        help='피해구제 환급 신청 상태를 확인합니다. 여러 은행에 신청한 경우에만 2번, 3번 행을 입력합니다.',
        example='신청상태: 신청예정\n신청은행 1: 우리은행\n신청일 1: 2026년 6월 15일\n신청은행 2: 농협은행\n신청일 2: 2026년 6월 15일',
        empty_policy='신청일이 아직 정해지지 않았으면 비워두세요. 추가 은행이 없으면 2번, 3번 행은 비워두세요.',
    ),
    'investigation.status': QuestionGuidance(
        help='경찰 또는 수사기관에 신고했는지 확인합니다. 신고한 경우 수사기관명과 신고일을 입력해 주세요.',
        example='신고상태: 신고\n수사기관명: 서울중부경찰서\n신고일: 2026년 6월 14일',
        empty_policy='아직 신고하지 않았다면 미신고로 선택하고 기관명과 신고일은 비워두세요.',
    ),
    'survey.transfer_actor': QuestionGuidance(
        help='이번 사고에서 전자금융거래를 실제로 실행한 사람이 누구인지 확인합니다.',
        example='예시: 신청인 본인이 모바일뱅킹으로 송금했다면 ‘손님 본인’을 선택합니다. 가족이 대신 이체했다면 ‘가족’을 선택합니다.',
        empty_policy='기억이 불명확하면 미확인으로 선택하세요.',
    ),
    'survey.app_and_smishing': QuestionGuidance(
        help='문자·카카오톡 링크 클릭, 악성앱 또는 원격제어앱 설치 여부를 확인합니다.',
        example='스미싱 링크 클릭: 아니오\n악성앱 설치: 아니오\n기타 상세: 상대방 안내로 앱을 설치한 기억은 없으나 추가 확인 필요',
        empty_policy='설치 여부가 불확실하면 미확인으로 선택하세요. 기타 상세는 필요한 경우에만 입력합니다.',
    ),
    'survey.provided_information': QuestionGuidance(
        help='상대방에게 제공했거나 노출된 개인정보·전자금융 관련 정보를 확인합니다. 제공하지 않은 항목은 체크하지 않습니다.',
        example='신분증: 체크\n개인정보: 체크\n계좌번호/비밀번호: 미체크\nOTP/보안카드/인증서: 미체크\n기타 전자금융 관련정보: 인증번호 6자리를 알려줌',
        empty_policy='제공한 정보가 없으면 ‘제공하거나 노출한 정보 없음’을 선택하세요.',
    ),
    'survey.banking_usage': QuestionGuidance(
        help='평소 인터넷뱅킹, 폰뱅킹, 오픈뱅킹 사용 여부와 사용 빈도를 확인합니다.',
        example='인터넷뱅킹: 사용함, 주 1회 정도\n폰뱅킹: 사용 안 함\n오픈뱅킹: 사용함, 월 1~2회 정도',
        empty_policy='사용하지 않는 채널은 ‘사용 안 함’을 선택하세요. 기억이 불명확하면 미확인으로 선택합니다.',
    ),
    'survey.id_phone_storage': QuestionGuidance(
        help='신분증, 휴대전화, 보안카드 등 접근매체를 평소 어떻게 보관했는지 확인합니다.',
        example='신분증 대여 경험: 아니오\n신분증 사본 저장: 예\n신분증 실물 보관: 지갑에 보관\n휴대전화 대여 경험: 아니오\n휴대전화 잠금 방식: 비밀번호와 지문인식 사용\n보안카드 보관 방식: 자택 서랍에 보관',
        empty_policy='정확히 기억나지 않으면 미확인으로 선택하거나 간단히 적어 주세요.',
    ),
    'survey.pre_incident_security': QuestionGuidance(
        help='사고 이전의 신분증 분실신고, 신분증 사본 저장, 계좌번호·비밀번호 저장, 휴대전화 잠금 설정 여부와 개인정보 유출 의심 정황을 확인합니다.',
        example='사고 이전 신분증 분실신고: 해당없음\n신분증 사본 저장: 예\n계좌번호/비밀번호 저장: 아니오\n휴대전화 잠금 설정: 예\n개인정보 유출 의심 정황: 신분증 사진과 인증번호를 상대방에게 보냈습니다.',
        empty_policy='특이사항이 없으면 ‘특이사항 없음’이라고 입력하거나 비워둘 수 있습니다.',
    ),
    'narrative.drafts': QuestionGuidance(
        help='사고 발생 경위와 사고 후 조치 내역을 문장으로 작성해 주세요. 전화, 문자, 카카오톡 등 최초 접촉 경위와 송금 이유, 송금 후 조치를 순서대로 적으면 됩니다.',
        example='2026년 6월 14일 오전 10시경 검찰청 직원을 사칭한 사람에게 전화를 받았습니다. 상대방은 제 계좌가 범죄에 연루되었다고 하며 안전한 계좌로 돈을 옮겨야 한다고 말했습니다. 저는 안내에 따라 같은 날 우리은행 계좌에서 농협은행 계좌로 1,500만원을 송금했습니다. 송금 후 가족과 통화하는 과정에서 사기 가능성을 알게 되었고, 같은 날 오후 8시 40분경 우리은행 고객센터에 지급정지를 요청한 뒤 서울중부경찰서에 피해 사실을 신고했습니다.',
        empty_policy='정확한 시간이 기억나지 않으면 ‘오전’, ‘오후’, ‘경’처럼 대략적으로 작성해도 됩니다.',
    ),
    'consent.required_bundle': QuestionGuidance(
        help='피해 신고 접수와 조사를 위해 필요한 개인정보 수집·이용 및 제공 동의 항목입니다.',
        example='필수 동의 항목 전체에 동의할 수 있으면 ‘필수 동의 항목 전체 동의’를 선택합니다. 일부 동의하지 않는 항목이 있으면 항목별로 선택합니다.',
        empty_policy='동의하지 않은 필수 항목이 있으면 공식 접수 단계에서 진행이 제한될 수 있습니다.',
    ),
    'delegation.proxy_used': QuestionGuidance(
        help='본인이 직접 신청하는지, 가족 등 대리인이 신청하는지 확인합니다.',
        example='본인이 직접 신청하면 ‘본인 직접 신청’을 선택합니다. 가족이 대신 방문해 신청하는 경우에는 ‘대리인 신청’을 선택합니다.',
        empty_policy='대리인이 없으면 본인 직접 신청으로 처리합니다.',
    ),
    'delegation.agent_details': QuestionGuidance(
        help='대리인이 신청하는 경우에만 입력합니다. 본인이 직접 신청하는 경우 이 항목은 비워두면 됩니다.',
        example='대리인 성명: 이수진\n생년월일: 1975년 8월 10일\n휴대전화번호: 010-9876-5432\n주소: 서울특별시 중구 세종대로 110\n위임 목적: 전자금융거래 사고 피해 신고 및 관련 서류 제출 위임',
        empty_policy='본인 직접 신청이면 비워두면 해당없음으로 처리됩니다.',
    ),
    'evidence.current_items': QuestionGuidance(
        help='현재 제출할 수 있는 증빙자료 상태를 확인합니다. 실제 보유한 자료만 ‘보유’로 선택해 주세요.',
        example='신분증 사본: 보유\n이체확인증 또는 거래내역: 보유\n사건사고사실확인원: 추후 제출\n통화내역 또는 문자 캡처: 보유\n출입국사실증명원: 해당없음',
        empty_policy='아직 확인하지 못한 증빙은 모름으로 두세요. 실제로 없는 자료를 보유로 선택하지 마세요.',
    ),
}


def blank_status_for_path(path: str) -> FieldStatus | None:
    if path in BLANK_AS_NOT_APPLICABLE_PATHS:
        return FieldStatus.NOT_APPLICABLE
    if path in BLANK_AS_UNKNOWN_PATHS:
        return FieldStatus.UNKNOWN
    return None


def placeholder_for_path(path: str) -> str:
    if path in PLACEHOLDERS:
        return PLACEHOLDERS[path]
    if path in DATE_LIKE_PATHS:
        return "예: 2026년 6월 14일 오후 6시 11분경"
    if path.endswith("memo") or path.endswith("other") or path.endswith("details"):
        return "예: 특이사항 없음"
    return "예: 미확인"


def guidance_for_question(question_id: str) -> QuestionGuidance | None:
    return QUESTION_GUIDANCE.get(question_id)


def text_fallback_path(paths: list[str]) -> str | None:
    for path in paths:
        if path in TEXT_FALLBACK_PATHS:
            return path
    return None


def should_mark_unknown_on_llm_failure(path: str) -> bool:
    if path in TEXT_FALLBACK_PATHS:
        return False
    return path.startswith(UNKNOWN_ON_LLM_FAILURE_PREFIXES)
