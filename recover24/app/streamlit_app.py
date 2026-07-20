"""Streamlit UI for the Recover24 V3 mixed-input workflow.

The important V3 change in this version:
- Structured official-form fields are collected with form/radio/selectbox widgets.
- Only free Korean narrative/ambiguous meaning is sent to answers.py/LLM.
- Both paths still produce Patch[] and enter patching.py, so the core pipeline is intact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow `streamlit run app/streamlit_app.py` from the project root or app dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from application.run_recover24_case import (
    DEFAULT_TEMPLATE_DIR,
    DemoKeywordProvider,
    Recover24CaseResult,
    answer_question,
    apply_form_patches,
    render_case_html,
    start_case,
)
from recover24.field_policy import guidance_for_question, placeholder_for_path
from recover24.form_patches import (
    build_boolean_choice_patches,
    build_evidence_patches,
    build_form_patches,
)
from recover24.patching import apply_patches
from recover24.document_view import build_document_view
from recover24.readiness import evaluate_readiness
from recover24.questions import build_next_questions
from recover24.models import (
    EvidenceStatus,
    FieldStatus,
    FraudType,
    Patch,
    Question,
    QuestionCategory,
    QuestionInputType,
    ReportStatus,
    TransactionType,
)
from recover24.providers.base import LLMProvider
from recover24.providers.gemma_colab import GemmaColabProvider
from recover24.questions import FIELD_LABELS, evidence_item_labels
from recover24.readiness import IssueSeverity, ReadinessStatus


st.set_page_config(page_title="Recover24 V3", page_icon="🛡️", layout="wide")


st.markdown("""
<style>
.r24-help {
  background: transparent;
  border: 0;
  padding: 0;
  margin: 0.35rem 0 0.75rem 0;
  font-size: 0.98rem;
  line-height: 1.65;
  color: #f9fafb;
}
.r24-small-muted {
  color: #d1d5db;
  font-size: 0.9rem;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
  color: rgba(229, 231, 235, 0.72) !important;
  opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


PROVIDER_DEMO = "Demo"
PROVIDER_GEMMA = "Gemma Colab"

YES_NO_UNKNOWN = ["미확인", "아니오", "예"]
YES_NO_NA_UNKNOWN = ["미확인", "아니오", "예", "해당없음"]
BOOLEAN_CHOICE_TO_VALUE = {
    "아니오": False,
    "예": True,
    "모름": "__unknown__",
    "미확인": "__unknown__",
    "해당없음": "__not_applicable__",
}

REPORT_STATUS_OPTIONS = {
    "미확인": "__unknown__",
    "신고/신청 안 함": ReportStatus.NOT_REPORTED,
    "예정": ReportStatus.PLANNED,
    "신고/신청 완료": ReportStatus.REPORTED,
    "진행 중": ReportStatus.IN_PROGRESS,
    "완료": ReportStatus.COMPLETED,
    "종결": ReportStatus.CLOSED,
    "기타": ReportStatus.OTHER,
    "모름": "__unknown__",
}

FRAUD_TYPE_OPTIONS = {
    "미확인": "__unknown__",
    "수사기관/금감원 사칭": FraudType.AUTHORITY_IMPERSONATION,
    "가족/지인 사칭": FraudType.FAMILY_IMPERSONATION,
    "대출빙자": FraudType.LOAN_SCAM,
    "스미싱/악성앱/원격제어": FraudType.SMISHING_MALWARE,
    "은행/카드사/택배 등 기관 사칭": FraudType.INSTITUTION_IMPERSONATION,
    "기타": FraudType.OTHER,
    "모름": "__unknown__",
}

TRANSACTION_TYPE_OPTIONS = {
    "미확인": TransactionType.UNKNOWN,
    "모바일뱅킹 이체": TransactionType.MOBILE_BANKING_TRANSFER,
    "인터넷뱅킹 이체": TransactionType.INTERNET_BANKING_TRANSFER,
    "폰뱅킹 이체": TransactionType.PHONE_BANKING_TRANSFER,
    "ATM 이체": TransactionType.ATM_TRANSFER,
    "카드/대출 관련": TransactionType.CARD_OR_LOAN,
    "모름": TransactionType.UNKNOWN,
}

EVIDENCE_STATUS_OPTIONS = {
    "모름": EvidenceStatus.UNKNOWN,
    "보유": EvidenceStatus.AVAILABLE,
    "없음": EvidenceStatus.MISSING,
    "추후 제출": EvidenceStatus.PLANNED,
    "해당 없음": EvidenceStatus.NOT_APPLICABLE,
    "모름": EvidenceStatus.UNKNOWN,
}


def initialize_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("question_limit", 3)
    st.session_state.setdefault("provider_name", PROVIDER_DEMO)
    st.session_state.setdefault("gemma_url", os.getenv("RECOVER24_GEMMA_COLAB_URL", ""))
    st.session_state.setdefault("ui_mode", "chat")
    st.session_state.setdefault("developer_mode", False)


def main() -> None:
    initialize_state()

    st.title("Recover24 V3 — 보이스피싱 피해 신고서 작성 Agent")
    st.caption("피해자와 대화하듯 필요한 정보를 확인하고, 작성 중인 신고서를 즉시 미리볼 수 있습니다.")

    provider, strict = render_sidebar()

    result: Recover24CaseResult | None = st.session_state.result
    if result is None:
        render_initial_input(provider, strict)
    else:
        render_case_workspace(result, provider, strict)


def render_sidebar() -> tuple[LLMProvider, bool]:
    with st.sidebar:
        st.header("설정")
        st.session_state.question_limit = st.slider("한 번에 보여줄 질문 수", 1, 5, st.session_state.question_limit)
        strict = st.checkbox("HTML 전체 항목 확인 모드", value=True)
        st.session_state.developer_mode = st.checkbox("개발자 디버그 정보 보기", value=st.session_state.developer_mode)

        st.divider()
        st.subheader("AI Provider")
        st.session_state.provider_name = st.radio(
            "자연어 해석에 사용할 provider",
            [PROVIDER_DEMO, PROVIDER_GEMMA],
            index=0 if st.session_state.provider_name == PROVIDER_DEMO else 1,
        )
        if st.session_state.provider_name == PROVIDER_GEMMA:
            st.session_state.gemma_url = st.text_input(
                "Gemma Colab URL",
                value=st.session_state.gemma_url,
                placeholder="https://xxxxx.trycloudflare.com",
            )
            st.caption("정형 form 질문은 Gemma를 호출하지 않습니다. 사건 경위/자연어 질문에만 사용합니다.")
        else:
            st.caption("Demo provider는 외부 LLM 없이 흐름을 확인하기 위한 결정적 provider입니다.")

        if st.button("새 케이스 시작", use_container_width=True):
            st.session_state.result = None
            st.session_state.ui_mode = "chat"
            st.rerun()

    provider = build_provider(st.session_state.provider_name, st.session_state.gemma_url)
    return provider, strict


def build_provider(provider_name: str, gemma_url: str) -> LLMProvider:
    if provider_name == PROVIDER_GEMMA:
        if not gemma_url.strip():
            st.sidebar.warning("Gemma URL이 비어 있어 Demo provider를 사용합니다.")
            return DemoKeywordProvider()
        return GemmaColabProvider(base_url=gemma_url.strip(), timeout_seconds=60)
    return DemoKeywordProvider()


def render_initial_input(provider: LLMProvider, strict: bool) -> None:
    st.subheader("1. 최초 피해 진술")
    sample = (
        "검찰청 직원을 사칭한 사람에게 전화를 받고, 제 계좌가 범죄에 연루되었다는 말을 들었습니다. "
        "상대방의 지시에 따라 우리은행 계좌에서 농협은행 계좌로 1,500만원을 송금했고, "
        "송금 후 가족에게 확인하다가 사기임을 알게 되어 은행에 지급정지를 요청하려고 합니다."
    )
    initial_text = st.text_area("피해자가 처음 말한 내용을 입력하세요", value=sample, height=180)

    if st.button("사건 생성하기", type="primary"):
        if not initial_text.strip():
            st.warning("최초 진술을 입력해주세요.")
            return
        try:
            result = start_case(
                initial_text,
                provider,
                question_limit=st.session_state.question_limit,
                strict_document_completion=strict,
            )
        except Exception as exc:  # pragma: no cover - UI display
            st.error(f"사건 생성 실패: {exc}")
            return
        st.session_state.result = result
        st.rerun()



def render_case_workspace(result: Recover24CaseResult, provider: LLMProvider, strict: bool) -> None:
    """Render either conversational intake or the official-form preview.

    The demo should feel like a guided 상담 flow, not an internal field editor.
    The official HTML is available on demand as a separate screen.
    """

    if st.session_state.ui_mode == "preview":
        render_preview_workspace(result)
        return

    render_chat_workspace(result, provider, strict)



def render_chat_workspace(result: Recover24CaseResult, provider: LLMProvider, strict: bool) -> None:
    top_cols = st.columns([1.2, 1])
    with top_cols[0]:
        st.subheader("신고서 작성 상담")
        st.caption("모르는 항목은 미확인, 해당 없는 항목은 해당없음으로 처리할 수 있습니다.")
    with top_cols[1]:
        if st.button("작성중인 전자금융거래 사고 피해 신고서 보기", use_container_width=True):
            st.session_state.ui_mode = "preview"
            st.rerun()

    render_compact_progress(result)
    st.divider()
    render_questions(result, provider, strict)

    if st.session_state.developer_mode:
        st.divider()
        render_debug_panels(result)


def render_preview_workspace(result: Recover24CaseResult) -> None:
    top_cols = st.columns([1, 1, 1])
    with top_cols[0]:
        if st.button("← 작성 계속하기", use_container_width=True):
            st.session_state.ui_mode = "chat"
            st.rerun()
    with top_cols[1]:
        st.metric("문서 항목 확인률", f"{result.readiness.document_completion_rate:.1%}")
    with top_cols[2]:
        status_text = {
            ReadinessStatus.NOT_READY: "추가 입력 필요",
            ReadinessStatus.NEEDS_REVIEW: "직원 검토 필요",
            ReadinessStatus.READY: "은행 검토 가능",
        }[result.readiness.status]
        st.metric("상태", status_text)

    st.subheader("작성중인 전자금융거래 사고 피해 신고서")
    render_html_preview(result, height=900)

    if st.session_state.developer_mode:
        st.divider()
        render_debug_panels(result)


def render_compact_progress(result: Recover24CaseResult) -> None:
    report = result.readiness
    st.progress(report.document_completion_rate, text=f"작성 진행률 {report.document_completion_rate:.1%}")

    if report.status == ReadinessStatus.READY:
        st.success("현재 작성 내용은 은행 검토가 가능한 수준입니다.")
    elif report.status == ReadinessStatus.NEEDS_REVIEW:
        st.warning("일부 항목은 담당자 검토가 필요합니다.")
    else:
        st.info("아래 질문에 답하면 신고서가 계속 채워집니다.")

def render_readiness(result: Recover24CaseResult) -> None:
    report = result.readiness
    status_label = {
        ReadinessStatus.NOT_READY: "NOT_READY · 사용자 추가 입력 필요",
        ReadinessStatus.NEEDS_REVIEW: "NEEDS_REVIEW · 직원 판단 필요",
        ReadinessStatus.READY: "READY · 은행 검토 가능",
    }[report.status]

    if report.status == ReadinessStatus.READY:
        st.success(status_label)
    elif report.status == ReadinessStatus.NEEDS_REVIEW:
        st.warning(status_label)
    else:
        st.error(status_label)

    st.progress(report.document_completion_rate, text=f"문서 항목 확인률 {report.document_completion_rate:.1%}")

    metric_cols = st.columns(3)
    metric_cols[0].metric("확인 필드", f"{report.answered_field_count}/{report.total_field_count}")
    metric_cols[1].metric("사용자 조치", "필요" if report.requires_user_action else "없음")
    metric_cols[2].metric("직원 판단", "필요" if report.requires_staff_decision else "없음")

    if report.issues:
        with st.expander("readiness 이슈 보기", expanded=True):
            for issue in report.issues[:10]:
                icon = "⛔" if issue.severity == IssueSeverity.BLOCKER else "🟡" if issue.severity == IssueSeverity.REVIEW else "ℹ️"
                st.markdown(f"{icon} **{issue.code}** · `{issue.resolution_owner.value}`")
                st.write(issue.message)
                if issue.labels:
                    st.caption(" / ".join(issue.labels[:8]))



def render_questions(result: Recover24CaseResult, provider: LLMProvider, strict: bool) -> None:
    if not result.questions:
        st.success("현재 추가로 확인할 질문이 없습니다. 작성중인 신고서를 확인해 주세요.")
        if st.button("작성중인 신고서 보기", use_container_width=True):
            st.session_state.ui_mode = "preview"
            st.rerun()
        return

    question = result.questions[0]

    st.markdown("### 다음 확인 항목")
    st.markdown(f"#### {_display_question_title(question)}")
    st.caption(_display_question_caption(question))
    render_question_guidance(question)

    if question.input_type == QuestionInputType.EVIDENCE or question.target_paths == ["evidence"]:
        render_evidence_question(result, question, strict)
    elif question.input_type == QuestionInputType.CHECKBOX:
        render_checkbox_question(result, question, strict)
    elif question.input_type == QuestionInputType.TABLE:
        render_form_question(result, question, strict, title="거래 정보 입력")
    elif question.input_type == QuestionInputType.FORM:
        render_form_question(result, question, strict, title="공식 정보 입력")
    else:
        render_llm_question(result, question, provider, strict)

    if len(result.questions) > 1:
        with st.expander("이후 확인 예정 항목"):
            for idx, q in enumerate(result.questions[1:], start=2):
                st.markdown(f"{idx}. **{_display_question_title(q)}**")
                st.caption(_display_question_caption(q))


def render_question_guidance(question: Question) -> None:
    guidance = guidance_for_question(question.question_id)
    if guidance is None:
        st.caption("정확하지 않은 값은 ‘미확인’으로, 해당 없는 항목은 ‘해당없음’으로 처리할 수 있습니다.")
        return

    st.markdown(f"<div class='r24-help'>{guidance.help}</div>", unsafe_allow_html=True)
    with st.expander("작성 예시 보기", expanded=False):
        st.markdown("아래는 실제 입력에 참고할 수 있는 예시입니다.")
        st.markdown(guidance.example.replace("\n", "  \n"))
        st.caption(guidance.empty_policy)

def render_form_question(result: Recover24CaseResult, question: Question, strict: bool, *, title: str) -> None:
    custom_renderers = {
        "optional_reports.loss_and_identity": render_optional_reports_question,
        "relief.application": render_relief_question,
        "investigation.status": render_investigation_question,
        "survey.transfer_actor": render_transfer_actor_question,
        "survey.app_and_smishing": render_app_smishing_question,
        "survey.provided_information": render_provided_information_question,
        "survey.banking_usage": render_banking_usage_question,
        "survey.id_phone_storage": render_id_phone_storage_question,
        "survey.pre_incident_security": render_pre_incident_security_question,
        "delegation.proxy_used": render_proxy_used_question,
    }
    renderer = custom_renderers.get(question.question_id)
    if renderer is not None:
        renderer(result, question, strict)
        return

    with st.form(f"form_{question.question_id}"):
        st.caption(title)
        values: dict[str, Any] = {}
        for path in question.target_paths:
            values[path] = render_widget_for_path(path, question.question_id)

        submitted = st.form_submit_button("입력값 반영하기", type="primary")

    if submitted:
        patches = build_form_patches(values, source_text=f"form:{question.question_id}")
        if not patches:
            st.warning("반영할 값이 없습니다. 필수값은 입력하고, 모르면 ‘미확인’ 또는 ‘해당없음’을 선택해주세요.")
            return
        update_result_with_patches(result, patches, strict)



def _choice_to_raw(choice: str) -> Any:
    return BOOLEAN_CHOICE_TO_VALUE.get(choice, choice)


def _radio_value(label: str, *, key: str, include_na: bool = False, default: str = "미확인") -> str:
    options = YES_NO_NA_UNKNOWN if include_na else YES_NO_UNKNOWN
    index = options.index(default) if default in options else 0
    return st.radio(label, options, index=index, horizontal=True, key=key)


def _submit_custom_values(result: Recover24CaseResult, question: Question, strict: bool, values: dict[str, Any]) -> None:
    patches = build_form_patches(values, source_text=f"form:{question.question_id}")
    if not patches:
        st.warning("반영할 값이 없습니다.")
        return
    update_result_with_patches(result, patches, strict)


def render_optional_reports_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        st.markdown("‘예’를 선택한 항목만 신고일을 입력합니다. 아니오 또는 해당없음인 경우 신고일은 문서에서 ‘-’로 표시됩니다.")
        values: dict[str, Any] = {}
        configs = [
            ("optional_reports.id_loss_reported", "optional_reports.id_loss_reported_date", "신분증 분실 신고"),
            ("optional_reports.phone_loss_reported", "optional_reports.phone_loss_reported_date", "휴대전화 분실 신고"),
            ("optional_reports.identity_theft_phone_reported", "optional_reports.identity_theft_phone_reported_date", "명의도용 휴대전화 개설 신고"),
        ]
        for flag_path, date_path, label in configs:
            choice = _radio_value(label, key=_widget_key(question.question_id, flag_path), include_na=True, default="아니오")
            values[flag_path] = _choice_to_raw(choice)
            if choice == "예":
                values[date_path] = st.text_input(label + "일", key=_widget_key(question.question_id, date_path), placeholder="예: 2026년 6월 14일")
            elif choice == "미확인":
                values[date_path] = "__unknown__"
            else:
                values[date_path] = "__not_applicable__"
        values["optional_reports.other"] = st.text_area("기타 사항", key=_widget_key(question.question_id, "optional_reports.other"), height=70, placeholder="예: 명의도용 방지서비스 가입 예정")
        submitted = st.form_submit_button("분실·명의도용 신고 여부 반영하기", type="primary")
    if submitted:
        _submit_custom_values(result, question, strict, values)


def render_relief_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        status_label = st.selectbox("피해구제 신청 상태", list(REPORT_STATUS_OPTIONS), index=1, key=_widget_key(question.question_id, "relief.status"))
        values["relief.status"] = REPORT_STATUS_OPTIONS[status_label]
        st.markdown("여러 은행에 신청한 경우에만 2번, 3번 행을 입력합니다. 빈 행은 신고서에 출력되지 않습니다.")
        for idx in range(1, 4):
            cols = st.columns(2)
            bank_path = f"relief.bank{idx}"
            date_path = f"relief.date{idx}"
            with cols[0]:
                values[bank_path] = st.text_input(f"신청은행 {idx}", key=_widget_key(question.question_id, bank_path), placeholder="예: 우리은행")
            with cols[1]:
                values[date_path] = st.text_input(f"신청일 {idx}", key=_widget_key(question.question_id, date_path), placeholder="예: 2026년 6월 15일")
        submitted = st.form_submit_button("피해구제 신청 정보 반영하기", type="primary")
    if submitted:
        # 빈 은행/일자는 반복 질문이 되지 않도록 행 단위로 처리.
        for idx in range(1, 4):
            bp, dp = f"relief.bank{idx}", f"relief.date{idx}"
            if not str(values.get(bp, "")).strip():
                values[bp] = "__not_applicable__" if idx > 1 else "__unknown__"
            if not str(values.get(dp, "")).strip():
                values[dp] = "__not_applicable__" if idx > 1 else "__unknown__"
        _submit_custom_values(result, question, strict, values)


def render_investigation_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        status_label = st.selectbox("경찰/수사기관 신고 상태", list(REPORT_STATUS_OPTIONS), index=0, key=_widget_key(question.question_id, "investigation.status"))
        values["investigation.status"] = REPORT_STATUS_OPTIONS[status_label]
        status_value = REPORT_STATUS_OPTIONS[status_label]
        needs_detail = status_value in {ReportStatus.REPORTED, ReportStatus.IN_PROGRESS, ReportStatus.COMPLETED, ReportStatus.CLOSED}
        if needs_detail:
            values["investigation.agency"] = st.text_input("수사기관명", key=_widget_key(question.question_id, "investigation.agency"), placeholder="예: 서울중부경찰서")
            values["investigation.reported_at"] = st.text_input("신고일", key=_widget_key(question.question_id, "investigation.reported_at"), placeholder="예: 2026년 6월 14일")
        else:
            st.markdown("미신고, 예정, 미확인 상태라면 수사기관명과 신고일은 입력하지 않아도 됩니다.")
            values["investigation.agency"] = "__not_applicable__" if status_value == ReportStatus.NOT_REPORTED else "__unknown__"
            values["investigation.reported_at"] = "__not_applicable__" if status_value == ReportStatus.NOT_REPORTED else "__unknown__"
        submitted = st.form_submit_button("수사기관 신고 상태 반영하기", type="primary")
    if submitted:
        _submit_custom_values(result, question, strict, values)


def render_transfer_actor_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        choice = st.radio("이번 사고에서 전자금융거래를 실제로 실행한 사람", ["미확인", "손님 본인", "가족", "지인", "불상의 제3자"], index=0, horizontal=True)
        submitted = st.form_submit_button("이체주체 반영하기", type="primary")
    if submitted:
        value = "__unknown__" if choice == "미확인" else choice
        _submit_custom_values(result, question, strict, {"survey.transfer_actor": value})


def render_app_smishing_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        for path, other_path, label in [
            ("survey.smishing_link_clicked", "survey.smishing_link_clicked_other_text", "스미싱/문자/카톡 링크를 클릭했나요?"),
            ("survey.malicious_app_installed", "survey.malicious_app_installed_other_text", "악성앱 또는 원격제어 앱을 설치했나요?"),
        ]:
            choice = st.radio(label, ["미확인", "아니오", "예", "기타/상세 필요"], index=0, horizontal=True, key=_widget_key(question.question_id, path))
            if choice == "기타/상세 필요":
                values[path] = "__unknown__"
                values[other_path] = st.text_input(label + " 상세", key=_widget_key(question.question_id, other_path), placeholder="예: 문자 링크를 눌러 원격제어 앱을 설치했습니다")
            else:
                values[path] = _choice_to_raw(choice)
                values[other_path] = "__not_applicable__"
        submitted = st.form_submit_button("악성 링크·앱 여부 반영하기", type="primary")
    if submitted:
        _submit_custom_values(result, question, strict, values)


def render_provided_information_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        st.markdown("상대방에게 실제로 제공했거나 노출된 항목만 체크해 주세요. 제공한 정보가 없으면 ‘제공하거나 노출한 정보 없음’을 선택합니다.")
        none_provided = st.checkbox("제공하거나 노출한 정보 없음", key=f"{question.question_id}_none")
        bool_paths = [
            ("survey.provided_id_card", "신분증"),
            ("survey.provided_personal_info", "개인정보"),
            ("survey.provided_device", "전자적 장치/휴대전화"),
            ("survey.provided_account_password", "계좌번호 및 계좌 비밀번호"),
            ("survey.provided_security_media", "OTP/보안카드/인증서"),
            ("survey.provided_other_financial_info", "기타 전자금융 관련정보"),
        ]
        values: dict[str, Any] = {}
        for path, label in bool_paths:
            values[path] = False if none_provided else st.checkbox(label, key=_widget_key(question.question_id, path))
        if values.get("survey.provided_other_financial_info"):
            detail = st.text_input("기타 전자금융 관련정보 상세", key=_widget_key(question.question_id, "survey.provided_other_financial_info_text"), placeholder="예: 인증번호 6자리 또는 보안카드 번호 일부")
            values["survey.provided_other_financial_info_text"] = detail if detail.strip() else "__unknown__"
        else:
            values["survey.provided_other_financial_info_text"] = "__not_applicable__"
        submitted = st.form_submit_button("제공정보 반영하기", type="primary")
    if submitted:
        if not none_provided and not any(values[p] for p, _ in bool_paths):
            st.warning("제공한 정보가 없으면 ‘제공하거나 노출한 정보 없음’을 체크해 주세요.")
            return
        _submit_custom_values(result, question, strict, values)


def render_banking_usage_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        configs = [
            ("survey.internet_banking_used", "survey.internet_banking_frequency", "인터넷뱅킹"),
            ("survey.phone_banking_used", "survey.phone_banking_frequency", "폰뱅킹"),
            ("survey.open_banking_used", "survey.open_banking_frequency", "오픈뱅킹"),
        ]
        for used_path, freq_path, label in configs:
            choice = st.radio(label + " 사용 여부", ["미확인", "사용 안 함", "사용함"], index=0, horizontal=True, key=_widget_key(question.question_id, used_path))
            if choice == "사용함":
                values[used_path] = True
                values[freq_path] = st.selectbox(label + " 사용 빈도", ["미확인", "월 1~2회", "주 1회 이상", "거의 매일", "기타"], index=0, key=_widget_key(question.question_id, freq_path))
            elif choice == "사용 안 함":
                values[used_path] = False
                values[freq_path] = "__not_applicable__"
            else:
                values[used_path] = "__unknown__"
                values[freq_path] = "__unknown__"
        submitted = st.form_submit_button("전자금융거래 이용 상태 반영하기", type="primary")
    if submitted:
        _submit_custom_values(result, question, strict, values)


def render_id_phone_storage_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        values["survey.id_lent"] = _choice_to_raw(_radio_value("가족/지인에게 신분증을 대여한 경험", key=_widget_key(question.question_id, "survey.id_lent")))
        values["survey.id_copy_stored_digitally"] = _choice_to_raw(_radio_value("신분증 사본을 휴대전화/PC/클라우드에 저장한 경험", key=_widget_key(question.question_id, "survey.id_copy_stored_digitally")))
        values["survey.id_physical_storage_method"] = st.selectbox("신분증 실물 보관 방식", ["미확인", "지갑", "자택 보관", "휴대전화 케이스", "기타"], index=0, key=_widget_key(question.question_id, "survey.id_physical_storage_method"))
        values["survey.phone_lent"] = _choice_to_raw(_radio_value("가족/지인에게 휴대전화 등 전자적 장치를 제공/대여한 경험", key=_widget_key(question.question_id, "survey.phone_lent")))
        values["survey.phone_lock_method"] = st.selectbox("평상시 휴대전화 잠금 방식", ["미확인", "비밀번호/PIN", "패턴", "지문/얼굴인식", "잠금 없음", "기타"], index=0, key=_widget_key(question.question_id, "survey.phone_lock_method"))
        values["survey.security_media_storage_method"] = st.selectbox("보안카드 등 접근매체 보관 방식", ["미확인", "자택 보관", "지갑 보관", "휴대전화 사진으로 저장", "해당없음", "기타"], index=0, key=_widget_key(question.question_id, "survey.security_media_storage_method"))
        submitted = st.form_submit_button("보관 상태 반영하기", type="primary")
    if submitted:
        for path in ["survey.id_physical_storage_method", "survey.phone_lock_method", "survey.security_media_storage_method"]:
            if values[path] == "미확인":
                values[path] = "__unknown__"
            elif values[path] == "해당없음":
                values[path] = "__not_applicable__"
        _submit_custom_values(result, question, strict, values)


def render_pre_incident_security_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        values: dict[str, Any] = {}
        values["survey.id_loss_reported_before_incident"] = _choice_to_raw(_radio_value("사고 이전 신분증 분실신고 여부", key=_widget_key(question.question_id, "survey.id_loss_reported_before_incident"), include_na=True))
        values["survey.id_copy_stored_before_incident"] = _choice_to_raw(_radio_value("사고 이전 신분증 사본 저장 여부", key=_widget_key(question.question_id, "survey.id_copy_stored_before_incident")))
        values["survey.id_copy_stored_before_incident_other_text"] = "__not_applicable__"
        values["survey.account_password_stored"] = _choice_to_raw(_radio_value("사고 이전 계좌번호/비밀번호 저장 여부", key=_widget_key(question.question_id, "survey.account_password_stored")))
        values["survey.account_password_stored_other_text"] = "__not_applicable__"
        phone_lock_choice = st.radio("사고 이전 휴대전화 잠금 설정 여부", ["미확인", "아니오", "예", "기타"], index=0, horizontal=True, key=_widget_key(question.question_id, "survey.phone_lock_enabled"))
        if phone_lock_choice == "기타":
            values["survey.phone_lock_enabled"] = "__unknown__"
            values["survey.phone_lock_enabled_other_text"] = st.text_input("휴대전화 잠금 관련 기타 내용", key=_widget_key(question.question_id, "survey.phone_lock_enabled_other_text"), placeholder="예: 패턴을 사용했으나 상대방에게 화면을 보여준 적이 있습니다")
        else:
            values["survey.phone_lock_enabled"] = _choice_to_raw(phone_lock_choice)
            values["survey.phone_lock_enabled_other_text"] = "__not_applicable__"
        values["survey.personal_info_leak_suspicion_details"] = st.text_area("개인정보 유출 의심 정황 및 기타 참고사항", height=80, key=_widget_key(question.question_id, "survey.personal_info_leak_suspicion_details"), placeholder="예: 신분증 사진과 인증번호를 상대방에게 보냈습니다")
        submitted = st.form_submit_button("사고 이전 보안 상태 반영하기", type="primary")
    if submitted:
        if not str(values["survey.personal_info_leak_suspicion_details"]).strip():
            values["survey.personal_info_leak_suspicion_details"] = "__not_applicable__"
        _submit_custom_values(result, question, strict, values)


def render_proxy_used_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"form_{question.question_id}"):
        choice = st.radio("신청 방식", ["본인이 직접 신청", "대리인이 신청", "미확인"], index=0, horizontal=True)
        submitted = st.form_submit_button("신청 방식 반영하기", type="primary")
    if submitted:
        if choice == "대리인이 신청":
            value = True
        elif choice == "본인이 직접 신청":
            value = False
        else:
            value = "__unknown__"
        _submit_custom_values(result, question, strict, {"delegation.proxy_used": value})



def render_checkbox_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    with st.form(f"checkbox_{question.question_id}"):
        if question.category == QuestionCategory.CONSENT:
            st.info("동의 항목은 사용자가 직접 확인해야 합니다.")
            agree_all = st.checkbox("필수 동의 항목 전체 동의", key=f"{question.question_id}_agree_all")
            values = {path: agree_all for path in question.target_paths}
            if not agree_all:
                for path in question.target_paths:
                    values[path] = st.checkbox(_label_for_path(path), key=_widget_key(question.question_id, path))
            submitted = st.form_submit_button("동의 상태 반영하기", type="primary")
            if submitted:
                patches = build_form_patches(values, source_text=f"form:{question.question_id}")
                update_result_with_patches(result, patches, strict)
            return

        if question.category == QuestionCategory.EXCLUSION:
            st.info("해당되는 항목만 체크하세요. 해당되는 항목이 없으면 ‘모든 항목 해당없음’을 선택하세요.")
            all_no = st.checkbox("모든 항목 해당없음", key=f"{question.question_id}_all_no")
            checked_paths: dict[str, bool] = {}
            if all_no:
                st.caption("모든 제외대상 항목을 ‘아니오’로 반영합니다.")
                for path in question.target_paths:
                    if path.startswith("exclusion.items.") or path == "exclusion.final_has_exclusion":
                        checked_paths[path] = False
            else:
                for path in question.target_paths:
                    checked_paths[path] = st.checkbox(_label_for_path(path), key=_widget_key(question.question_id, path))
            submitted = st.form_submit_button("제외대상 확인 반영하기", type="primary")
        else:
            choices: dict[str, str] = {}
            for path in question.target_paths:
                choices[path] = st.radio(
                    _label_for_path(path),
                    YES_NO_NA_UNKNOWN,
                    index=0,
                    horizontal=True,
                    key=_widget_key(question.question_id, path),
                )
            submitted = st.form_submit_button("선택값 반영하기", type="primary")
            checked_paths = {}

    if question.category == QuestionCategory.EXCLUSION:
        if submitted:
            if not checked_paths and not all_no:
                st.warning("해당되는 항목이 없으면 ‘모든 항목 해당없음’을 선택해 주세요.")
                return
            patches = [
                Patch(path=path, value=value, status=FieldStatus.ANSWERED, source_text=f"form:{question.question_id}", confidence=1.0)
                for path, value in checked_paths.items()
            ]
            update_result_with_patches(result, patches, strict)
        return

    if submitted:
        patches = build_boolean_choice_patches(choices, source_text=f"form:{question.question_id}")
        if not patches:
            st.warning("반영할 선택값이 없습니다.")
            return
        update_result_with_patches(result, patches, strict)

def render_evidence_question(result: Recover24CaseResult, question: Question, strict: bool) -> None:
    labels = evidence_item_labels()
    with st.form(f"evidence_{question.question_id}"):
        st.info("실제로 보유한 자료만 ‘보유’로 선택해 주세요. 아직 발급받지 않은 자료는 ‘추후 제출’, 확인하지 못한 자료는 ‘모름’으로 둡니다.")
        statuses: dict[str, Any] = {}
        notes: dict[str, str] = {}
        for kind, label in labels.items():
            cols = st.columns([1.1, 1.4])
            with cols[0]:
                label_choice = st.selectbox(
                    label,
                    list(EVIDENCE_STATUS_OPTIONS),
                    index=0,
                    key=f"evidence_{kind}_status",
                )
                statuses[kind] = EVIDENCE_STATUS_OPTIONS[label_choice]
            with cols[1]:
                notes[kind] = st.text_input("비고", key=f"evidence_{kind}_note", placeholder="예: 통화내역 캡처 보유 / 경찰 확인서 발급 예정")
        submitted = st.form_submit_button("증빙 상태 반영하기", type="primary")

    if submitted:
        patches = build_evidence_patches(statuses, notes, source_text=f"form:{question.question_id}")
        if not patches:
            st.warning("반영할 증빙 상태가 없습니다.")
            return
        update_result_with_patches(result, patches, strict)



def render_llm_question(result: Recover24CaseResult, question: Question, provider: LLMProvider, strict: bool) -> None:
    st.markdown("사고 경위나 사고 후 조치 내역을 문장으로 작성해 주세요. 정확한 시간이 기억나지 않으면 ‘경’ 또는 ‘쯤’으로 작성해도 됩니다.")
    answer_key = "answer_text_" + question.question_id.replace(".", "_") + "_" + str(abs(hash(tuple(question.target_paths))))
    answer_text = st.text_area(
        "답변 입력",
        key=answer_key,
        height=180,
        placeholder="예: 검찰청 직원을 사칭한 사람에게 전화를 받고 송금했습니다. 이후 은행에 지급정지를 요청하고 경찰서에 신고했습니다.",
    )

    cols = st.columns([1, 1])
    with cols[0]:
        submit = st.button("답변 반영하기", type="primary", use_container_width=True)
    with cols[1]:
        preview = st.button("작성중인 신고서 보기", use_container_width=True)

    if preview:
        st.session_state.ui_mode = "preview"
        st.rerun()

    if submit:
        if not answer_text.strip():
            st.warning("답변을 입력해 주세요. 아직 정리하기 어렵다면 ‘미확인’이라고 입력할 수 있습니다.")
            return
        try:
            new_result = answer_question(
                result.case,
                question,
                answer_text,
                provider,
                question_limit=st.session_state.question_limit,
                strict_document_completion=strict,
            )
        except Exception as exc:  # pragma: no cover - UI display
            st.error(f"AI 답변 반영 실패: {exc}")
            return
        st.session_state.result = new_result
        st.rerun()

def update_result_with_patches(result: Recover24CaseResult, patches: list[Patch], strict: bool) -> None:
    try:
        new_result = apply_form_patches(
            result.case,
            patches,
            question_limit=st.session_state.question_limit,
            strict_document_completion=strict,
        )
    except Exception as exc:  # pragma: no cover - UI display
        st.error(f"입력값 반영 실패: {exc}")
        return
    st.session_state.result = new_result
    st.rerun()


def render_widget_for_path(path: str, question_id: str) -> Any:
    label = _label_for_path(path)
    key = _widget_key(question_id, path)

    if path in {"relief.status", "investigation.status"}:
        choice = st.selectbox(label, list(REPORT_STATUS_OPTIONS), index=0, key=key)
        return REPORT_STATUS_OPTIONS[choice]

    if path == "incident.fraud_type":
        choice = st.selectbox(label, list(FRAUD_TYPE_OPTIONS), index=0, key=key)
        return FRAUD_TYPE_OPTIONS[choice]

    if path.endswith("transaction_type"):
        choice = st.selectbox(label, list(TRANSACTION_TYPE_OPTIONS), index=0, key=key)
        return TRANSACTION_TYPE_OPTIONS[choice]

    if _is_boolish_path(path):
        choice = st.radio(label, YES_NO_NA_UNKNOWN, index=0, horizontal=True, key=key)
        return BOOLEAN_CHOICE_TO_VALUE.get(choice, "__unknown__")

    if path.endswith("amount_krw"):
        st.caption("원 단위 숫자로 입력하세요. 모르면 0원으로 두면 나중에 다시 확인할 수 있습니다.")
        amount = st.number_input(label, min_value=0, step=10000, key=key)
        return "__skip__" if amount == 0 else int(amount)

    placeholder = placeholder_for_path(path)

    if path.endswith("overview") or path.endswith("memo") or path.endswith("other") or path.endswith("purpose") or path.endswith("details"):
        return st.text_area(label, key=key, height=80, placeholder=placeholder)

    return st.text_input(label, key=key, placeholder=placeholder)


def render_html_preview(result: Recover24CaseResult, *, height: int = 720) -> None:
    try:
        html = render_case_html(result.case, template_dir=DEFAULT_TEMPLATE_DIR)
    except Exception as exc:  # pragma: no cover - Streamlit error display
        st.error(f"HTML 렌더링 실패: {exc}")
        return

    st.download_button(
        label="공식 HTML 다운로드",
        data=html,
        file_name=f"{result.case.case_id}_official_report.html",
        mime="text/html",
        use_container_width=True,
    )
    st.components.v1.html(html, height=height, scrolling=True)


def render_debug_panels(result: Recover24CaseResult) -> None:
    with st.expander("Case JSON"):
        st.json(result.case.to_dict())
    with st.expander("Latest Patch[]"):
        st.json([_patch_to_dict(patch) for patch in result.patches])
    with st.expander("Document View Context"):
        st.json(result.document_view.to_dict())



def _display_question_title(question: Question) -> str:
    titles = {
        "applicant.basic": "신청인 기본정보를 확인해 주세요.",
        "applicant.additional": "연락처와 추가 정보를 확인해 주세요.",
        "applicant.corporate": "법인 신청인 경우에만 입력해 주세요.",
        "exclusion.primary_checklist": "피해 신고 제외대상에 해당하는 항목이 있나요?",
        "exclusion.secondary_checklist": "추가 제외대상 및 최종 해당 여부를 확인해 주세요.",
        "incident.timeline": "사고 발생·인지·지급정지 요청 시점을 확인해 주세요.",
        "incident.summary": "피해유형과 사건개요를 정리해 주세요.",
        "transaction.primary_details": "피해 거래 정보를 입력해 주세요.",
        "optional_reports.loss_and_identity": "분실신고·명의도용 신고 여부를 확인해 주세요.",
        "relief.application": "피해구제 신청 상태를 확인해 주세요.",
        "investigation.status": "경찰 또는 수사기관 신고 상태를 확인해 주세요.",
        "survey.transfer_actor": "전자금융거래를 실행한 사람을 확인해 주세요.",
        "survey.app_and_smishing": "악성 링크·앱 설치 여부를 확인해 주세요.",
        "survey.provided_information": "상대방에게 제공한 정보를 확인해 주세요.",
        "survey.banking_usage": "평소 전자금융거래 이용 상태를 확인해 주세요.",
        "survey.id_phone_storage": "신분증·휴대전화·보안매체 보관 상태를 확인해 주세요.",
        "survey.pre_incident_security": "사고 이전 보안 상태와 유출 의심 정황을 확인해 주세요.",
        "narrative.drafts": "사고 경위와 사고 후 조치 내역을 작성해 주세요.",
        "consent.required_bundle": "필수 개인정보 동의 항목을 확인해 주세요.",
        "delegation.proxy_used": "본인 신청인지 대리 신청인지 확인해 주세요.",
        "delegation.agent_details": "대리인 정보를 입력해 주세요.",
        "evidence.current_items": "첨부서류 보유 상태를 확인해 주세요.",
    }
    return titles.get(question.question_id, question.prompt)


def _display_question_caption(question: Question) -> str:
    if question.question_id.startswith("exclusion."):
        return "해당되는 항목만 체크합니다. 해당 항목이 없으면 ‘모든 항목 해당없음’을 선택하세요."
    if question.input_type == QuestionInputType.LLM_TEXT:
        return "문장으로 작성하면 신고서 문체로 정리합니다."
    if question.input_type == QuestionInputType.FORM:
        return "안내문을 참고해 아는 범위에서 입력해 주세요."
    if question.input_type == QuestionInputType.EVIDENCE:
        return "보유·추후 제출·미제출·해당없음 중 현재 상태를 선택합니다."
    return "필요한 항목만 확인합니다."

def _patch_to_dict(patch: Any) -> dict[str, Any]:
    return {
        "path": patch.path,
        "value": patch.value.value if hasattr(patch.value, "value") else patch.value,
        "status": patch.status.value,
        "source_text": patch.source_text,
        "confidence": patch.confidence,
    }


def _label_for_path(path: str) -> str:
    return FIELD_LABELS.get(path, path)


def _widget_key(question_id: str, path: str) -> str:
    return "widget_" + question_id.replace(".", "_") + "_" + path.replace(".", "_")


def _is_boolish_path(path: str) -> bool:
    if path in {
        "applicant.sms_consent",
        "exclusion.final_has_exclusion",
        "delegation.proxy_used",
        "survey.id_copy_stored_before_incident",
        "survey.account_password_stored",
        "survey.id_loss_reported_before_incident",
        "survey.phone_lock_enabled",
        "survey.id_copy_stored_digitally",
    }:
        return True
    if path.startswith("exclusion.items.") or path.startswith("consent."):
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


if __name__ == "__main__":
    main()
