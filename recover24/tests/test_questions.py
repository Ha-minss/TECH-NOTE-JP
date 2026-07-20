from recover24.models import (
    FieldStatus,
    FraudType,
    Patch,
    QuestionCategory,
    RecoveryCase,
    ReportStatus,
    TransactionType,
)
from recover24.patching import apply_patches
from recover24.questions import (
    all_question_label_paths,
    build_next_questions,
    collect_field_value_paths,
    evidence_item_labels,
)


def _question_by_id(case: RecoveryCase, question_id: str):
    questions = build_next_questions(case, limit=50)
    return next((question for question in questions if question.question_id == question_id), None)


def test_every_model_fieldvalue_path_has_question_label():
    case = RecoveryCase.new("CASE-Q-COVERAGE")

    model_paths = collect_field_value_paths(case)
    label_paths = all_question_label_paths()

    assert model_paths - label_paths == set()


def test_all_labeled_paths_appear_in_question_groups_or_are_evidence_special_case():
    case = RecoveryCase.new("CASE-Q-GROUP-COVERAGE")
    questions = build_next_questions(case, limit=50)
    target_paths = {path for question in questions for path in question.target_paths}

    assert all_question_label_paths() - target_paths == set()


def test_evidence_prompt_lists_all_official_attachment_labels():
    case = RecoveryCase.new("CASE-Q-EVIDENCE")
    questions = build_next_questions(case, limit=50)
    evidence_question = next(question for question in questions if question.question_id == "evidence.current_items")

    for label in evidence_item_labels().values():
        assert label in evidence_question.prompt


def test_transaction_question_asks_only_missing_transaction_fields():
    case = RecoveryCase.new("CASE-Q-001")
    case = apply_patches(
        case,
        [
            Patch("transactions.0.source_bank", "국민은행"),
            Patch("transactions.0.source_account_number", "123-456"),
            Patch("transactions.0.destination_bank", "카카오뱅크"),
            Patch("transactions.0.destination_account_number", "3333-01-123456"),
            Patch("transactions.0.amount_krw", 1_000_000),
            Patch("transactions.0.transaction_type", TransactionType.MOBILE_BANKING_TRANSFER),
            Patch("transactions.0.holder_type", "타인"),
        ],
    )

    question = _question_by_id(case, "transaction.primary_details")

    assert question is not None
    assert question.category == QuestionCategory.TRANSACTION
    assert question.target_paths == [
        "transactions.0.destination_account_holder",
        "transactions.0.transferred_at",
    ]
    assert "수취인/예금주 이름" in question.prompt
    assert "송금시각" in question.prompt
    assert "출금은행" not in question.prompt
    assert "입금은행" not in question.prompt


def test_transaction_question_skips_answered_unknown_and_not_applicable_fields():
    case = RecoveryCase.new("CASE-Q-002")
    case = apply_patches(
        case,
        [
            Patch("transactions.0.source_bank", "국민은행"),
            Patch("transactions.0.source_account_number", "123-456"),
            Patch("transactions.0.destination_bank", "카카오뱅크"),
            Patch("transactions.0.destination_account_number", "3333-01-123456"),
            Patch("transactions.0.destination_account_holder", status=FieldStatus.UNKNOWN, source_text="예금주는 모르겠습니다"),
            Patch("transactions.0.transferred_at", status=FieldStatus.NOT_APPLICABLE, source_text="정확한 시간은 해당 없음"),
            Patch("transactions.0.amount_krw", 1_000_000),
            Patch("transactions.0.transaction_type", TransactionType.MOBILE_BANKING_TRANSFER),
            Patch("transactions.0.holder_type", "타인"),
        ],
    )

    question = _question_by_id(case, "transaction.primary_details")

    assert question is None


def test_default_case_returns_grouped_questions_in_priority_order():
    case = RecoveryCase.new("CASE-Q-003")

    questions = build_next_questions(case, limit=3)

    assert [question.question_id for question in questions] == [
        "applicant.basic",
        "applicant.additional",
        "applicant.corporate",
    ]
    assert questions[0].target_paths == [
        "applicant.name",
        "applicant.birth_date",
        "applicant.mobile_number",
        "applicant.address",
    ]
    assert "성명" in questions[0].prompt
    assert "생년월일" in questions[0].prompt


def test_report_groups_cover_relief_and_investigation_details():
    case = RecoveryCase.new("CASE-Q-004")
    case = apply_patches(
        case,
        [
            Patch("applicant.name", "김민수"),
            Patch("applicant.birth_date", "1990-01-01"),
            Patch("applicant.mobile_number", "010-1234-5678"),
            Patch("applicant.address", "서울시"),
            Patch("applicant.customer_number", status=FieldStatus.NOT_APPLICABLE),
            Patch("applicant.phone_number", status=FieldStatus.NOT_APPLICABLE),
            Patch("applicant.email", status=FieldStatus.NOT_APPLICABLE),
            Patch("applicant.memo", status=FieldStatus.NOT_APPLICABLE),
            Patch("applicant.sms_consent", True),
            Patch("applicant.company_name", status=FieldStatus.NOT_APPLICABLE),
            Patch("applicant.business_number", status=FieldStatus.NOT_APPLICABLE),
            Patch("incident.first_occurred_at", "2026-06-20 14:00"),
            Patch("incident.recognized_at", "2026-06-20 15:00"),
            Patch("incident.first_freeze_at", "2026-06-20 16:00"),
            Patch("incident.fraud_type", FraudType.AUTHORITY_IMPERSONATION),
            Patch("incident.overview", "검찰 사칭 피해"),
            Patch("transactions.0.source_bank", "국민은행"),
            Patch("transactions.0.source_account_number", "123-456"),
            Patch("transactions.0.amount_krw", 1_000_000),
            Patch("transactions.0.destination_bank", "카카오뱅크"),
            Patch("transactions.0.destination_account_number", "3333"),
            Patch("transactions.0.destination_account_holder", "홍길동"),
            Patch("transactions.0.holder_type", "타인"),
            Patch("transactions.0.transaction_type", TransactionType.MOBILE_BANKING_TRANSFER),
            Patch("transactions.0.transferred_at", "2026-06-20 14:10"),
            Patch("relief.status", ReportStatus.IN_PROGRESS),
            Patch("investigation.status", ReportStatus.REPORTED),
        ],
    )

    relief_question = _question_by_id(case, "relief.application")
    investigation_question = _question_by_id(case, "investigation.status")

    assert relief_question is not None
    assert "relief.bank1" in relief_question.target_paths
    assert "relief.date1" in relief_question.target_paths
    assert investigation_question is not None
    assert investigation_question.target_paths == ["investigation.agency", "investigation.reported_at"]


def test_consent_question_comes_after_other_main_missing_groups_when_limit_is_large():
    case = RecoveryCase.new("CASE-Q-005")
    questions = build_next_questions(case, limit=50)
    ids = [question.question_id for question in questions]

    assert "consent.required_bundle" in ids
    assert ids.index("consent.required_bundle") > ids.index("delegation.agent_details")


def test_delegation_details_are_skipped_when_proxy_not_used():
    case = RecoveryCase.new("CASE-Q-006")
    case = apply_patches(case, [Patch("delegation.proxy_used", False)])

    question = _question_by_id(case, "delegation.agent_details")

    assert question is None
