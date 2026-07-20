"""Recover24 V3 raw state models.

models.py is the case record, not the document renderer.

Rules:
- Store raw facts only: numbers stay numbers, booleans stay booleans.
- Do not store HTML labels like "1,000,000원" or checkbox marks like "☑" here.
- Every user-answerable field can distinguish:
  answered / not_asked / unknown / not_applicable.
- Only patching.py should update RecoveryCase values at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class FieldStatus(str, Enum):
    """State of one user-answerable field.

    ANSWERED: we have a real value.
    NOT_ASKED: we have not asked the user yet.
    UNKNOWN: user was asked but said they do not know / cannot confirm.
    NOT_APPLICABLE: user confirmed this field does not apply.
    """

    ANSWERED = "answered"
    NOT_ASKED = "not_asked"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(slots=True)
class FieldValue(Generic[T]):
    """A value plus its collection state.

    Why wrap values?
    - None alone cannot tell "not asked yet" from "asked, but user does not know".
    - Official forms need that distinction because the next question depends on it.
    """

    value: T | None = None
    status: FieldStatus = FieldStatus.NOT_ASKED
    source_text: str | None = None

    @classmethod
    def answered(cls, value: T, source_text: str | None = None) -> "FieldValue[T]":
        return cls(value=value, status=FieldStatus.ANSWERED, source_text=source_text)

    @classmethod
    def unknown(cls, source_text: str | None = None) -> "FieldValue[T]":
        return cls(value=None, status=FieldStatus.UNKNOWN, source_text=source_text)

    @classmethod
    def not_applicable(cls, source_text: str | None = None) -> "FieldValue[T]":
        return cls(value=None, status=FieldStatus.NOT_APPLICABLE, source_text=source_text)

    @property
    def has_answer(self) -> bool:
        return self.status == FieldStatus.ANSWERED and self.value is not None

    @property
    def needs_question(self) -> bool:
        return self.status == FieldStatus.NOT_ASKED


class FraudType(str, Enum):
    AUTHORITY_IMPERSONATION = "authority_impersonation"  # 검찰/경찰/금감원 등 수사기관 사칭
    FAMILY_IMPERSONATION = "family_impersonation"        # 자녀/가족 사칭 메신저피싱
    LOAN_SCAM = "loan_scam"                              # 대출빙자
    SMISHING_MALWARE = "smishing_malware"                # 스미싱/악성앱/원격제어
    INSTITUTION_IMPERSONATION = "institution_impersonation"  # 카드사/은행/택배 등 기관 사칭
    OTHER = "other"


class TransactionType(str, Enum):
    MOBILE_BANKING_TRANSFER = "mobile_banking_transfer"
    INTERNET_BANKING_TRANSFER = "internet_banking_transfer"
    PHONE_BANKING_TRANSFER = "phone_banking_transfer"
    ATM_TRANSFER = "atm_transfer"
    CARD_OR_LOAN = "card_or_loan"
    UNKNOWN = "unknown"


class ReportStatus(str, Enum):
    NOT_REPORTED = "not_reported"
    PLANNED = "planned"
    REPORTED = "reported"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CLOSED = "closed"
    OTHER = "other"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    AVAILABLE = "available"          # 보유
    MISSING = "missing"              # 없음
    PLANNED = "planned"              # 추후 제출 예정
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    NOT_ASKED = "not_asked"


class QuestionCategory(str, Enum):
    APPLICANT = "applicant"
    EXCLUSION = "exclusion"
    INCIDENT = "incident"
    TRANSACTION = "transaction"
    REPORT_AND_FREEZE = "report_and_freeze"
    SURVEY = "survey"
    NARRATIVE = "narrative"
    EVIDENCE = "evidence"
    CONSENT = "consent"
    DELEGATION = "delegation"


class QuestionInputType(str, Enum):
    """How the UI should collect an answer for a question.

    This keeps the V3 pipeline intact while preventing structured official-form
    fields from going through an LLM unnecessarily.
    """

    FORM = "form"                  # text/date/number/select widgets -> Patch[]
    RADIO = "radio"                # yes/no/unknown or status choices -> Patch[]
    CHECKBOX = "checkbox"          # explicit consent/checklist values -> Patch[]
    TABLE = "table"                # transaction-like grouped structured fields -> Patch[]
    EVIDENCE = "evidence"          # attachment status selectboxes -> Patch[]
    LLM_TEXT = "llm_text"          # free Korean narrative -> answers.py/Gemma


@dataclass(slots=True)
class Applicant:
    """Page 1 applicant facts."""

    name: FieldValue[str] = field(default_factory=FieldValue[str])
    birth_date: FieldValue[str] = field(default_factory=FieldValue[str])
    customer_number: FieldValue[str] = field(default_factory=FieldValue[str])
    company_name: FieldValue[str] = field(default_factory=FieldValue[str])
    business_number: FieldValue[str] = field(default_factory=FieldValue[str])
    phone_number: FieldValue[str] = field(default_factory=FieldValue[str])
    mobile_number: FieldValue[str] = field(default_factory=FieldValue[str])
    email: FieldValue[str] = field(default_factory=FieldValue[str])
    address: FieldValue[str] = field(default_factory=FieldValue[str])
    memo: FieldValue[str] = field(default_factory=FieldValue[str])
    sms_consent: FieldValue[bool] = field(default_factory=FieldValue[bool])


@dataclass(slots=True)
class ExclusionScreening:
    """Page 1-2 exclusion checklist.

    Keys are stable item ids, not HTML checkbox marks.
    document_view.py converts these values to checkbox strings.
    """

    items: dict[str, FieldValue[bool]] = field(
        default_factory=lambda: {f"exclude_{i}": FieldValue[bool]() for i in range(1, 18)}
    )
    final_has_exclusion: FieldValue[bool] = field(default_factory=FieldValue[bool])


@dataclass(slots=True)
class Incident:
    """Page 2 incident summary facts."""

    first_occurred_at: FieldValue[str] = field(default_factory=FieldValue[str])
    recognized_at: FieldValue[str] = field(default_factory=FieldValue[str])
    first_freeze_at: FieldValue[str] = field(default_factory=FieldValue[str])
    fraud_type: FieldValue[FraudType] = field(default_factory=FieldValue[FraudType])
    overview: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class Transaction:
    """One damaged transaction row from Page 2."""

    source_bank: FieldValue[str] = field(default_factory=FieldValue[str])
    source_account_number: FieldValue[str] = field(default_factory=FieldValue[str])
    amount_krw: FieldValue[int] = field(default_factory=FieldValue[int])
    destination_bank: FieldValue[str] = field(default_factory=FieldValue[str])
    destination_account_number: FieldValue[str] = field(default_factory=FieldValue[str])
    destination_account_holder: FieldValue[str] = field(default_factory=FieldValue[str])
    holder_type: FieldValue[str] = field(default_factory=FieldValue[str])  # 본인 / 타인 / 불명
    transaction_type: FieldValue[TransactionType] = field(default_factory=FieldValue[TransactionType])
    transferred_at: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class OptionalReports:
    """Page 2 optional loss/identity-theft reports."""

    id_loss_reported: FieldValue[bool] = field(default_factory=FieldValue[bool])
    id_loss_reported_date: FieldValue[str] = field(default_factory=FieldValue[str])
    phone_loss_reported: FieldValue[bool] = field(default_factory=FieldValue[bool])
    phone_loss_reported_date: FieldValue[str] = field(default_factory=FieldValue[str])
    identity_theft_phone_reported: FieldValue[bool] = field(default_factory=FieldValue[bool])
    identity_theft_phone_reported_date: FieldValue[str] = field(default_factory=FieldValue[str])
    other: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class ReliefApplication:
    """Page 2 victim relief refund application status."""

    status: FieldValue[ReportStatus] = field(default_factory=FieldValue[ReportStatus])
    bank1: FieldValue[str] = field(default_factory=FieldValue[str])
    date1: FieldValue[str] = field(default_factory=FieldValue[str])
    bank2: FieldValue[str] = field(default_factory=FieldValue[str])
    date2: FieldValue[str] = field(default_factory=FieldValue[str])
    bank3: FieldValue[str] = field(default_factory=FieldValue[str])
    date3: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class Investigation:
    """Page 2 police/investigation status."""

    status: FieldValue[ReportStatus] = field(default_factory=FieldValue[ReportStatus])
    agency: FieldValue[str] = field(default_factory=FieldValue[str])
    reported_at: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class SecuritySurvey:
    """Page 3 electronic-finance incident survey facts.

    This is intentionally raw. Checkbox marks are generated in document_view.py.
    """

    transfer_actor: FieldValue[str] = field(default_factory=FieldValue[str])
    smishing_link_clicked: FieldValue[bool] = field(default_factory=FieldValue[bool])
    smishing_link_clicked_other_text: FieldValue[str] = field(default_factory=FieldValue[str])
    malicious_app_installed: FieldValue[bool] = field(default_factory=FieldValue[bool])
    malicious_app_installed_other_text: FieldValue[str] = field(default_factory=FieldValue[str])

    provided_id_card: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_personal_info: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_device: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_account_password: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_security_media: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_other_financial_info: FieldValue[bool] = field(default_factory=FieldValue[bool])
    provided_other_financial_info_text: FieldValue[str] = field(default_factory=FieldValue[str])

    internet_banking_used: FieldValue[bool] = field(default_factory=FieldValue[bool])
    internet_banking_frequency: FieldValue[str] = field(default_factory=FieldValue[str])
    phone_banking_used: FieldValue[bool] = field(default_factory=FieldValue[bool])
    phone_banking_frequency: FieldValue[str] = field(default_factory=FieldValue[str])
    open_banking_used: FieldValue[bool] = field(default_factory=FieldValue[bool])
    open_banking_frequency: FieldValue[str] = field(default_factory=FieldValue[str])

    id_lent: FieldValue[bool] = field(default_factory=FieldValue[bool])
    id_copy_stored_digitally: FieldValue[bool] = field(default_factory=FieldValue[bool])
    id_physical_storage_method: FieldValue[str] = field(default_factory=FieldValue[str])
    phone_lent: FieldValue[bool] = field(default_factory=FieldValue[bool])
    phone_lock_method: FieldValue[str] = field(default_factory=FieldValue[str])
    security_media_storage_method: FieldValue[str] = field(default_factory=FieldValue[str])

    id_loss_reported_before_incident: FieldValue[bool] = field(default_factory=FieldValue[bool])
    id_copy_stored_before_incident: FieldValue[bool] = field(default_factory=FieldValue[bool])
    id_copy_stored_before_incident_other_text: FieldValue[str] = field(default_factory=FieldValue[str])
    account_password_stored: FieldValue[bool] = field(default_factory=FieldValue[bool])
    account_password_stored_other_text: FieldValue[str] = field(default_factory=FieldValue[str])
    phone_lock_enabled: FieldValue[bool] = field(default_factory=FieldValue[bool])
    phone_lock_enabled_other_text: FieldValue[str] = field(default_factory=FieldValue[str])
    personal_info_leak_suspicion_details: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class Narrative:
    """Page 4 narrative raw/draft fields.

    LLM may draft text here from raw facts, but HTML formatting still belongs to document_view/html_renderer.
    """

    incident_circumstances: FieldValue[str] = field(default_factory=FieldValue[str])
    post_action: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class Consent:
    """Page 5-6 required consent answers.

    Consent must be explicit. Do not infer True from context.
    """

    unique_id_collection_agreed: FieldValue[bool] = field(default_factory=FieldValue[bool])
    personal_credit_collection_agreed: FieldValue[bool] = field(default_factory=FieldValue[bool])
    unique_id_provision_agreed: FieldValue[bool] = field(default_factory=FieldValue[bool])
    personal_credit_provision_agreed: FieldValue[bool] = field(default_factory=FieldValue[bool])


@dataclass(slots=True)
class EvidenceItem:
    """One evidence/attachment item from Page 7."""

    kind: str
    status: EvidenceStatus = EvidenceStatus.NOT_ASKED
    note: str | None = None
    source_text: str | None = None


@dataclass(slots=True)
class Delegation:
    """Page 8 delegation/proxy facts."""

    proxy_used: FieldValue[bool] = field(default_factory=FieldValue[bool])
    agent_name: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_birth_date: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_phone_number: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_mobile_number: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_email: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_address: FieldValue[str] = field(default_factory=FieldValue[str])
    agent_memo: FieldValue[str] = field(default_factory=FieldValue[str])
    request_purpose: FieldValue[str] = field(default_factory=FieldValue[str])


@dataclass(slots=True)
class RecoveryCase:
    """Single source of truth for one Recover24 case."""

    case_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    applicant: Applicant = field(default_factory=Applicant)
    exclusion: ExclusionScreening = field(default_factory=ExclusionScreening)
    incident: Incident = field(default_factory=Incident)
    transactions: list[Transaction] = field(default_factory=lambda: [Transaction()])
    optional_reports: OptionalReports = field(default_factory=OptionalReports)
    relief: ReliefApplication = field(default_factory=ReliefApplication)
    investigation: Investigation = field(default_factory=Investigation)
    survey: SecuritySurvey = field(default_factory=SecuritySurvey)
    narrative: Narrative = field(default_factory=Narrative)
    evidence: list[EvidenceItem] = field(default_factory=list)
    consent: Consent = field(default_factory=Consent)
    delegation: Delegation = field(default_factory=Delegation)

    @classmethod
    def new(cls, case_id: str) -> "RecoveryCase":
        return cls(case_id=case_id)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True, slots=True)
class Patch:
    """A requested change to RecoveryCase.

    extraction.py and answers.py create Patch objects.
    patching.py is the only module that applies them.
    """

    path: str
    value: Any = None
    status: FieldStatus = FieldStatus.ANSWERED
    source_text: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Question:
    """One user-facing question for a missing part of the case."""

    question_id: str
    category: QuestionCategory
    prompt: str
    target_paths: list[str]
    required: bool = True
    input_type: QuestionInputType = QuestionInputType.LLM_TEXT


def _to_plain(value: Any) -> Any:
    """Convert dataclasses/enums to JSON-friendly plain Python values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    return value
