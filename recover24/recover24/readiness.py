"""Submission readiness checks for Recover24 V3.

readiness.py answers: "What can this case do next?"

Rules:
- Do not mutate RecoveryCase.
- Do not call an LLM.
- Do not render HTML.
- Do not ask questions.
- Separate user-action gaps from staff-review gaps so the bank does not
  re-ask questions the customer already answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from .models import EvidenceStatus, FieldStatus, FieldValue, FraudType, RecoveryCase, ReportStatus
from .questions import EVIDENCE_ITEM_LABELS, FIELD_LABELS, collect_field_value_paths


class ReadinessStatus(str, Enum):
    """High-level routing state for the case."""

    NOT_READY = "not_ready"              # user must still provide/confirm information
    NEEDS_REVIEW = "needs_review"        # user answered, but staff judgment is required
    READY = "ready"                      # package is ready for bank review/submission


class IssueSeverity(str, Enum):
    """How strongly an issue blocks the next step."""

    BLOCKER = "blocker"  # blocks official submission
    REVIEW = "review"    # needs staff decision, not more duplicate questioning
    WARNING = "warning"  # useful to show, but does not block by itself


class ResolutionOwner(str, Enum):
    """Who should act next on the issue."""

    USER = "user"
    STAFF = "staff"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    """One concrete reason why the case is not fully ready."""

    code: str
    severity: IssueSeverity
    category: str
    message: str
    paths: list[str]
    labels: list[str]
    resolution_owner: ResolutionOwner

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "paths": list(self.paths),
            "labels": list(self.labels),
            "resolution_owner": self.resolution_owner.value,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Submission/readiness summary for UI and bank-staff routing."""

    status: ReadinessStatus
    can_render_document: bool
    can_request_staff_review: bool
    can_submit_officially: bool
    requires_user_action: bool
    requires_staff_decision: bool
    document_completion_rate: float
    answered_field_count: int
    total_field_count: int
    issues: list[ReadinessIssue]

    @property
    def blocker_issues(self) -> list[ReadinessIssue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.BLOCKER]

    @property
    def review_issues(self) -> list[ReadinessIssue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.REVIEW]

    @property
    def warning_issues(self) -> list[ReadinessIssue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "can_render_document": self.can_render_document,
            "can_request_staff_review": self.can_request_staff_review,
            "can_submit_officially": self.can_submit_officially,
            "requires_user_action": self.requires_user_action,
            "requires_staff_decision": self.requires_staff_decision,
            "document_completion_rate": self.document_completion_rate,
            "answered_field_count": self.answered_field_count,
            "total_field_count": self.total_field_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


# Critical fields must be answered with usable values for an official package.
# Everything else still participates in strict HTML completion, but these produce
# clearer high-priority issue messages.
CRITICAL_APPLICANT_PATHS: tuple[str, ...] = (
    "applicant.name",
    "applicant.birth_date",
    "applicant.mobile_number",
    "applicant.address",
)

CRITICAL_INCIDENT_PATHS: tuple[str, ...] = (
    "incident.first_occurred_at",
    "incident.recognized_at",
    "incident.first_freeze_at",
    "incident.fraud_type",
    "incident.overview",
)

CRITICAL_TRANSACTION_PATHS: tuple[str, ...] = (
    "transactions.0.source_bank",
    "transactions.0.source_account_number",
    "transactions.0.amount_krw",
    "transactions.0.destination_bank",
    "transactions.0.destination_account_number",
    "transactions.0.destination_account_holder",
    "transactions.0.transaction_type",
    "transactions.0.transferred_at",
)

CRITICAL_NARRATIVE_PATHS: tuple[str, ...] = (
    "narrative.incident_circumstances",
    "narrative.post_action",
)

CONSENT_PATHS: tuple[str, ...] = (
    "consent.unique_id_collection_agreed",
    "consent.personal_credit_collection_agreed",
    "consent.unique_id_provision_agreed",
    "consent.personal_credit_provision_agreed",
)

EXCLUSION_PATHS: tuple[str, ...] = tuple(
    [f"exclusion.items.exclude_{index}" for index in range(1, 18)]
    + ["exclusion.final_has_exclusion"]
)

# Evidence page has 13 official attachment rows. The user should at least confirm
# each row as available / missing / planned / not applicable / unknown.
OFFICIAL_EVIDENCE_KINDS: tuple[str, ...] = tuple(EVIDENCE_ITEM_LABELS)

# Conditional review rules. These do not necessarily mean "ask the customer again";
# they mean the case should be routed with context to staff if the evidence is not available.
FRAUD_TYPE_REVIEW_EVIDENCE: dict[FraudType, tuple[str, str]] = {
    FraudType.FAMILY_IMPERSONATION: (
        "other_evidence",
        "자녀·가족 사칭 유형은 카카오톡/문자 대화 캡처 등 사칭 경로 증빙 상태를 직원이 확인해야 합니다.",
    ),
    FraudType.AUTHORITY_IMPERSONATION: (
        "phone_evidence",
        "수사기관 사칭 유형은 통화기록·문자·앱 안내 등 접촉 경로 증빙 상태를 직원이 확인해야 합니다.",
    ),
    FraudType.INSTITUTION_IMPERSONATION: (
        "phone_evidence",
        "기관 사칭 유형은 문자·전화·앱 안내 등 접촉 경로 증빙 상태를 직원이 확인해야 합니다.",
    ),
    FraudType.LOAN_SCAM: (
        "phone_evidence",
        "대출빙자 유형은 통화기록·문자·상담내역 등 접촉 경로 증빙 상태를 직원이 확인해야 합니다.",
    ),
    FraudType.SMISHING_MALWARE: (
        "phone_evidence",
        "스미싱/악성앱 유형은 문자 링크·앱 설치·원격제어 관련 증빙 상태를 직원이 확인해야 합니다.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_readiness(case: RecoveryCase, *, strict_document_completion: bool = True) -> ReadinessReport:
    """Evaluate whether a case is ready, needs more user input, or needs staff review.

    strict_document_completion=True means every non-ignored HTML/model field must
    be at least confirmed. Confirmed can be answered, unknown, or not_applicable;
    NOT_ASKED means the customer-facing flow is not finished yet.
    """

    issues: list[ReadinessIssue] = []

    actionable_paths = _actionable_field_paths(case)
    completion = _completion_stats(case, actionable_paths)

    if strict_document_completion:
        unanswered_paths = [path for path in sorted(actionable_paths) if _field_status(case, path) == FieldStatus.NOT_ASKED]
        if unanswered_paths:
            issues.append(
                _issue(
                    code="document_fields_not_confirmed",
                    severity=IssueSeverity.BLOCKER,
                    category="document_completion",
                    message=(
                        "공식 신고서 HTML에 들어가는 항목 중 아직 확인되지 않은 값이 있습니다. "
                        "제출 전에 사용자에게 추가 질문이 필요합니다."
                    ),
                    paths=unanswered_paths,
                    owner=ResolutionOwner.USER,
                )
            )

    issues.extend(_critical_field_issues(case))
    issues.extend(_consent_issues(case))
    issues.extend(_exclusion_issues(case))
    issues.extend(_evidence_confirmation_issues(case, strict_document_completion=strict_document_completion))
    issues.extend(_conditional_review_issues(case))
    issues.extend(_report_status_review_issues(case))
    issues.extend(_delegation_issues(case))

    status = _status_from_issues(issues)
    requires_user_action = any(
        issue.severity == IssueSeverity.BLOCKER and issue.resolution_owner == ResolutionOwner.USER
        for issue in issues
    )
    requires_staff_decision = any(
        issue.resolution_owner == ResolutionOwner.STAFF
        and issue.severity in {IssueSeverity.BLOCKER, IssueSeverity.REVIEW}
        for issue in issues
    )

    return ReadinessReport(
        status=status,
        can_render_document=True,
        can_request_staff_review=status in {ReadinessStatus.NEEDS_REVIEW, ReadinessStatus.READY},
        can_submit_officially=status == ReadinessStatus.READY,
        requires_user_action=requires_user_action,
        requires_staff_decision=requires_staff_decision,
        document_completion_rate=completion["rate"],
        answered_field_count=completion["answered"],
        total_field_count=completion["total"],
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Issue builders
# ---------------------------------------------------------------------------


def _critical_field_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    issues: list[ReadinessIssue] = []
    groups: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
        (
            "missing_applicant_core",
            "applicant",
            "신청인 필수정보가 부족합니다. 성명, 생년월일, 휴대전화번호, 주소를 확인해야 합니다.",
            CRITICAL_APPLICANT_PATHS,
        ),
        (
            "missing_incident_core",
            "incident",
            "사고 발생·인지·지급정지 시각 또는 피해유형/사건개요가 부족합니다.",
            CRITICAL_INCIDENT_PATHS,
        ),
        (
            "missing_transaction_identity",
            "transaction",
            "피해 거래를 특정하기 위한 핵심 거래정보가 부족합니다.",
            CRITICAL_TRANSACTION_PATHS,
        ),
        (
            "missing_narrative_core",
            "narrative",
            "신고서에 들어갈 사고 경위와 사고 인지 후 조치 내역이 부족합니다.",
            CRITICAL_NARRATIVE_PATHS,
        ),
    )

    for code, category, message, paths in groups:
        missing_or_unusable = [path for path in paths if not _has_answer(case, path)]
        if missing_or_unusable:
            owner = ResolutionOwner.USER if any(_field_status(case, path) == FieldStatus.NOT_ASKED for path in missing_or_unusable) else ResolutionOwner.STAFF
            severity = IssueSeverity.BLOCKER if owner == ResolutionOwner.USER else IssueSeverity.REVIEW
            issues.append(
                _issue(
                    code=code,
                    severity=severity,
                    category=category,
                    message=message,
                    paths=missing_or_unusable,
                    owner=owner,
                )
            )
    return issues


def _consent_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    missing = [path for path in CONSENT_PATHS if _field_status(case, path) == FieldStatus.NOT_ASKED]
    not_agreed = [path for path in CONSENT_PATHS if _field_status(case, path) != FieldStatus.NOT_ASKED and _field_value(case, path) is not True]

    issues: list[ReadinessIssue] = []
    if missing:
        issues.append(
            _issue(
                code="missing_required_consent",
                severity=IssueSeverity.BLOCKER,
                category="consent",
                message="개인정보/개인신용정보 필수 동의 4개가 아직 확인되지 않았습니다.",
                paths=missing,
                owner=ResolutionOwner.USER,
            )
        )
    if not_agreed:
        issues.append(
            _issue(
                code="required_consent_not_agreed",
                severity=IssueSeverity.BLOCKER,
                category="consent",
                message="필수 개인정보/개인신용정보 동의 중 동의하지 않았거나 미확인으로 표시된 항목이 있습니다.",
                paths=not_agreed,
                owner=ResolutionOwner.USER,
            )
        )
    return issues


def _exclusion_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    not_confirmed = [path for path in EXCLUSION_PATHS if _field_status(case, path) == FieldStatus.NOT_ASKED]
    applies = [path for path in EXCLUSION_PATHS if _field_value(case, path) is True]

    issues: list[ReadinessIssue] = []
    if not_confirmed:
        issues.append(
            _issue(
                code="exclusion_screening_not_confirmed",
                severity=IssueSeverity.BLOCKER,
                category="exclusion",
                message="피해 신고 제외대상 체크리스트가 아직 모두 확인되지 않았습니다.",
                paths=not_confirmed,
                owner=ResolutionOwner.USER,
            )
        )
    if applies:
        issues.append(
            _issue(
                code="possible_exclusion_applies",
                severity=IssueSeverity.BLOCKER,
                category="exclusion",
                message="피해 신고 제외대상에 해당할 가능성이 있어 직원 확인이 필요합니다.",
                paths=applies,
                owner=ResolutionOwner.STAFF,
            )
        )
    return issues


def _evidence_confirmation_issues(case: RecoveryCase, *, strict_document_completion: bool) -> list[ReadinessIssue]:
    if not strict_document_completion:
        return []

    not_confirmed = [kind for kind in OFFICIAL_EVIDENCE_KINDS if _evidence_status(case, kind) == EvidenceStatus.NOT_ASKED]
    if not not_confirmed:
        return []

    return [
        ReadinessIssue(
            code="evidence_rows_not_confirmed",
            severity=IssueSeverity.BLOCKER,
            category="evidence",
            message="공식 첨부서류 13개 항목의 보유/미보유/추후제출/해당없음 상태가 아직 모두 확인되지 않았습니다.",
            paths=[f"evidence.{kind}.status" for kind in not_confirmed],
            labels=[EVIDENCE_ITEM_LABELS[kind] for kind in not_confirmed],
            resolution_owner=ResolutionOwner.USER,
        )
    ]


def _conditional_review_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    fraud_type = _field_value(case, "incident.fraud_type")
    if not isinstance(fraud_type, FraudType) or fraud_type not in FRAUD_TYPE_REVIEW_EVIDENCE:
        return []

    evidence_kind, message = FRAUD_TYPE_REVIEW_EVIDENCE[fraud_type]
    status = _evidence_status(case, evidence_kind)
    if status == EvidenceStatus.AVAILABLE:
        return []

    note = _evidence_note(case, evidence_kind)
    suffix = f" 현재 메모: {note}" if note else ""
    return [
        ReadinessIssue(
            code="conditional_evidence_needs_staff_review",
            severity=IssueSeverity.REVIEW,
            category="evidence",
            message=f"{message}{suffix}",
            paths=[f"evidence.{evidence_kind}.status"],
            labels=[EVIDENCE_ITEM_LABELS.get(evidence_kind, evidence_kind)],
            resolution_owner=ResolutionOwner.STAFF,
        )
    ]


def _report_status_review_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    issues: list[ReadinessIssue] = []

    investigation_status = _field_value(case, "investigation.status")
    if investigation_status in {ReportStatus.NOT_REPORTED, ReportStatus.PLANNED, ReportStatus.UNKNOWN}:
        issues.append(
            _issue(
                code="investigation_status_needs_review",
                severity=IssueSeverity.REVIEW,
                category="investigation",
                message="경찰/수사기관 신고가 미신고·예정·미확인 상태입니다. 접수 가능 여부와 보완 안내를 직원이 확인해야 합니다.",
                paths=["investigation.status"],
                owner=ResolutionOwner.STAFF,
            )
        )

    relief_status = _field_value(case, "relief.status")
    if relief_status in {ReportStatus.NOT_REPORTED, ReportStatus.PLANNED, ReportStatus.UNKNOWN}:
        issues.append(
            _issue(
                code="relief_status_needs_review",
                severity=IssueSeverity.REVIEW,
                category="relief",
                message="피해구제 신청 상태가 미신청·예정·미확인입니다. 지급정지/피해구제 절차 안내가 필요합니다.",
                paths=["relief.status"],
                owner=ResolutionOwner.STAFF,
            )
        )

    return issues


def _delegation_issues(case: RecoveryCase) -> list[ReadinessIssue]:
    proxy_used = _field_value(case, "delegation.proxy_used")
    if proxy_used is not True:
        return []

    paths = (
        "delegation.agent_name",
        "delegation.agent_birth_date",
        "delegation.agent_mobile_number",
        "delegation.agent_address",
        "delegation.request_purpose",
    )
    missing = [path for path in paths if not _has_answer(case, path)]
    if not missing:
        return []

    return [
        _issue(
            code="missing_delegation_required_fields",
            severity=IssueSeverity.BLOCKER,
            category="delegation",
            message="대리 신청으로 표시되어 있으나 위임장 필수 대리인 정보가 부족합니다.",
            paths=missing,
            owner=ResolutionOwner.USER,
        )
    ]


# ---------------------------------------------------------------------------
# Completion and path helpers
# ---------------------------------------------------------------------------


def _actionable_field_paths(case: RecoveryCase) -> set[str]:
    """Return paths that should be confirmed for the current case.

    Some HTML sections are conditional. For example, if the customer confirmed
    they are applying directly, blank delegation-agent fields should not block
    readiness; they are not applicable to this case.
    """

    paths = set(collect_field_value_paths(case)) & set(FIELD_LABELS)

    # If no proxy is used, agent details are not action-required.
    if _field_value(case, "delegation.proxy_used") is False:
        paths = {path for path in paths if not path.startswith("delegation.agent_") and path != "delegation.request_purpose"}

    # Optional dates are required only when the related report exists.
    if _field_value(case, "optional_reports.id_loss_reported") is False:
        paths.discard("optional_reports.id_loss_reported_date")
    if _field_value(case, "optional_reports.phone_loss_reported") is False:
        paths.discard("optional_reports.phone_loss_reported_date")
    if _field_value(case, "optional_reports.identity_theft_phone_reported") is False:
        paths.discard("optional_reports.identity_theft_phone_reported_date")

    # Usage frequency is meaningful only when the user said they use that channel.
    if _field_value(case, "survey.internet_banking_used") is False:
        paths.discard("survey.internet_banking_frequency")
    if _field_value(case, "survey.phone_banking_used") is False:
        paths.discard("survey.phone_banking_frequency")
    if _field_value(case, "survey.open_banking_used") is False:
        paths.discard("survey.open_banking_frequency")

    # Other-detail fields are required only when the associated flag is true.
    if _field_value(case, "survey.provided_other_financial_info") is not True:
        paths.discard("survey.provided_other_financial_info_text")

    # Multi-bank relief rows 2 and 3 are optional unless already touched.
    for path in ("relief.bank2", "relief.date2", "relief.bank3", "relief.date3"):
        field = _get_field(case, path)
        if isinstance(field, FieldValue) and field.status == FieldStatus.NOT_ASKED:
            paths.discard(path)

    return paths


def _completion_stats(case: RecoveryCase, paths: Iterable[str]) -> dict[str, float | int]:
    path_list = sorted(set(paths))
    total = len(path_list) + len(OFFICIAL_EVIDENCE_KINDS)
    confirmed = sum(1 for path in path_list if _field_status(case, path) != FieldStatus.NOT_ASKED)
    confirmed += sum(1 for kind in OFFICIAL_EVIDENCE_KINDS if _evidence_status(case, kind) != EvidenceStatus.NOT_ASKED)
    rate = 1.0 if total == 0 else round(confirmed / total, 4)
    return {"answered": confirmed, "total": total, "rate": rate}


def _field_status(case: RecoveryCase, path: str) -> FieldStatus | None:
    field = _get_field(case, path)
    return field.status if isinstance(field, FieldValue) else None


def _field_value(case: RecoveryCase, path: str) -> Any:
    field = _get_field(case, path)
    if isinstance(field, FieldValue) and field.status == FieldStatus.ANSWERED:
        return field.value
    return None


def _has_answer(case: RecoveryCase, path: str) -> bool:
    field = _get_field(case, path)
    return isinstance(field, FieldValue) and field.status == FieldStatus.ANSWERED and field.value is not None


def _get_field(root: Any, path: str) -> FieldValue[Any] | None:
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


def _evidence_by_kind(case: RecoveryCase) -> dict[str, Any]:
    return {item.kind: item for item in case.evidence}


def _evidence_status(case: RecoveryCase, kind: str) -> EvidenceStatus:
    item = _evidence_by_kind(case).get(kind)
    return item.status if item is not None else EvidenceStatus.NOT_ASKED


def _evidence_note(case: RecoveryCase, kind: str) -> str | None:
    item = _evidence_by_kind(case).get(kind)
    return item.note if item is not None else None


def _issue(
    *,
    code: str,
    severity: IssueSeverity,
    category: str,
    message: str,
    paths: Iterable[str],
    owner: ResolutionOwner,
) -> ReadinessIssue:
    path_list = list(paths)
    return ReadinessIssue(
        code=code,
        severity=severity,
        category=category,
        message=message,
        paths=path_list,
        labels=[_label_for_path(path) for path in path_list],
        resolution_owner=owner,
    )


def _label_for_path(path: str) -> str:
    if path.startswith("evidence."):
        parts = path.split(".")
        if len(parts) >= 2:
            return EVIDENCE_ITEM_LABELS.get(parts[1], path)
    return FIELD_LABELS.get(path, path)


def _status_from_issues(issues: list[ReadinessIssue]) -> ReadinessStatus:
    if any(issue.severity == IssueSeverity.BLOCKER for issue in issues):
        return ReadinessStatus.NOT_READY
    if any(issue.severity == IssueSeverity.REVIEW for issue in issues):
        return ReadinessStatus.NEEDS_REVIEW
    return ReadinessStatus.READY
