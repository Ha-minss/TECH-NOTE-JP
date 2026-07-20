from __future__ import annotations

from collections.abc import Iterable


MERCHANT_OBSERVABLE_FIELDS = {
    "failed_physical_terminal",
    "visible_error_message",
    "error_time",
    "payment_method",
}

FORBIDDEN_RESPONSE_PHRASES = (
    "해결 완료",
    "정상 처리 완료",
    "이미 해결",
)


def ensure_confidence(output, *, minimum: float = 0.5) -> None:
    if getattr(output, "confidence", 0.0) < minimum:
        raise ValueError(f"confidence below threshold: {getattr(output, 'confidence', 0.0)}")


def ensure_issue_family(issue_family: str) -> None:
    if issue_family not in {"payment_approval_failure", "pos_front_connection_issue"}:
        raise ValueError(f"unsupported issue_family: {issue_family}")


def ensure_allowed_data_needs(selected_data_needs, *, allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    invalid = [item.name for item in selected_data_needs if item.name not in allowed_set]
    if invalid:
        raise ValueError(f"invalid data_needs: {', '.join(invalid)}")
    if not selected_data_needs:
        raise ValueError("no selected data needs")


def filter_clarification_questions(questions, *, known_missing_fields: Iterable[str]):
    allowed_fields = set(known_missing_fields) or MERCHANT_OBSERVABLE_FIELDS
    filtered = []
    seen_fields: set[str] = set()
    for question in questions:
        if question.field not in MERCHANT_OBSERVABLE_FIELDS:
            continue
        if question.field not in allowed_fields:
            continue
        if question.field in seen_fields:
            continue
        seen_fields.add(question.field)
        filtered.append(question)
        if len(filtered) == 2:
            break
    return filtered


def ensure_safe_merchant_response(output, *, state: str) -> None:
    if output.contains_unconfirmed_claim:
        raise ValueError("response contains unconfirmed claim")
    if any(phrase in output.merchant_response for phrase in FORBIDDEN_RESPONSE_PHRASES):
        raise ValueError("response contains forbidden completion language")
    if state != "READY_FOR_REVIEW" and not output.mentions_uncertainty:
        raise ValueError("uncertain state requires uncertainty copy")


__all__ = [
    "MERCHANT_OBSERVABLE_FIELDS",
    "ensure_allowed_data_needs",
    "ensure_confidence",
    "ensure_issue_family",
    "ensure_safe_merchant_response",
    "filter_clarification_questions",
]
