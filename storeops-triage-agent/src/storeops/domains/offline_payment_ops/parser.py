"""Offline payment parser implementation."""

from __future__ import annotations

from storeops.core.types import ParsedCase


class OfflinePaymentCaseParser:
    """Deterministic parser for offline payment and terminal incidents."""

    def parse(
        self,
        merchant_message: str,
        *,
        store_id: str,
        case_hint: str | None = None,
    ) -> ParsedCase:
        text = merchant_message.lower()
        hint = (case_hint or "").lower()
        joined = f"{text} {hint}"
        symptoms: list[str] = []
        context_flags: list[str] = []
        missing_fields: list[str] = []

        if self._contains_any(
            joined,
            [
                "결제",
                "승인",
                "신용승인",
                "카드",
                "payment",
                "approval",
                "card",
                "declined",
                "decline",
            ],
        ):
            symptoms.append("approval_failure")

        if self._contains_any(
            joined,
            [
                "pos",
                "front",
                "프론트",
                "요청",
                "전달",
                "연동",
                "통신",
                "timeout",
                "network",
                "pairing",
            ],
        ):
            symptoms.append("pos_request_failure")
            context_flags.append("pos_front_context")

        if self._contains_any(
            joined,
            [
                "새 단말기",
                "신규 단말기",
                "기존 단말기",
                "단말기 추가",
                "단말기 설치",
                "설치",
                "교체",
                "new terminal",
                "existing terminal",
                "installed",
                "installation",
                "replacement",
                "duplicate tid",
            ],
        ):
            context_flags.append("new_terminal_recently_installed")

        if self._contains_any(
            joined,
            [
                "가맹점",
                "가맹점 번호",
                "미등록",
                "등록",
                "van",
                "merchant registration",
                "merchant number",
                "registration",
            ],
        ):
            context_flags.append("merchant_registration_context")

        if self._contains_any(
            joined,
            [
                "단말기 번호",
                "기기 번호",
                "식별",
                "시리얼",
                "device",
                "terminal identifier",
                "serial",
                "device number",
            ],
        ):
            context_flags.append("terminal_identity_context")

        if self._contains_any(
            joined,
            [
                "incident-time",
                "current records",
                "records disagree",
                "temporal",
                "historical",
                "현재",
                "당시",
                "이력",
            ],
        ):
            context_flags.append("temporal_conflict_context")

        if "approval_failure" in symptoms:
            if not self._contains_any(
                joined,
                [
                    "term-",
                    "terminal_id",
                    "terminal identifier",
                    "device number",
                    "serial",
                    "단말기 번호",
                    "기기 번호",
                    "시리얼",
                    "기존 단말기",
                    "신규 단말기",
                    "새 단말기",
                ],
            ):
                missing_fields.append("failed_physical_terminal")
            if not self._contains_any(
                joined,
                [
                    "오류",
                    "에러",
                    "오류 문구",
                    "응답 코드",
                    "미등록",
                    "시간 초과",
                    "error",
                    "timeout",
                    "response",
                    "code",
                    "message",
                ],
            ):
                missing_fields.append("visible_error_message")

        issue_family = "payment_approval_failure"
        if "pos_request_failure" in symptoms:
            issue_family = "pos_front_connection_issue"

        expanded = " ".join(
            [
                merchant_message,
                case_hint or "",
                issue_family,
                " ".join(symptoms),
                " ".join(context_flags),
            ]
        )
        return ParsedCase(
            store_id=store_id,
            merchant_message=merchant_message,
            issue_family=issue_family,
            symptoms=symptoms or ["unknown_payment_problem"],
            context_flags=context_flags,
            missing_fields=missing_fields,
            retrieval_query=expanded,
            planner_query=expanded,
        )

    @staticmethod
    def _contains_any(value: str, tokens: list[str]) -> bool:
        return any(token in value for token in tokens)


__all__ = ["OfflinePaymentCaseParser", "ParsedCase"]