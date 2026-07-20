from __future__ import annotations

from dataclasses import dataclass

from storeops.core.policy_checks import PolicyCheck


@dataclass(frozen=True)
class _NeedRule:
    data_need: str
    priority: str
    indicators: tuple[str, ...]
    check_text: str


class OfflinePaymentPolicyCheckExtractor:
    """Extract SOP-backed evidence checks from retrieved policy documents."""

    RULES = (
        _NeedRule(
            data_need="terminal_inventory",
            priority="required",
            indicators=("전체 단말기", "단말기 목록", "terminal inventory", "registered terminal"),
            check_text="Check the store's registered terminal inventory.",
        ),
        _NeedRule(
            data_need="payment_identifier_config",
            priority="required",
            indicators=("결제 식별", "식별 설정", "TID", "payment identifier", "tid config"),
            check_text="Check terminal payment identifier settings.",
        ),
        _NeedRule(
            data_need="historical_identifier_config",
            priority="required",
            indicators=("과거 결제 식별", "incident-time", "historical", "현재 또는 과거"),
            check_text="Check historical payment identifier settings when timing may matter.",
        ),
        _NeedRule(
            data_need="activation_timeline",
            priority="required",
            indicators=("활성화", "개시", "activation", "first payment"),
            check_text="Check terminal activation or first verification timing.",
        ),
        _NeedRule(
            data_need="installation_history",
            priority="supporting",
            indicators=("설치", "교체", "설정 변경", "installation", "replacement", "recent change"),
            check_text="Check recent installation, replacement, or configuration change history.",
        ),
        _NeedRule(
            data_need="merchant_registration_status",
            priority="required",
            indicators=("가맹점", "등록 상태", "merchant registration", "merchant number"),
            check_text="Check merchant registration status.",
        ),
        _NeedRule(
            data_need="van_registration_status",
            priority="required",
            indicators=("VAN", "van registration", "결제망"),
            check_text="Check VAN registration status.",
        ),
        _NeedRule(
            data_need="approval_failure_history",
            priority="required",
            indicators=("승인 실패", "승인 오류", "오류 문구", "approval failure", "response code"),
            check_text="Check recent approval failure history and response codes.",
        ),
        _NeedRule(
            data_need="terminal_identity_record",
            priority="required",
            indicators=("단말기 식별", "단말기 번호", "serial", "device number", "terminal identity"),
            check_text="Check terminal identity records.",
        ),
        _NeedRule(
            data_need="pos_front_connection_history",
            priority="required",
            indicators=("POS", "Front", "front connection", "연결", "pairing"),
            check_text="Check POS/front connection history.",
        ),
        _NeedRule(
            data_need="request_delivery_history",
            priority="required",
            indicators=("요청 전달", "request delivery", "delivery failure", "전달 실패"),
            check_text="Check payment request delivery history.",
        ),
        _NeedRule(
            data_need="recent_change_summary",
            priority="supporting",
            indicators=("최근", "recent", "change summary"),
            check_text="Check recent operational changes.",
        ),
    )

    def extract(self, *, parsed_case, query: str, retrieved_policies, tool_catalog) -> list[PolicyCheck]:
        del parsed_case, query
        allowed = set(tool_catalog._by_need.keys())
        checks: list[PolicyCheck] = []
        seen: set[tuple[str, str]] = set()
        for policy in retrieved_policies:
            content = getattr(policy, "content", "")
            normalized = content.lower()
            for rule in self.RULES:
                if rule.data_need not in allowed:
                    continue
                quote = self._source_quote(content, normalized, rule.indicators)
                if quote is None:
                    continue
                key = (getattr(policy, "document_id"), rule.data_need)
                if key in seen:
                    continue
                seen.add(key)
                checks.append(
                    PolicyCheck(
                        policy_id=getattr(policy, "document_id"),
                        policy_title=getattr(policy, "title", None),
                        check_text=rule.check_text,
                        matched_data_need=rule.data_need,
                        priority=rule.priority,  # type: ignore[arg-type]
                        reason=(
                            f"Retrieved SOP says to verify '{rule.check_text}', "
                            f"which maps to allowed data_need '{rule.data_need}' in the tool catalog."
                        ),
                        source_quote=quote,
                    )
                )
        return checks

    @staticmethod
    def _source_quote(content: str, normalized: str, indicators: tuple[str, ...]) -> str | None:
        for indicator in indicators:
            if indicator.lower() not in normalized:
                continue
            for line in content.splitlines():
                stripped = line.strip(" -\t")
                if indicator.lower() in stripped.lower():
                    return stripped[:220]
            index = normalized.find(indicator.lower())
            if index >= 0:
                return content[max(0, index - 60): index + 160].strip().replace("\n", " ")
        return None


__all__ = ["OfflinePaymentPolicyCheckExtractor"]
