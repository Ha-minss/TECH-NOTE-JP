from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable


class DataNeedPriority(StrEnum):
    REQUIRED = "required"
    SUPPORTING = "supporting"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class DataNeed:
    name: str
    priority: DataNeedPriority
    reason: str


@dataclass(frozen=True)
class PlannedToolCall:
    tool_name: str
    data_need: str
    reason: str
    required: bool


@dataclass(frozen=True)
class PlannerOutput:
    case_type: str
    data_needs: list[DataNeed | dict]
    planned_tool_calls: list[PlannedToolCall | dict]
    clarification_candidates: list[str]
    forbidden_actions: list[str]
    retrieved_policy_ids: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_needs", [self._coerce_data_need(item) for item in self.data_needs])
        object.__setattr__(
            self,
            "planned_tool_calls",
            [self._coerce_tool_call(item) for item in self.planned_tool_calls],
        )

    @staticmethod
    def _coerce_data_need(item: DataNeed | dict) -> DataNeed:
        if isinstance(item, DataNeed):
            return item
        return DataNeed(
            name=item["name"],
            priority=DataNeedPriority(item["priority"]),
            reason=item["reason"],
        )

    @staticmethod
    def _coerce_tool_call(item: PlannedToolCall | dict) -> PlannedToolCall:
        if isinstance(item, PlannedToolCall):
            return item
        return PlannedToolCall(
            tool_name=item["tool_name"],
            data_need=item["data_need"],
            reason=item["reason"],
            required=bool(item["required"]),
        )


@dataclass(frozen=True)
class ToolSpec:
    tool_name: str
    description: str
    provides_data_needs: list[str]
    input_schema: dict[str, str]
    read_only: bool
    stage: str


class ToolCatalog:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = tools
        self._by_need: dict[str, ToolSpec] = {}
        for tool in tools:
            for data_need in tool.provides_data_needs:
                self._by_need[data_need] = tool

    @classmethod
    def load(cls, path: Path | str) -> "ToolCatalog":
        raw_tools = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([ToolSpec(**raw_tool) for raw_tool in raw_tools])

    def tool_for_data_need(self, data_need: str) -> ToolSpec:
        try:
            return self._by_need[data_need]
        except KeyError as exc:
            raise KeyError(f"No tool provides data need: {data_need}") from exc


@dataclass(frozen=True)
class _Rule:
    case_type: str
    patterns: tuple[str, ...]
    data_needs: tuple[tuple[str, DataNeedPriority, str], ...]
    clarifications: tuple[str, ...] = ()


class Planner:
    """Rule-backed deterministic planner."""

    def __init__(self, tool_catalog: ToolCatalog):
        self.tool_catalog = tool_catalog
        self.rules = self._rules()

    def plan(
        self,
        *,
        query: str,
        retrieved_policies: Iterable[object],
        parsed_case=None,
    ) -> PlannerOutput:
        del parsed_case
        policy_ids = [getattr(policy, "document_id") for policy in retrieved_policies]
        rule = self._select_rule(query, policy_ids)
        data_needs = [
            DataNeed(name=name, priority=priority, reason=reason)
            for name, priority, reason in rule.data_needs
        ]
        planned_tool_calls = [
            PlannedToolCall(
                tool_name=self.tool_catalog.tool_for_data_need(data_need.name).tool_name,
                data_need=data_need.name,
                reason=data_need.reason,
                required=data_need.priority == DataNeedPriority.REQUIRED,
            )
            for data_need in data_needs
            if self.tool_catalog.tool_for_data_need(data_need.name).stage != "post_assessment"
        ]
        return PlannerOutput(
            case_type=rule.case_type,
            data_needs=data_needs,
            planned_tool_calls=planned_tool_calls,
            clarification_candidates=list(rule.clarifications),
            forbidden_actions=[
                "payment_execution",
                "refund",
                "payment_cancellation",
                "config_mutation",
                "external_handoff_without_approval",
            ],
            retrieved_policy_ids=policy_ids,
        )

    def _select_rule(self, query: str, policy_ids: list[str]) -> _Rule:
        normalized = self._normalize(query)

        if any(
            pattern in normalized
            for pattern in (
                "pos",
                "front",
                "프론트",
                "금액이안넘어",
                "결제요청",
                "전달",
                "requestdelivery",
                "timeout",
            )
        ):
            return self._rule_by_case_type("pos_front_connection_issue")

        if any(
            pattern in normalized
            for pattern in (
                "가맹점번호",
                "미등록",
                "등록대기",
                "van",
                "merchantregistration",
                "merchantnumber",
                "registration",
            )
        ):
            return self._rule_by_case_type("merchant_registration_check")

        if any(
            pattern in normalized
            for pattern in (
                "단말기번호",
                "시리얼",
                "기기번호",
                "식별",
                "맞는지확인",
                "terminalidentifier",
                "identity",
                "serial",
                "devicenumber",
            )
        ):
            return self._rule_by_case_type("terminal_identity_check")

        if any(
            pattern in normalized
            for pattern in (
                "새단말기",
                "신규단말기",
                "설치했",
                "교체",
                "바꾼뒤부터",
                "duplicatetid",
                "newterminal",
                "terminalinstallation",
                "replacement",
                "historicaltid",
                "temporalconflict",
            )
        ):
            return self._rule_by_case_type("terminal_installation_payment_failure")

        if any(
            pattern in normalized
            for pattern in (
                "승인오류",
                "결제가안돼",
                "문구",
                "기억",
                "모르",
                "어느기기",
                "여러개",
                "정상처럼",
                "현장에서는",
                "ambiguouspayment",
                "missingvisibleerror",
                "failedphysicalterminal",
            )
        ):
            return self._rule_by_case_type("ambiguous_payment_failure")

        for rule in self.rules:
            if any(pattern in normalized for pattern in rule.patterns):
                return rule
        if "SOP-PAY-OP-005" in policy_ids:
            return self._rule_by_case_type("ambiguous_payment_failure")
        if "SOP-PAY-OP-004" in policy_ids:
            return self._rule_by_case_type("pos_front_connection_issue")
        if "SOP-PAY-OP-003" in policy_ids:
            return self._rule_by_case_type("merchant_registration_check")
        if "SOP-PAY-OP-002" in policy_ids:
            return self._rule_by_case_type("terminal_installation_payment_failure")
        if "SOP-PAY-OP-001" in policy_ids:
            return self._rule_by_case_type("ambiguous_payment_failure")
        return self._rule_by_case_type("ambiguous_payment_failure")


    def _rule_by_case_type(self, case_type: str) -> _Rule:
        return next(rule for rule in self.rules if rule.case_type == case_type)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    @staticmethod
    def _rules() -> list[_Rule]:
        required = DataNeedPriority.REQUIRED
        supporting = DataNeedPriority.SUPPORTING
        return [
            _Rule(
                case_type="pos_front_connection_issue",
                patterns=("pos", "front", "requestdelivery", "timeout"),
                data_needs=(
                    ("pos_front_connection_history", required, "Check POS/front connection health."),
                    ("request_delivery_history", required, "Check payment request delivery failures."),
                ),
            ),
            _Rule(
                case_type="merchant_registration_check",
                patterns=("merchantnumber", "registration", "van"),
                data_needs=(
                    ("merchant_registration_status", required, "Check merchant registration status."),
                    ("approval_failure_history", required, "Check approval failure history."),
                    ("payment_identifier_config", required, "Check payment identifier config."),
                    ("terminal_identity_record", required, "Check terminal identity records."),
                ),
            ),
            _Rule(
                case_type="terminal_identity_check",
                patterns=("terminalidentifier", "identity", "serial", "devicenumber"),
                data_needs=(
                    ("terminal_inventory", required, "Check terminal inventory."),
                    ("terminal_identity_record", required, "Cross-check registered identity data."),
                    ("payment_identifier_config", required, "Check payment identifier config."),
                    ("approval_failure_history", required, "Check failure history on the affected terminal."),
                ),
            ),
            _Rule(
                case_type="terminal_installation_payment_failure",
                patterns=("newterminal", "installation", "replacement", "duplicatetid"),
                data_needs=(
                    ("terminal_inventory", required, "Separate old and newly installed terminals."),
                    ("payment_identifier_config", required, "Check payment identifier config."),
                    ("activation_timeline", required, "Compare activation timing against failure timing."),
                    ("approval_failure_history", required, "Check which terminal the failures were tied to."),
                ),
            ),
            _Rule(
                case_type="ambiguous_payment_failure",
                patterns=("approvalerror", "cardpayment", "genericfailure"),
                data_needs=(
                    ("approval_failure_history", required, "Check whether failures are recorded."),
                    ("terminal_inventory", required, "Check current terminal inventory."),
                    ("recent_change_summary", supporting, "Look for recent installation or config changes."),
                ),
                clarifications=("failed_physical_terminal", "visible_error_message", "affected_device", "error_phrase"),
            ),
        ]


__all__ = [
    "DataNeed",
    "DataNeedPriority",
    "PlannedToolCall",
    "Planner",
    "PlannerOutput",
    "ToolCatalog",
    "ToolSpec",
]
