"""Offline payment evidence normalization rules."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from storeops.core.contracts import EvidenceRecord, ToolResponse, ToolStatus


class OfflinePaymentEvidenceBuilder:
    """Builds terminal/payment-specific EvidenceRecord objects."""

    def build(
        self,
        *,
        scenario_id: str,
        tool_responses: Iterable[ToolResponse],
    ) -> list[EvidenceRecord]:
        responses = list(tool_responses)
        evidence: list[EvidenceRecord] = []
        evidence.extend(self._duplicate_tid_evidence(scenario_id, responses))
        evidence.extend(self._temporal_tid_conflict_evidence(scenario_id, responses))
        evidence.extend(self._terminal_identity_evidence(scenario_id, responses))
        evidence.extend(self._van_registration_evidence(scenario_id, responses))
        evidence.extend(self._pos_front_evidence(scenario_id, responses))
        return evidence

    def _duplicate_tid_evidence(
        self,
        scenario_id: str,
        responses: list[ToolResponse],
    ) -> list[EvidenceRecord]:
        response = self._success_response(responses, "get_tid_config")
        if response is None:
            return []
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in response.data:
            groups.setdefault(str(row["tid"]), []).append(row)
        duplicate = next((rows for rows in groups.values() if len(rows) > 1), None)
        if duplicate is None:
            return []
        return [
            self._record(
                scenario_id,
                response.tool_name,
                ",".join(str(row["tid_assignment_id"]) for row in duplicate),
                "duplicate_tid_assignment",
                {
                    "tid": duplicate[0]["tid"],
                    "terminal_ids": [row["terminal_id"] for row in duplicate],
                },
                duplicate[-1]["observed_at"],
                supports=["duplicate_tid"],
            )
        ]

    def _temporal_tid_conflict_evidence(
        self,
        scenario_id: str,
        responses: list[ToolResponse],
    ) -> list[EvidenceRecord]:
        current = self._success_response(responses, "get_tid_config")
        history = self._success_response(responses, "get_tid_history")
        errors = self._success_response(responses, "get_recent_approval_errors")
        if current is None or history is None or errors is None or not errors.data:
            return []

        incident_time = datetime.fromisoformat(str(errors.data[-1]["observed_at"]))
        incident_rows = [
            row
            for row in history.data
            if datetime.fromisoformat(str(row["valid_from"])) <= incident_time
            and (
                row["valid_to"] is None
                or incident_time < datetime.fromisoformat(str(row["valid_to"]))
            )
        ]
        incident_tids = [row["tid"] for row in incident_rows]
        current_tids = [row["tid"] for row in current.data]
        duplicate_at_incident = len(set(incident_tids)) < len(incident_tids)
        current_distinct = len(set(current_tids)) == len(current_tids)
        if not (duplicate_at_incident and current_distinct):
            return []

        return [
            self._record(
                scenario_id,
                history.tool_name,
                ",".join(str(row["tid_assignment_id"]) for row in incident_rows),
                "incident_time_tid_configuration",
                {"duplicate_at_incident": True, "incident_time": incident_time.isoformat()},
                incident_rows[-1]["observed_at"],
                supports=["temporary_duplicate_tid"],
            ),
            self._record(
                scenario_id,
                current.tool_name,
                ",".join(str(row["tid_assignment_id"]) for row in current.data),
                "current_tid_configuration",
                {"current_distinct": True},
                current.data[-1]["observed_at"],
                contradicts=["temporary_duplicate_tid"],
            ),
        ]

    def _terminal_identity_evidence(
        self,
        scenario_id: str,
        responses: list[ToolResponse],
    ) -> list[EvidenceRecord]:
        response = self._success_response(responses, "get_terminal_identity")
        if response is None:
            return []
        mismatches = [
            row
            for row in response.data
            if row.get("device_number") != row.get("registered_device_number")
            or row.get("physical_serial") != row.get("registered_serial")
        ]
        if not mismatches:
            return []
        row = mismatches[0]
        return [
            self._record(
                scenario_id,
                response.tool_name,
                str(row["identity_record_id"]),
                "terminal_identity_mismatch",
                {
                    "terminal_id": row["terminal_id"],
                    "device_number_matches": False,
                    "serial_matches": False,
                },
                row["observed_at"],
                supports=["terminal_identifier_mismatch"],
            )
        ]

    def _van_registration_evidence(
        self,
        scenario_id: str,
        responses: list[ToolResponse],
    ) -> list[EvidenceRecord]:
        response = self._success_response(responses, "get_van_registration")
        if response is None:
            return []
        incomplete = [row for row in response.data if row.get("registration_status") != "active"]
        if not incomplete:
            return []
        row = incomplete[0]
        return [
            self._record(
                scenario_id,
                response.tool_name,
                str(row["van_registration_id"]),
                "van_registration_incomplete",
                {"status": row["registration_status"]},
                row["observed_at"],
                supports=["van_merchant_registration_missing"],
            )
        ]

    def _pos_front_evidence(
        self,
        scenario_id: str,
        responses: list[ToolResponse],
    ) -> list[EvidenceRecord]:
        response = self._success_response(responses, "get_pos_front_connection_logs")
        if response is None:
            return []
        abnormal = [
            row
            for row in response.data
            if row.get("status") in {"disconnected", "failed", "timeout", "mismatch"}
        ]
        if not abnormal:
            return []
        return [
            self._record(
                scenario_id,
                response.tool_name,
                ",".join(str(row["record_id"]) for row in abnormal),
                "pos_front_request_delivery_failure",
                {"statuses": [row["status"] for row in abnormal]},
                abnormal[0]["observed_at"],
                supports=["pos_front_connection_issue"],
            )
        ]

    @staticmethod
    def _success_response(
        responses: list[ToolResponse],
        tool_name: str,
    ) -> ToolResponse | None:
        return next(
            (
                response
                for response in responses
                if response.tool_name == tool_name and response.status is ToolStatus.SUCCESS
            ),
            None,
        )

    @staticmethod
    def _record(
        scenario_id: str,
        source_tool: str,
        source_record_id: str,
        fact_type: str,
        value: object,
        observed_at: str,
        *,
        supports: list[str] | None = None,
        contradicts: list[str] | None = None,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=f"EV-{scenario_id}-{fact_type.upper()}",
            source_tool=source_tool,
            source_record_id=source_record_id,
            fact_type=fact_type,
            normalized_value=value,
            observed_at=datetime.fromisoformat(str(observed_at)),
            supports=supports or [],
            contradicts=contradicts or [],
        )


__all__ = ["OfflinePaymentEvidenceBuilder"]
