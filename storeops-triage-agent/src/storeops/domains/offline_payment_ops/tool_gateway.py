"""Offline payment read-only tool gateway for synthetic fixture databases."""

from __future__ import annotations

import sqlite3

from storeops.core.contracts import ToolError, ToolResponse, ToolStatus
from storeops.infra.tools import ToolGateway


class OfflinePaymentToolGateway(ToolGateway):
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        scenario_id: str,
        operator_id: str,
        trace_id: str,
    ) -> None:
        super().__init__(connection, operator_id=operator_id, trace_id=trace_id)
        self.scenario_id = scenario_id

    def _failure(self, tool_name: str, store_id: str) -> ToolResponse | None:
        row = self.connection.execute(
            """
            SELECT * FROM tool_failure_injections
            WHERE scenario_id = ? AND tool_name = ?
            """,
            (self.scenario_id, tool_name),
        ).fetchone()
        if row is None:
            return None
        error_type = {
            "timeout": "ToolTimeoutError",
            "unavailable": "ToolUnavailableError",
        }.get(row["failure_mode"], "ToolUnavailableError")
        return ToolResponse(
            tool_name=tool_name,
            trace_id=self.trace_id,
            store_id=store_id,
            status=ToolStatus.ERROR,
            error=ToolError(
                error_type=error_type,
                message=row["error_message"],
            ),
        )

    def get_tid_config(self, store_id: str) -> ToolResponse:
        failure = self._failure("get_tid_config", store_id)
        return failure or super().get_tid_config(store_id)

    def get_support_route(self, store_id: str, issue_type: str) -> ToolResponse:
        failure = self._failure("get_support_route", store_id)
        return failure or super().get_support_route(store_id, issue_type)

    def get_terminal_identity(self, store_id: str) -> ToolResponse:
        return self._query(
            tool_name="get_terminal_identity",
            store_id=store_id,
            sql="""
                SELECT t.terminal_id, t.device_number, t.physical_serial,
                       i.identity_record_id, i.registered_device_number,
                       i.registered_serial, i.observed_at, i.recorded_at, i.available_at
                FROM terminals t
                JOIN terminal_identities i ON i.terminal_id = t.terminal_id
                WHERE t.store_id = ? AND i.valid_to IS NULL
                ORDER BY t.terminal_id
            """,
            params=(store_id,),
        )

    def get_installation_history(self, store_id: str) -> ToolResponse:
        return self._query(
            tool_name="get_installation_history",
            store_id=store_id,
            sql="""
                SELECT * FROM installation_events
                WHERE store_id = ? ORDER BY observed_at
            """,
            params=(store_id,),
        )

    def get_van_registration(self, store_id: str) -> ToolResponse:
        return self._query(
            tool_name="get_van_registration",
            store_id=store_id,
            sql="""
                SELECT * FROM van_registrations
                WHERE store_id = ? AND valid_to IS NULL
                ORDER BY observed_at
            """,
            params=(store_id,),
        )

    def get_pos_front_connection_logs(self, store_id: str) -> ToolResponse:
        return self._query(
            tool_name="get_pos_front_connection_logs",
            store_id=store_id,
            sql="""
                SELECT 'snapshot' AS record_type, link_id AS record_id,
                       pairing_status AS status, updated_at AS observed_at
                FROM pos_front_links WHERE store_id = ?
                UNION ALL
                SELECT 'event', connection_event_id, event_status, observed_at
                FROM pos_front_connection_events WHERE store_id = ?
                ORDER BY observed_at
            """,
            params=(store_id, store_id),
        )

    def get_tid_history(self, store_id: str) -> ToolResponse:
        return self._query(
            tool_name="get_tid_history",
            store_id=store_id,
            sql="""
                SELECT * FROM tid_assignments
                WHERE store_id = ? ORDER BY valid_from
            """,
            params=(store_id,),
        )


__all__ = ["OfflinePaymentToolGateway"]
