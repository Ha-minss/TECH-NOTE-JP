"""Read-only tool gateway for deterministic fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime

from storeops.core.contracts import ToolError, ToolResponse, ToolStatus


class ToolGateway:
    def __init__(self, connection: sqlite3.Connection, *, operator_id: str, trace_id: str) -> None:
        self.connection = connection
        self.operator_id = operator_id
        self.trace_id = trace_id

    def _authorized(self, store_id: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM store_operator_access
            WHERE operator_id = ? AND store_id = ? AND active = 1
            """,
            (self.operator_id, store_id),
        ).fetchone()
        return row is not None

    def _query(self, *, tool_name: str, store_id: str, sql: str, params: Sequence[object]) -> ToolResponse:
        if not self._authorized(store_id):
            return ToolResponse(
                tool_name=tool_name,
                trace_id=self.trace_id,
                store_id=store_id,
                status=ToolStatus.ERROR,
                error=ToolError(
                    error_type='ToolAuthorizationError',
                    message='Operator is not authorized for this store.',
                ),
            )

        rows = [dict(row) for row in self.connection.execute(sql, params).fetchall()]
        response = ToolResponse(
            tool_name=tool_name,
            trace_id=self.trace_id,
            store_id=store_id,
            status=ToolStatus.SUCCESS if rows else ToolStatus.NOT_FOUND,
            data=rows,
            provenance=[f"{tool_name}:{row.get(next(iter(row)), 'unknown')}" for row in rows],
        )
        return self._apply_freshness(response)

    def _apply_freshness(self, response: ToolResponse) -> ToolResponse:
        if not response.data:
            return response
        delays = []
        for row in response.data:
            observed_at = row.get('observed_at') or row.get('recorded_at')
            available_at = row.get('available_at')
            if not observed_at or not available_at:
                continue
            observed = datetime.fromisoformat(str(observed_at))
            available = datetime.fromisoformat(str(available_at))
            delays.append(available - observed)
        if any(delay.total_seconds() >= 24 * 60 * 60 for delay in delays):
            response.freshness = 'delayed'
            response.warnings.append('Source data became available at least 24 hours after observation.')
        elif delays:
            response.freshness = 'current'
        else:
            response.freshness = 'unknown'
            response.warnings.append('Source rows do not expose freshness timestamps.')
        return response

    def get_store_info(self, store_id: str) -> ToolResponse:
        return self._query(tool_name='get_store_info', store_id=store_id, sql='SELECT * FROM stores WHERE store_id = ?', params=(store_id,))

    def get_terminals(self, store_id: str) -> ToolResponse:
        return self._query(tool_name='get_terminals', store_id=store_id, sql='SELECT * FROM terminals WHERE store_id = ? ORDER BY terminal_id', params=(store_id,))

    def get_tid_config(self, store_id: str) -> ToolResponse:
        return self._query(tool_name='get_tid_config', store_id=store_id, sql='SELECT * FROM tid_assignments WHERE store_id = ? AND valid_to IS NULL ORDER BY terminal_id', params=(store_id,))

    def get_activation_history(self, store_id: str) -> ToolResponse:
        return self._query(tool_name='get_activation_history', store_id=store_id, sql='SELECT * FROM activation_events WHERE store_id = ? ORDER BY observed_at', params=(store_id,))

    def get_recent_approval_errors(self, store_id: str) -> ToolResponse:
        return self._query(tool_name='get_recent_approval_errors', store_id=store_id, sql="SELECT * FROM approval_events WHERE store_id = ? AND event_result != 'approved' ORDER BY observed_at", params=(store_id,))

    def get_support_route(self, store_id: str, issue_type: str) -> ToolResponse:
        return self._query(tool_name='get_support_route', store_id=store_id, sql="SELECT * FROM support_routes WHERE store_id = ? AND issue_type = ? AND record_status = 'active'", params=(store_id, issue_type))


__all__ = ['ToolGateway']
