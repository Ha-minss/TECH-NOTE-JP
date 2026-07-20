from __future__ import annotations

import sqlite3
from typing import Callable, Protocol

from storeops.core.planner import PlannerOutput
from storeops.core.contracts import ToolResponse


class ToolGatewayFactory(Protocol):
    def __call__(
        self,
        connection: sqlite3.Connection,
        *,
        operator_id: str,
        trace_id: str,
        scenario_id: str | None = None,
    ): ...


class ToolExecutor:
    """Generic planner-to-tool executor for read-only workflow tools."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        operator_id: str,
        trace_id: str,
        scenario_id: str | None = None,
        gateway_factory: ToolGatewayFactory,
    ) -> None:
        self.gateway = self._build_gateway(
            gateway_factory,
            connection,
            operator_id=operator_id,
            trace_id=trace_id,
            scenario_id=scenario_id,
        )

    @staticmethod
    def _build_gateway(gateway_factory, connection, *, operator_id: str, trace_id: str, scenario_id: str | None):
        try:
            return gateway_factory(
                connection,
                operator_id=operator_id,
                trace_id=trace_id,
                scenario_id=scenario_id,
            )
        except TypeError:
            return gateway_factory(
                connection,
                operator_id=operator_id,
                trace_id=trace_id,
            )

    def execute(self, *, store_id: str, plan: PlannerOutput) -> list[ToolResponse]:
        responses: list[ToolResponse] = []
        seen: set[str] = set()
        for call in plan.planned_tool_calls:
            if call.tool_name in seen:
                continue
            seen.add(call.tool_name)
            responses.append(self._call(call.tool_name, store_id))

            if call.tool_name == "get_tid_config" and hasattr(self.gateway, "get_tid_history"):
                history_tool = "get_tid_history"
                if history_tool not in seen:
                    seen.add(history_tool)
                    responses.append(self._call(history_tool, store_id))
        return responses

    def _call(self, tool_name: str, store_id: str) -> ToolResponse:
        method = getattr(self.gateway, tool_name)
        return method(store_id)


__all__ = ["ToolExecutor", "ToolGatewayFactory"]
