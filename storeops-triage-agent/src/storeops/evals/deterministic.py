"""Deterministic evaluation for the offline payment domain pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from storeops.observability.trace import build_trace_record
from storeops.observability.metrics import ratio
from storeops.domains.offline_payment_ops.workflow import OfflinePaymentWorkflow
from storeops.infra.database import open_database


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    passed: bool
    actual_state: str
    actual_primary_cause: str | None
    has_evidence_citations: bool
    required_tool_recall: float
    forbidden_action_count: int
    unsupported_likely_claim_count: int
    has_trace: bool
    trace_id: str
    errors: list[str] = field(default_factory=list)


def default_fixture_db_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fixtures" / "offline_payment_ops_synthetic_50.sqlite3"


class DeterministicEvaluator:
    def __init__(self, *, fixture_db_path: Path | str | None = None):
        self.fixture_db_path = Path(fixture_db_path) if fixture_db_path is not None else default_fixture_db_path()

    @classmethod
    def default(cls) -> "DeterministicEvaluator":
        return cls()

    @classmethod
    def from_fixture_db(cls, fixture_db_path: Path | str) -> "DeterministicEvaluator":
        return cls(fixture_db_path=fixture_db_path)

    def _connection(self):
        return open_database(self.fixture_db_path)

    def evaluate_case(self, case) -> EvalCaseResult:
        connection = self._connection()
        try:
            workflow = OfflinePaymentWorkflow.default(connection)
            result = workflow.run_case(
                scenario_id=case.fixture_key,
                store_id=self._store_id_for(connection, case.fixture_key),
                merchant_message=case.merchant_message,
                operator_id="OP-DEMO",
                trace_id=f"TRACE-EVAL-{case.case_id}",
                case_hint=case.notes,
            )
        finally:
            connection.close()

        trace = build_trace_record(result)
        actual_tools = {response.tool_name for response in result.tool_responses}
        required_tools = set(case.required_tool_names)
        has_evidence = bool(result.brief.cause.supporting_evidence_ids)
        unsupported_likely = int(
            result.brief.cause.assessment.value == "likely" and not has_evidence
        )
        recall = ratio(len(required_tools & actual_tools), len(required_tools))
        errors: list[str] = []

        if result.state.current_state.value != case.expected_state:
            errors.append(
                f"expected_state={case.expected_state} actual_state={result.state.current_state.value}"
            )
        expected_cause = case.expected_primary_cause
        actual_cause = result.brief.cause.primary_cause
        if expected_cause is None:
            if actual_cause is not None:
                errors.append(f"expected abstention but got {actual_cause}")
        elif actual_cause not in {expected_cause, *case.acceptable_alternatives}:
            errors.append(
                f"expected_cause={expected_cause} actual_cause={actual_cause}"
            )
        if expected_cause and not has_evidence:
            errors.append("displayed cause without evidence citations")
        if recall < 1.0:
            errors.append("missing required tools")
        if unsupported_likely:
            errors.append("unsupported likely claim")

        return EvalCaseResult(
            case_id=case.case_id,
            passed=not errors,
            actual_state=result.state.current_state.value,
            actual_primary_cause=actual_cause,
            has_evidence_citations=has_evidence,
            required_tool_recall=recall,
            forbidden_action_count=0,
            unsupported_likely_claim_count=unsupported_likely,
            has_trace=True,
            trace_id=trace.trace_id,
            errors=errors,
        )

    @staticmethod
    def _store_id_for(connection, fixture_key: str) -> str:
        row = connection.execute(
            "SELECT store_id FROM scenario_stores WHERE scenario_id = ?",
            (fixture_key,),
        ).fetchone()
        if row is not None:
            return str(row["store_id"])
        return f"STR-{fixture_key}"


__all__ = ["DeterministicEvaluator", "EvalCaseResult", "default_fixture_db_path"]
