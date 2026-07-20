from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from storeops.core.interfaces import (
    CaseParser,
    EvidenceBuilder,
    EvidenceReasoner,
    SafetyGateProtocol,
    ToolExecutorProtocol,
)
from storeops.core.planner import ToolCatalog
from storeops.core.policy_checks import EvidencePlanBuilder
from storeops.core.types import WorkflowResult
from storeops.core.contracts import CaseBrief, CaseState, EvidenceRecord, WorkflowState


BriefBuilder = Callable[[Any, CaseState], CaseBrief]
ConfirmedFactFormatter = Callable[[list[EvidenceRecord]], list[str]]
FallbackResponseBuilder = Callable[[WorkflowState, list[str]], str]


@dataclass
class Workflow:
    connection: sqlite3.Connection
    parser: CaseParser
    retriever: Any
    tool_catalog: ToolCatalog
    checklist_extractor: Any
    evidence_plan_builder: EvidencePlanBuilder
    executor_factory: Callable[..., ToolExecutorProtocol]
    evidence_builder: EvidenceBuilder
    reasoner: EvidenceReasoner
    safety_gate: SafetyGateProtocol
    brief_builder: BriefBuilder
    confirmation_fact_formatter: ConfirmedFactFormatter | None = None
    fallback_response_builder: FallbackResponseBuilder | None = None
    clarification_generator: Any | None = None
    response_drafter: Any | None = None

    def run_case(
        self,
        *,
        scenario_id: str,
        store_id: str,
        merchant_message: str,
        operator_id: str,
        trace_id: str,
        case_hint: str | None = None,
    ) -> WorkflowResult:
        parsed = self.parser.parse(
            merchant_message,
            store_id=store_id,
            case_hint=case_hint,
        )
        retrieved = self.retriever.search(parsed.retrieval_query, top_k=3)
        policy_checks = self.checklist_extractor.extract(
            parsed_case=parsed,
            query=parsed.planner_query,
            retrieved_policies=retrieved,
            tool_catalog=self.tool_catalog,
        )
        plan, policy_check_trace = self.evidence_plan_builder.build(
            policy_checks=policy_checks,
            tool_catalog=self.tool_catalog,
            retrieved_policy_ids=[result.document_id for result in retrieved],
        )
        tool_responses = self.executor_factory(
            self.connection,
            operator_id=operator_id,
            trace_id=trace_id,
            scenario_id=scenario_id,
        ).execute(store_id=store_id, plan=plan)
        evidence = self.evidence_builder.build(
            scenario_id=scenario_id,
            tool_responses=tool_responses,
        )
        cause = self.reasoner.reason(evidence=evidence, parsed_case=parsed)
        required_tools = [
            call.tool_name for call in plan.planned_tool_calls if call.required
        ]
        decision = self.safety_gate.apply(
            parsed_case=parsed,
            planned_required_tools=required_tools,
            tool_responses=tool_responses,
            evidence=evidence,
            cause_assessment=cause,
        )
        clarification_questions: list[str] = []
        if (
            decision.state is WorkflowState.NEEDS_CLARIFICATION
            and self.clarification_generator is not None
        ):
            clarification_questions = [
                item.question
                for item in self.clarification_generator.generate(parsed_case=parsed)
            ]

        now = datetime.fromisoformat("2026-06-20T17:00:00+09:00")
        state = CaseState(
            case_id=f"CASE-{scenario_id}",
            trace_id=trace_id,
            scenario_id=scenario_id,
            store_id=store_id,
            merchant_message=merchant_message,
            current_state=decision.state,
            evidence=evidence,
            tool_calls=[response.tool_name for response in tool_responses],
            created_at=now,
            updated_at=now,
        )
        brief = self.brief_builder(decision, state)
        drafted_merchant_response = None
        if self.response_drafter is not None:
            drafted_merchant_response = self.response_drafter.draft(
                state=decision.state.value,
                primary_cause=decision.cause.primary_cause,
                confirmed_facts=self._confirmed_facts(evidence),
                clarification_questions=clarification_questions,
                fallback_text=self._fallback_merchant_response(
                    decision.state,
                    clarification_questions=clarification_questions,
                ),
            )
            brief = brief.model_copy(
                update={"merchant_response": drafted_merchant_response}
            )
        return WorkflowResult(
            state=state,
            parsed_case=parsed,
            retrieved_policy_ids=[result.document_id for result in retrieved],
            plan=plan,
            tool_responses=tool_responses,
            evidence=evidence,
            brief=brief,
            clarification_questions=clarification_questions,
            drafted_merchant_response=drafted_merchant_response,
            llm_traces=self._collect_llm_traces(),
            policy_checks=policy_checks,
            policy_check_trace=policy_check_trace,
        )

    def _confirmed_facts(self, evidence: list[EvidenceRecord]) -> list[str]:
        if self.confirmation_fact_formatter is not None:
            return self.confirmation_fact_formatter(evidence)
        return [
            f"{record.fact_type}:{record.normalized_value}"
            for record in evidence[:3]
        ]

    def _fallback_merchant_response(
        self,
        state: WorkflowState,
        *,
        clarification_questions: list[str],
    ) -> str:
        if self.fallback_response_builder is not None:
            return self.fallback_response_builder(state, clarification_questions)
        if state is WorkflowState.READY_FOR_REVIEW:
            return "확인된 시스템 근거를 바탕으로 검토 결과를 안내드리겠습니다."
        if state is WorkflowState.NEEDS_CLARIFICATION and clarification_questions:
            joined = " / ".join(clarification_questions)
            return f"추가 확인이 필요한 항목은 다음과 같습니다: {joined}"
        return "현재 확인된 기록만으로는 원인을 확정하기 어렵습니다."

    def _collect_llm_traces(self) -> list[Any]:
        traces: list[Any] = []
        for component in [
            self.parser,
            self.checklist_extractor,
            self.clarification_generator,
            self.response_drafter,
        ]:
            if component is None:
                continue
            trace = getattr(component, "last_trace", None)
            if trace is not None:
                traces.append(trace)
        return traces


__all__ = ["Workflow", "WorkflowResult"]

