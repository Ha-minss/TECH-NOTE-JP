from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Type

from storeops.core.contracts import CaseBrief, CaseState, EvidenceRecord, WorkflowState
from storeops.core.executor import ToolExecutor
from storeops.core.planner import ToolCatalog
from storeops.core.policy_checks import EvidencePlanBuilder
from storeops.core.safety import SafetyDecision, SafetyGate
from storeops.core.types import WorkflowResult
from storeops.core.workflow import Workflow
from storeops.core.retrieval import DeterministicEmbeddingProvider, HybridPolicyRetriever
from storeops.domains.offline_payment_ops.evidence_rules import OfflinePaymentEvidenceBuilder
from storeops.domains.offline_payment_ops.parser import OfflinePaymentCaseParser
from storeops.domains.offline_payment_ops.policy_check_rules import OfflinePaymentPolicyCheckExtractor
from storeops.domains.offline_payment_ops.reasoner_rules import OfflinePaymentReasoner
from storeops.domains.offline_payment_ops.tool_gateway import OfflinePaymentToolGateway


class OfflinePaymentToolExecutor(ToolExecutor):
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        operator_id: str,
        trace_id: str,
        scenario_id: str,
    ) -> None:
        super().__init__(
            connection,
            operator_id=operator_id,
            trace_id=trace_id,
            scenario_id=scenario_id,
            gateway_factory=OfflinePaymentToolGateway,
        )


class OfflinePaymentWorkflow(Workflow):
    @classmethod
    def default(cls, connection: sqlite3.Connection) -> "OfflinePaymentWorkflow":
        return build_offline_payment_workflow(connection, workflow_class=cls)

    @classmethod
    def with_llm(
        cls,
        connection: sqlite3.Connection,
        *,
        client,
        model_name: str = "synthetic-llm",
    ) -> "OfflinePaymentWorkflow":
        return build_offline_payment_workflow(
            connection,
            client=client,
            model_name=model_name,
            workflow_class=cls,
        )

    def run_scenario(
        self,
        scenario_id: str,
        *,
        operator_id: str,
        trace_id: str,
    ) -> WorkflowResult:
        store_id, merchant_message, case_hint = offline_payment_scenario_input(
            self.connection,
            scenario_id,
        )
        return self.run_case(
            scenario_id=scenario_id,
            store_id=store_id,
            merchant_message=merchant_message,
            case_hint=case_hint,
            operator_id=operator_id,
            trace_id=trace_id,
        )


def build_offline_payment_workflow(
    connection: sqlite3.Connection,
    *,
    client=None,
    model_name: str = "synthetic-llm",
    workflow_class: Type[Workflow] = OfflinePaymentWorkflow,
) -> Workflow:
    project_root = Path(__file__).resolve().parents[4]
    policy_dir = project_root / "data" / "policies" / "offline_payment_ops"
    catalog_path = project_root / "data" / "tool_catalog" / "offline_payment_ops_tools.json"
    retriever = HybridPolicyRetriever.from_policy_dir(
        policy_dir,
        embedding_provider=DeterministicEmbeddingProvider(),
        dense_weight=0.6,
        bm25_weight=0.4,
    )
    parser = OfflinePaymentCaseParser()
    tool_catalog = ToolCatalog.load(catalog_path)
    checklist_extractor = OfflinePaymentPolicyCheckExtractor()
    clarification_generator = None
    response_drafter = None

    if client is not None:
        from storeops.llm.case_parser import LLMCaseParser
        from storeops.llm.checklist_extractor import LLMEvidenceChecklistExtractor
        from storeops.llm.clarification import ClarificationQuestionGenerator
        from storeops.llm.drafting import MerchantResponseDrafter
        from storeops.llm.runtime import LLMRuntime

        runtime = LLMRuntime(client=client, model_name=model_name)
        parser = LLMCaseParser(runtime=runtime, fallback_parser=parser)
        checklist_extractor = LLMEvidenceChecklistExtractor(runtime=runtime)
        clarification_generator = ClarificationQuestionGenerator(runtime=runtime)
        response_drafter = MerchantResponseDrafter(runtime=runtime)

    return workflow_class(
        connection=connection,
        parser=parser,
        retriever=retriever,
        tool_catalog=tool_catalog,
        executor_factory=lambda conn, **kwargs: OfflinePaymentToolExecutor(conn, **kwargs),
        evidence_builder=OfflinePaymentEvidenceBuilder(),
        reasoner=OfflinePaymentReasoner(),
        safety_gate=SafetyGate(),
        brief_builder=build_offline_payment_brief,
        confirmation_fact_formatter=offline_payment_confirmed_facts,
        fallback_response_builder=offline_payment_fallback_response,
        clarification_generator=clarification_generator,
        response_drafter=response_drafter,
        checklist_extractor=checklist_extractor,
        evidence_plan_builder=EvidencePlanBuilder(),
    )


def offline_payment_scenario_input(
    connection: sqlite3.Connection,
    scenario_id: str,
) -> tuple[str, str, str]:
    row = connection.execute(
        """
        SELECT
            ss.store_id AS store_id,
            s.merchant_message AS merchant_message,
            s.title AS title
        FROM scenarios s
        JOIN scenario_stores ss
          ON ss.scenario_id = s.scenario_id
        WHERE s.scenario_id = ?
        """,
        (scenario_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    merchant_message = str(row["merchant_message"]).strip() or "Payment approval failure was reported."
    return (
        str(row["store_id"]),
        merchant_message,
        "merchant-reported payment issue",
    )


def build_offline_payment_brief(decision: SafetyDecision, state: CaseState) -> CaseBrief:
    cause = decision.cause.primary_cause
    route = {
        "duplicate_tid": "installation_or_van_owner_review",
        "terminal_identifier_mismatch": "installation_partner",
        "van_merchant_registration_missing": "van_registration_owner",
        "pos_front_connection_issue": "pos_front_support",
    }.get(cause or "")
    actions = decision.cause.next_checks or ["manual_review_required"]
    response = (
        "확인된 시스템 근거를 바탕으로 담당자가 검토 결과를 안내드리겠습니다."
        if cause
        else "현재 확인된 기록만으로는 원인을 확정하기 어려워 운영자 검토가 필요합니다."
    )
    return CaseBrief(
        cause=decision.cause,
        state=decision.state,
        operator_actions=actions,
        recommended_route=route,
        merchant_response=response,
    )


def offline_payment_confirmed_facts(evidence: list[EvidenceRecord]) -> list[str]:
    return [f"{record.fact_type}:{record.normalized_value}" for record in evidence[:3]]


def offline_payment_fallback_response(
    state: WorkflowState,
    clarification_questions: list[str],
) -> str:
    if state is WorkflowState.READY_FOR_REVIEW:
        return "확인된 시스템 근거를 바탕으로 담당자가 검토 결과를 안내드리겠습니다."
    if state is WorkflowState.NEEDS_CLARIFICATION and clarification_questions:
        joined = " / ".join(clarification_questions)
        return f"추가 확인이 필요한 항목은 다음과 같습니다: {joined}"
    return "현재 확인된 기록만으로는 원인을 확정하기 어렵습니다."


__all__ = [
    "OfflinePaymentToolExecutor",
    "OfflinePaymentWorkflow",
    "build_offline_payment_brief",
    "build_offline_payment_workflow",
    "offline_payment_confirmed_facts",
    "offline_payment_fallback_response",
    "offline_payment_scenario_input",
]
