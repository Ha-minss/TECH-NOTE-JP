"""Thin orchestration handler for the reward-missing template."""

from __future__ import annotations

from src.recall_agent.core.models import ExecutionContext, ExecutionRequest, RuleReport
from src.recall_agent.templates.reward_missing.incident_analyzer import analyze_incident
from src.recall_agent.templates.reward_missing.report_builder import build_report
from src.recall_agent.templates.reward_missing.repository import (
    build_reconciled_result,
    load_complaint,
)


class RewardRecallHandler:
    def run(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> RuleReport:
        complaint = load_complaint(context.dataset_path, request.complaint_id)
        rows = build_reconciled_result(context)
        analysis = analyze_incident(rows, complaint)
        return build_report(request, context, analysis)
