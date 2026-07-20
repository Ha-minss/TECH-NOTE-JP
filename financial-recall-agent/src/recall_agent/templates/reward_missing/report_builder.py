"""Build the stable public report for a reward-missing investigation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.recall_agent.core.models import ExecutionContext, ExecutionRequest, RuleReport
from src.recall_agent.policy_rag.policy_basis_retriever import (
    basis_ids_from_rule_and_config,
    retrieve_policy_basis,
)
from src.recall_agent.templates.reward_missing.incident_analyzer import IncidentAnalysis


ROW_COLUMNS = (
    "rule_template_id", "product_config_id", "customer_id", "purchase_id",
    "card_id", "product_id", "purchase_date", "purchase_month", "amount",
    "merchant_name", "merchant_category", "reward_batch_id", "processing_route",
    "policy_expected_reward_amount", "bank_expected_reward_amount",
    "paid_reward_amount", "harm_amount", "detected_error_types",
    "policy_eligibility_reason",
)


def _safe_int(value: Any) -> int:
    return 0 if value is None or pd.isna(value) else int(value)


def _rows(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    keep = [column for column in ROW_COLUMNS if column in frame.columns]
    records = frame[keep].head(limit).to_dict(orient="records")
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in records
    ]


def _error_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series.dropna().astype(str):
        for error_type in filter(None, (part.strip() for part in value.split("|"))):
            counts[error_type] = counts.get(error_type, 0) + 1
    return counts


def build_report(
    request: ExecutionRequest,
    context: ExecutionContext,
    analysis: IncidentAnalysis,
) -> RuleReport:
    raw_rule = dict(context.rule.raw)
    basis_ids = basis_ids_from_rule_and_config(raw_rule, dict(context.product_config))
    policy_basis = retrieve_policy_basis(
        basis_ids,
        product_config=dict(context.product_config),
        policy_basis_index_path=context.policy_basis_path,
    )
    confirmed = not analysis.complainant_affected.empty
    affected_ids = analysis.affected["customer_id"].dropna().astype(str).unique()
    harm_series = analysis.affected.get("harm_amount", pd.Series(dtype=float)).fillna(0)
    complainant_harm = analysis.complainant_affected.get(
        "harm_amount", pd.Series(dtype=float)
    ).fillna(0)

    return {
        "execution_id": f"exec_{uuid.uuid4().hex[:16]}",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "runner_type": "GENERIC_RUN_RULE",
        "handler_name": "RewardRecallHandler",
        "rule_id": context.rule.rule_id,
        "legacy_rule_ids": raw_rule.get("legacy_rule_ids") or [],
        "rule_version": raw_rule.get("rule_version"),
        "rule_template_id": context.rule.rule_template_id,
        "rule_template": context.rule.rule_template,
        "harm_type": raw_rule.get("harm_type"),
        "product_id": context.product_config_ref.product_id,
        "product_config_id": context.product_config_ref.config_id,
        "product_policy_version": context.product_config_ref.policy_version,
        "product_policy_config_path": str(context.product_config_ref.path),
        "data_contract_id": raw_rule.get("rule_execution", {}).get("data_contract_id"),
        "complaint": analysis.complaint,
        "complainant_confirmed": confirmed,
        "complainant_customer_id": analysis.complainant_customer_id,
        "complainant_harm_amount": _safe_int(complainant_harm.sum()),
        "incident_scope": analysis.incident_scope,
        "affected_customer_count": len(affected_ids),
        "affected_transaction_count": len(analysis.affected),
        "unreported_customer_count": analysis.unreported_affected[
            "customer_id"
        ].dropna().astype(str).nunique(),
        "unreported_transaction_count": len(analysis.unreported_affected),
        "total_harm_amount": _safe_int(harm_series.sum()),
        "error_type_counts": _error_counts(
            analysis.affected.get("detected_error_types", pd.Series(dtype=str))
        ),
        "decision": {
            "status": "REQUIRES_HUMAN_CONFIRMATION" if confirmed else "NO_CONFIRMED_COMPLAINANT_HARM",
            "amount_label": "PoC estimated harm / requires review",
            "recommended_action": "Review evidence and approve any customer remediation.",
            "automatic_refund_allowed": False,
            "human_review_required": True,
        },
        "complainant_affected_rows": _rows(analysis.complainant_affected, request.max_customer_rows),
        "unreported_affected_rows": _rows(analysis.unreported_affected, request.max_customer_rows),
        "normal_exclusion_rows_for_complainant": _rows(analysis.normal_exclusions, request.max_customer_rows),
        "requires_human_review_rows": _rows(analysis.requires_review, request.max_customer_rows),
        "policy_basis": policy_basis,
        "audit": {
            "bundle_id": context.bundle_id,
            "rule_template_id": context.rule.rule_template_id,
            "product_config_id": context.product_config_ref.config_id,
            "product_policy_version": context.product_config_ref.policy_version,
            "llm_generated_sql": False,
            "free_form_sql_allowed": False,
            "used_private_ground_truth": False,
            "used_v3_injected_anomaly_manifest": False,
            "repository_name": raw_rule.get("rule_execution", {}).get("repository_name"),
            "orchestrator_name": "generic_rule_runner",
            "handler_name": "RewardRecallHandler",
            "human_review_required": True,
            "runtime_prohibited_sources": [
                "private_ground_truth.csv",
                "v3_injected_anomaly_manifest.csv",
            ],
        },
    }
