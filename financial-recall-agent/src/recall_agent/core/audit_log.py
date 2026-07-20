"""Append-only audit logging for Financial Recall executions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.recall_agent.core.artifact_hash import resolve_project_path, sha256_json_canonical
from src.recall_agent.core.demo_paths import DEMO_AUDIT_LOG_PATH


DEFAULT_AUDIT_LOG_PATH = DEMO_AUDIT_LOG_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_audit_record(
    *,
    report: dict[str, Any],
    bundle_id: str | None,
    artifact_verification: dict[str, Any],
    product_config_validation: dict[str, Any] | None,
    data_contract_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    complaint = report.get("complaint") or {}
    decision = report.get("decision") or {}
    review_reasons = []
    if product_config_validation:
        review_reasons.extend(product_config_validation.get("review_required_reasons") or [])
    if decision.get("human_review_required"):
        review_reasons.append("human_review_required_before_customer_refund")
    if decision.get("automatic_refund_allowed") is False:
        review_reasons.append("automatic_refund_disabled")

    return {
        "run_id": report.get("execution_id"),
        "timestamp": report.get("executed_at") or _now_iso(),
        "bundle_id": bundle_id,
        "rule_id": report.get("rule_id"),
        "rule_version": report.get("rule_version"),
        "rule_template_id": report.get("rule_template_id"),
        "rule_template": report.get("rule_template"),
        "harm_type": report.get("harm_type"),
        "product_id": report.get("product_id"),
        "product_config_id": report.get("product_config_id"),
        "product_policy_version": report.get("product_policy_version"),
        "complaint_id": complaint.get("complaint_id"),
        "complainant_customer_id": report.get("complainant_customer_id"),
        "candidate_categories": [report.get("harm_type")] if report.get("harm_type") else [],
        "selected_rule_id": report.get("rule_id"),
        "registry_hash": artifact_verification.get("registry_hash"),
        "bundle_hash": artifact_verification.get("bundle_hash"),
        "config_hash": artifact_verification.get("config_hash"),
        "selected_product_config_id": artifact_verification.get("selected_product_config_id") or report.get("product_config_id"),
        "allowed_product_config_hashes": artifact_verification.get("allowed_product_config_hashes") or [],
        "data_contract_hash": artifact_verification.get("data_contract_hash"),
        "sql_hashes": artifact_verification.get("sql_hashes") or [],
        "data_contract_version": (
            data_contract_validation or {}
        ).get("data_contract_id") or report.get("data_contract_id"),
        "input_snapshot_hash": sha256_json_canonical(complaint),
        "policy_basis_ids": [item.get("basis_id") for item in report.get("policy_basis") or []],
        "result_summary": {
            "complainant_confirmed": report.get("complainant_confirmed"),
            "affected_customers": report.get("affected_customer_count"),
            "affected_transactions": report.get("affected_transaction_count"),
            "unreported_customers": report.get("unreported_customer_count"),
            "estimated_total_gap": report.get("total_harm_amount"),
            "error_type_counts": report.get("error_type_counts") or {},
        },
        "review_status": "pending_human_review" if decision.get("human_review_required") else "review_not_required",
        "review_required_reasons": sorted(set(review_reasons)),
        "automatic_refund_allowed": decision.get("automatic_refund_allowed"),
    }


def append_audit_log(record: dict[str, Any], path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> Path:
    """Append one JSON record to an audit JSONL file."""
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return resolved
