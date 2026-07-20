"""CLI-safe rendering helpers kept outside execution orchestration."""

from __future__ import annotations

import json

from src.recall_agent.core.models import RuleReport


def print_report_summary(report: RuleReport) -> None:
    keys = (
        "execution_id", "runner_type", "handler_name", "rule_id",
        "rule_template_id", "rule_template", "product_id", "product_config_id",
        "product_policy_version", "complainant_confirmed",
        "complainant_customer_id", "complainant_harm_amount",
        "affected_customer_count", "affected_transaction_count",
        "unreported_customer_count", "unreported_transaction_count",
        "total_harm_amount", "error_type_counts", "decision",
    )
    summary = {key: report.get(key) for key in keys}
    summary["complaint_id"] = (report.get("complaint") or {}).get("complaint_id")
    summary["audit_log_path"] = (report.get("audit") or {}).get("audit_log_path")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
