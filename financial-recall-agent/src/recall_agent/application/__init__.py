"""Public application API wired to the explicit approved plugin set."""

from __future__ import annotations

from pathlib import Path

from src.recall_agent.composition import PLUGIN_REGISTRY
from src.recall_agent.core.models import ExecutionRequest, RuleReport
from src.recall_agent.core.rule_runner import execute_rule


def run_rule(
    *,
    rule_id: str,
    complaint_id: str,
    bundle_path: str,
    product_config_id: str | None = None,
    rule_registry_path: str | None = None,
    dataset_base_path: str | None = None,
    sql_dir: str | None = None,
    policy_basis_index_path: str | None = None,
    max_customer_rows: int = 50,
    audit_log_path: str | None = None,
    dev_mode: bool | None = None,
) -> RuleReport:
    return execute_rule(
        ExecutionRequest(
            rule_id=rule_id,
            complaint_id=complaint_id,
            product_config_id=product_config_id,
            bundle_path=Path(bundle_path),
            rule_registry_path=rule_registry_path,
            dataset_base_path=dataset_base_path,
            sql_dir=sql_dir,
            policy_basis_index_path=policy_basis_index_path,
            max_customer_rows=max_customer_rows,
            audit_log_path=audit_log_path,
            dev_mode=dev_mode,
        ),
        PLUGIN_REGISTRY,
    )

