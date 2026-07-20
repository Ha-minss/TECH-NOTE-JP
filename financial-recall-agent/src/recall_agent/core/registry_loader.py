"""Rule registry loader and validator.

This module contains product-independent registry validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.recall_agent.core.artifact_hash import resolve_project_path
from src.recall_agent.core.demo_paths import DEMO_RULE_REGISTRY_PATH
from typing import Any


DEFAULT_RULE_REGISTRY_PATH = DEMO_RULE_REGISTRY_PATH



def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rule_registry(rule_registry_path: str | Path = DEFAULT_RULE_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_json(rule_registry_path)
    if registry.get("registry_type") != "RECALL_RULE_REGISTRY":
        raise ValueError(
            f"Invalid registry_type: {registry.get('registry_type')!r}. "
            "Expected 'RECALL_RULE_REGISTRY'."
        )
    return registry


def get_rule(registry: dict[str, Any], rule_id: str) -> dict[str, Any]:
    for rule in registry.get("rules", []):
        if rule.get("rule_id") == rule_id or rule_id in (rule.get("legacy_rule_ids") or []):
            return rule
    available = [rule.get("rule_id") for rule in registry.get("rules", [])]
    aliases = {rule.get("rule_id"): rule.get("legacy_rule_ids", []) for rule in registry.get("rules", [])}
    raise ValueError(f"Rule not found: {rule_id}. Available rules: {available}, aliases: {aliases}")


def get_rule_template(rule: dict[str, Any]) -> str:
    """Return the executable handler template name from the registry."""
    return (
        rule.get("rule_execution", {}).get("rule_template")
        or rule.get("rule_template")
        or rule.get("calculation_policy", {}).get("rule_template")
        or ""
    )


def get_rule_template_id(rule: dict[str, Any]) -> str:
    """Return the stable business template identifier."""
    return (
        rule.get("rule_template_id")
        or rule.get("rule_execution", {}).get("rule_template_id")
        or rule.get("calculation_policy", {}).get("rule_template_id")
        or get_rule_template(rule)
        or ""
    )


def get_product_config_path(rule: dict[str, Any]) -> str:
    path = (
        rule.get("product_scope", {}).get("product_policy_config_path")
        or rule.get("product_config_path")
        or rule.get("rule_execution", {}).get("product_policy_config_path")
    )
    if not path:
        raise ValueError(
            f"Rule {rule.get('rule_id')} has no product config path. "
            "Generic rule templates must select Product Config from the approved bundle."
        )
    return str(path)


def get_product_id(rule: dict[str, Any]) -> str:
    product_id = (
        rule.get("product_scope", {}).get("product_id")
        or rule.get("product_id")
    )
    if not product_id:
        raise ValueError(
            f"Rule {rule.get('rule_id')} has no product_id. "
            "Generic rule templates must select product_id from Product Config."
        )
    return str(product_id)


def validate_rule_for_execution(registry: dict[str, Any], rule: dict[str, Any]) -> None:
    """Validate governance gates before running any handler.

    This is deliberately strict for MVP because the system deals with financial harm.
    """
    execution_mode = registry.get("execution_mode", "MVP_DEMO")

    global_controls = registry.get("global_controls", {})
    if global_controls.get("llm_can_generate_sql") is not False:
        raise ValueError("Registry must set global_controls.llm_can_generate_sql = false.")
    if global_controls.get("llm_can_calculate_refund_amount") is not False:
        raise ValueError("Registry must set global_controls.llm_can_calculate_refund_amount = false.")
    if global_controls.get("deterministic_rule_engine_required") is not True:
        raise ValueError("Registry must require deterministic_rule_engine_required = true.")

    allowed_statuses = registry.get(
        "allowed_config_approval_statuses_by_execution_mode", {}
    ).get(execution_mode, [])

    rule_status = rule.get("rule_status")
    approval_status = rule.get("approval_status")

    if rule_status not in {"ACTIVE_FOR_MVP_DEMO", "ACTIVE_FOR_PRODUCTION"}:
        raise ValueError(f"Rule is not active: rule_status={rule_status!r}")

    if approval_status not in allowed_statuses:
        raise ValueError(
            f"Rule approval_status={approval_status!r} is not allowed "
            f"for execution_mode={execution_mode!r}. Allowed: {allowed_statuses}"
        )

    data_policy = rule.get("data_access_policy", {})
    if data_policy.get("free_form_sql_allowed") is not False:
        raise ValueError("Rule must set data_access_policy.free_form_sql_allowed = false.")

    if data_policy.get("approved_repository_only") is not True:
        raise ValueError("Rule must set data_access_policy.approved_repository_only = true.")

    calculation_policy = rule.get("calculation_policy", {})
    if calculation_policy.get("independent_policy_calculation_required") is not True:
        raise ValueError(
            "Rule must set calculation_policy.independent_policy_calculation_required = true."
        )

    template = get_rule_template(rule)
    if not template:
        raise ValueError(f"Rule {rule.get('rule_id')} has no rule_template.")


