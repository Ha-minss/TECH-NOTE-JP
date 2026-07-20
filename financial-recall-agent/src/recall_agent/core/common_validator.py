"""Product-independent validation shared by all rule templates."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.recall_agent.core.registry_loader import get_rule_template_id


class ProductConfigValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductConfigValidationError(message)


def _parse_date(value: str | None) -> date | None:
    if value in {None, ""}:
        return None
    return date.fromisoformat(str(value))


def validate_common_product_config(
    *,
    config: dict[str, Any],
    rule: dict[str, Any],
    registry: dict[str, Any],
    execution_date: str | None = None,
) -> dict[str, Any]:
    execution_mode = registry.get("execution_mode", "MVP_DEMO")
    allowed_statuses = registry.get(
        "allowed_config_approval_statuses_by_execution_mode", {}
    ).get(execution_mode, [])
    expected_template_id = get_rule_template_id(rule)

    _require(config.get("config_type") == "PRODUCT_POLICY_CONFIG", "Invalid product config type.")
    _require(bool(config.get("config_id")), "Product config_id is required.")
    _require(bool(config.get("product_id")), "Product config product_id is required.")
    _require(
        config.get("approval_status") in allowed_statuses,
        "Product config approval_status is not allowed for execution mode.",
    )
    _require(
        config.get("production_approval_status")
        in {"NOT_APPROVED_FOR_PRODUCTION", "APPROVED_FOR_PRODUCTION"},
        "Product config production approval status is missing/invalid.",
    )
    _require(
        config.get("supported_rule_template_id") == expected_template_id,
        "Product config rule_template_id does not match rule template.",
    )
    _require(
        config.get("product_family") == rule.get("product_scope", {}).get("product_family_id"),
        "Product config product_family does not match rule product family.",
    )

    source_document = config.get("source_document") or {}
    _require(
        bool(source_document.get("source_document_hash_sha256")),
        "Product config source document hash is required.",
    )
    effective_from = _parse_date(config.get("effective_from"))
    effective_to = _parse_date(config.get("effective_to"))
    _require(effective_from is not None, "Product config effective_from is required.")
    if execution_date:
        run_date = _parse_date(execution_date)
        _require(run_date is not None and effective_from <= run_date, "Execution date is before config effective_from.")
        _require(effective_to is None or run_date <= effective_to, "Execution date is after config effective_to.")

    controls = config.get("runtime_controls") or {}
    _require(controls.get("llm_may_use_this_config_for_calculation") is False, "LLM must not calculate from product config.")
    _require(controls.get("rag_may_use_this_config_for_calculation") is False, "RAG must not calculate from product config.")
    _require(controls.get("deterministic_rule_engine_must_use_this_config") is True, "Deterministic rule engine control is required.")
    _require(controls.get("human_review_required_before_refund") is True, "Human review must be required before refund.")
    _require(controls.get("automatic_customer_refund_allowed") is False, "Automatic customer refund must be disabled.")

    return {
        "rule_template_id": expected_template_id,
        "config_id": config.get("config_id"),
        "product_config_id": config.get("config_id"),
        "product_id": config.get("product_id"),
        "product_family": config.get("product_family"),
        "product_policy_version": config.get("policy_version"),
        "approval_status": config.get("approval_status"),
        "effective_from": config.get("effective_from"),
        "effective_to": config.get("effective_to"),
        "source_document_hash_sha256": source_document.get("source_document_hash_sha256"),
        "review_required_reasons": [],
    }
