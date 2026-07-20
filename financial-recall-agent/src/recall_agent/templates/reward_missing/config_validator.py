"""Reward-missing-specific Product Config validation."""

from __future__ import annotations

from typing import Any

from src.recall_agent.core.common_validator import (
    ProductConfigValidationError,
    validate_common_product_config,
)
from src.recall_agent.core.registry_loader import get_rule_template_id


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductConfigValidationError(message)


def reward_policy(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("h07_reward_policy") or config.get("h07_cashback_policy") or {}


def validate_reward_config(
    *,
    config: dict[str, Any],
    rule: dict[str, Any],
    registry: dict[str, Any],
    execution_date: str | None = None,
) -> dict[str, Any]:
    result = validate_common_product_config(
        config=config,
        rule=rule,
        registry=registry,
        execution_date=execution_date,
    )
    policy = reward_policy(config)
    template_id = get_rule_template_id(rule)

    _require(bool(policy), "reward policy is required for reward-missing template.")
    _require(policy.get("rule_template_id") == template_id, "Reward policy rule_template_id does not match.")
    _require(bool(policy.get("rounding_policy")), "rounding_policy is required.")
    _require(bool(policy.get("monthly_cap")), "monthly_cap policy is required.")
    _require(bool(policy.get("payment_schedule")), "payment_schedule is required.")
    _require(
        bool((policy.get("eligibility") or {}).get("excluded_merchant_categories")),
        "excluded merchant categories are required.",
    )

    reasons = result["review_required_reasons"]
    rounding = policy.get("rounding_policy") or {}
    cap = policy.get("monthly_cap") or {}
    calendar = (policy.get("payment_schedule") or {}).get("business_calendar") or {}
    if rounding.get("is_mvp_assumption"):
        reasons.append("rounding_policy_is_mvp_assumption")
    if cap.get("allocation_method_is_mvp_assumption"):
        reasons.append("monthly_cap_allocation_is_mvp_assumption")
    if calendar.get("mvp_implementation") == "WEEKEND_ONLY":
        reasons.append("holiday_business_day_policy_not_finalized")
    return result
