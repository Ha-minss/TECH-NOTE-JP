"""Approved reward-missing template plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.recall_agent.templates.reward_missing.config_validator import validate_reward_config
from src.recall_agent.templates.reward_missing.handler import RewardRecallHandler


@dataclass(frozen=True)
class RewardMissingPlugin:
    rule_template: str = "reward_policy_recalculate_and_reconcile"

    def create_handler(self) -> RewardRecallHandler:
        return RewardRecallHandler()

    def validate_config(
        self,
        *,
        config: dict[str, Any],
        rule: dict[str, Any],
        registry: dict[str, Any],
        execution_date: str | None = None,
    ) -> dict[str, Any]:
        return validate_reward_config(
            config=config,
            rule=rule,
            registry=registry,
            execution_date=execution_date,
        )
