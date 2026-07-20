from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest


pytestmark = pytest.mark.core


def test_core_models_define_typed_execution_boundary():
    from src.recall_agent.core.models import (
        ApprovedSql,
        ExecutionContext,
        ExecutionRequest,
        RuleHandler,
    )

    assert ApprovedSql.__dataclass_fields__
    assert ExecutionContext.__dataclass_fields__
    assert ExecutionRequest.__dataclass_fields__
    assert getattr(RuleHandler, "_is_protocol", False) is True


def test_plugin_registry_accepts_new_template_without_core_changes():
    from src.recall_agent.core.plugins import PluginRegistry

    @dataclass(frozen=True)
    class FakePlugin:
        rule_template: str = "fake_template"

        def create_handler(self):
            return object()

        def validate_config(self, *, config, rule, registry, execution_date=None):
            return {"validated": True}

    plugin = FakePlugin()
    registry = PluginRegistry([plugin])

    assert registry.require("fake_template") is plugin


def test_common_product_validator_does_not_require_h07_policy_fields():
    from src.recall_agent.core.common_validator import validate_common_product_config

    registry = {
        "execution_mode": "MVP_DEMO",
        "allowed_config_approval_statuses_by_execution_mode": {
            "MVP_DEMO": ["APPROVED_FOR_MVP_DEMO"]
        },
    }
    rule = {
        "rule_template_id": "H02_FEE_OVERCHARGE",
        "product_scope": {"product_family_id": "ACCOUNT_FEE"},
    }
    config = {
        "config_type": "PRODUCT_POLICY_CONFIG",
        "config_id": "FEE_CONFIG_V1",
        "product_id": "FEE_PRODUCT",
        "product_family": "ACCOUNT_FEE",
        "supported_rule_template_id": "H02_FEE_OVERCHARGE",
        "approval_status": "APPROVED_FOR_MVP_DEMO",
        "production_approval_status": "NOT_APPROVED_FOR_PRODUCTION",
        "effective_from": "2026-01-01",
        "effective_to": None,
        "source_document": {"source_document_hash_sha256": "abc"},
        "runtime_controls": {
            "llm_may_use_this_config_for_calculation": False,
            "rag_may_use_this_config_for_calculation": False,
            "deterministic_rule_engine_must_use_this_config": True,
            "human_review_required_before_refund": True,
            "automatic_customer_refund_allowed": False,
        },
    }

    result = validate_common_product_config(
        config=config,
        rule=rule,
        registry=registry,
    )

    assert result["product_config_id"] == "FEE_CONFIG_V1"


def test_reward_handler_has_small_typed_orchestration_signature():
    from src.recall_agent.templates.reward_missing.handler import RewardRecallHandler

    parameters = list(inspect.signature(RewardRecallHandler.run).parameters)
    source_lines = inspect.getsource(RewardRecallHandler.run).splitlines()

    assert parameters == ["self", "request", "context"]
    assert len(source_lines) <= 45


def test_policy_basis_retriever_imports_without_faiss():
    from src.recall_agent.policy_rag.policy_basis_retriever import (
        RAG_USAGE,
        basis_ids_for_harm_type,
    )

    assert RAG_USAGE == "POLICY_BASIS_RETRIEVAL_ONLY"
    assert "H07-BASIS-RATE" in basis_ids_for_harm_type("H07")
