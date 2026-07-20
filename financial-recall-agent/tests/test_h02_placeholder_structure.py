import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.core


def test_h02_placeholder_files_exist_without_runtime_sql():
    base = Path("src/recall_agent/templates/fee_overcharge")
    assert (base / "handler.py").exists()
    assert not (base / "repository.py").exists()
    assert not (base / "evidence_builder.py").exists()
    assert (base / "README.md").exists()
    assert not Path("sql/fee_overcharge").exists(), "H02 SQL should wait for real policy documents."


def test_h02_registry_rule_is_not_executable_placeholder():
    registry = json.loads(Path("data/demo/rules/rule_registry.json").read_text(encoding="utf-8"))
    h02 = next(rule for rule in registry["rules"] if rule["rule_id"] == "H02-FEE-WAIVER-OVERCHARGE-001")
    assert h02["rule_status"] == "PLACEHOLDER_NOT_EXECUTABLE"
    assert h02["rule_execution"]["rule_template"] == "fee_policy_recalculate_and_reconcile"


def test_core_rejects_h02_before_placeholder_handler_execution():
    from src.recall_agent.application import run_rule

    with pytest.raises(ValueError, match="not part of approved bundle"):
        run_rule(
            rule_id="H02-FEE-WAIVER-OVERCHARGE-001",
            complaint_id="CP-H02-PLACEHOLDER",
            bundle_path="data/demo/rules/bundles/h07_reward_missing_mvp.json",
        )

