import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.recall_agent.core.bundle_loader import (
    load_bundle,
    verify_bundle_static_artifacts,
)
from src.recall_agent.core.data_contract_validator import validate_data_contract
from src.recall_agent.templates.reward_missing.config_validator import validate_reward_config
from src.recall_agent.core.registry_loader import get_rule, load_json, load_rule_registry
from src.recall_agent.application import run_rule
from src.recall_agent.interfaces.cli.h07_reward_missing_demo import H07_BUNDLE_PATH, H07_PRODUCT_CONFIG_ID
from src.recall_agent.core.runtime_controls import path_override_allowed


def test_sql_mutation_fails_bundle_hash_verification(tmp_path):
    bundle = load_bundle(H07_BUNDLE_PATH)
    mutated_sql = tmp_path / "01_build_base_frame.sql"
    original_sql = Path(bundle["rules"][0]["approved_sql"][0]["path"])
    mutated_sql.write_text(original_sql.read_text(encoding="utf-8") + "\n-- mutated\n", encoding="utf-8")

    bundle["rules"][0]["approved_sql"][0]["path"] = str(mutated_sql)

    with pytest.raises(ValueError, match="Approved SQL hash mismatch"):
        verify_bundle_static_artifacts(bundle, rule_id="H07-REWARD-MISSING-TEMPLATE")


def test_product_config_unapproved_status_fails_runtime_validation():
    registry = load_rule_registry()
    rule = get_rule(registry, "H07-REWARD-MISSING-TEMPLATE")
    bundle = load_bundle(H07_BUNDLE_PATH)
    config_path = bundle["rules"][0]["allowed_product_configs"][0]["product_config_path"]
    config = load_json(config_path)
    config["approval_status"] = "DRAFT"

    with pytest.raises(ValueError, match="approval_status"):
        validate_reward_config(config=config, rule=rule, registry=registry)


def test_missing_required_column_fails_data_contract_validation(tmp_path):
    contract = load_json("data/demo/rules/data_contracts/h07_synthetic_v3.json")
    source_dataset = Path(contract["base_path"])
    temp_dataset = tmp_path / "dataset"
    shutil.copytree(source_dataset, temp_dataset)

    purchases_path = temp_dataset / "card_purchases.csv"
    purchases = pd.read_csv(purchases_path)
    purchases = purchases.drop(columns=["amount"])
    purchases.to_csv(purchases_path, index=False)

    with pytest.raises(ValueError, match="Required columns missing"):
        validate_data_contract(contract=contract, dataset_base_path=temp_dataset)


@pytest.mark.duckdb
def test_run_rule_writes_append_only_audit_log(tmp_path):
    pytest.importorskip("duckdb")
    audit_log = tmp_path / "audit_log.jsonl"
    report = run_rule(
        rule_id="H07-REWARD-MISSING-TEMPLATE",
        complaint_id="EVAL_BASE_0001",
        product_config_id=H07_PRODUCT_CONFIG_ID,
        bundle_path=H07_BUNDLE_PATH,
        audit_log_path=str(audit_log),
    )

    assert audit_log.exists()
    records = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == report["execution_id"]
    assert record["selected_rule_id"] == "H07-REWARD-MISSING-TEMPLATE"
    assert record["rule_template_id"] == "H07_REWARD_MISSING"
    assert record["product_config_id"] == "JB_SMART_CASHBACK_CHECK__2022-07__v2"
    assert record["product_policy_version"] == "2022-07"
    assert record["registry_hash"]
    assert record["config_hash"]
    assert record["sql_hashes"]
    assert record["review_status"] == "pending_human_review"


def test_normal_ui_mode_disallows_arbitrary_path_override():
    assert path_override_allowed(False) is False
    assert path_override_allowed(True) is True



