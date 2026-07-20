import json
from pathlib import Path


RULES_DIR = Path("data/demo/rules")
PRODUCT_CONFIG = (
    RULES_DIR
    / "product_policy_configs"
    / "JB_SMART_CASHBACK_CHECK__2022-07__v2.json"
)
RULE_REGISTRY = RULES_DIR / "rule_registry.json"
DATA_CONTRACT = RULES_DIR / "data_contracts" / "h07_synthetic_v3.json"
GOVERNANCE = RULES_DIR / "governance" / "recall_agent_governance.json"
SCHEMA_DICTIONARY = Path(
    "data/demo/datasets/jb_h07_synthetic_dataset_v3/schema_data_dictionary.json"
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_product_config_v2_is_approved_only_for_mvp_demo():
    config = load(PRODUCT_CONFIG)
    assert config["schema_version"] == "2.0"
    assert config["config_type"] == "PRODUCT_POLICY_CONFIG"
    assert config["config_id"] == "JB_SMART_CASHBACK_CHECK__2022-07__v2"
    assert config["approval_status"] == "APPROVED_FOR_MVP_DEMO"
    assert config["production_approval_status"] == "NOT_APPROVED_FOR_PRODUCTION"
    assert config["source_document"]["source_document_hash_sha256"] == (
        "063c9a63be3f8fccb40d3471aa7dc0562197fc091a0c2855abb23619c165f58f"
    )


def test_product_config_v2_has_deterministic_runtime_controls():
    config = load(PRODUCT_CONFIG)
    policy = config["h07_reward_policy"]
    assert config["supported_rule_template_id"] == "H07_REWARD_MISSING"
    assert policy["rule_template_id"] == "H07_REWARD_MISSING"
    assert policy["rule_template"] == "reward_policy_recalculate_and_reconcile"
    assert policy["payment_schedule"]["business_calendar"]["calendar_id"] == (
        "KR_BANKING_CALENDAR"
    )
    assert policy["monthly_cap"]["allocation_order"] == ["purchase_date", "purchase_id"]
    assert {
        item["error_type"] for item in policy["reconciliation"]["detectable_error_types"]
    } == {
        "POLICY_CALCULATION_ERROR",
        "PAYMENT_EXECUTION_ERROR",
        "EXPECTED_REWARD_ROW_MISSING",
        "REWARD_LEDGER_ROW_MISSING",
    }
    controls = config["runtime_controls"]
    assert controls["llm_may_use_this_config_for_calculation"] is False
    assert controls["rag_may_use_this_config_for_calculation"] is False
    assert controls["deterministic_rule_engine_must_use_this_config"] is True
    assert controls["human_review_required_before_refund"] is True
    assert controls["automatic_customer_refund_allowed"] is False


def test_rule_registry_points_to_product_config_and_execution_components():
    registry = load(RULE_REGISTRY)
    assert registry["schema_version"] == "2.0"
    assert registry["registry_type"] == "RECALL_RULE_REGISTRY"
    assert registry["execution_mode"] == "MVP_DEMO"
    assert registry["production_approval_status"] == "NOT_APPROVED_FOR_PRODUCTION"
    rule = registry["rules"][0]
    assert rule["rule_id"] == "H07-REWARD-MISSING-TEMPLATE"
    assert rule["rule_template_id"] == "H07_REWARD_MISSING"
    assert "H07-SMART-CASHBACK-001" in rule["legacy_rule_ids"]
    assert rule["product_scope"]["product_policy_config_selection"] == (
        "FROM_APPROVED_BUNDLE_ALLOWED_PRODUCT_CONFIGS"
    )
    assert (
        rule["rule_execution"]["rule_template"]
        == "reward_policy_recalculate_and_reconcile"
    )
    assert rule["approval_status"] == "APPROVED_FOR_MVP_DEMO"
    assert set(rule["calculation_policy"]["detectable_error_types"]) == {
        "POLICY_CALCULATION_ERROR",
        "PAYMENT_EXECUTION_ERROR",
        "EXPECTED_REWARD_ROW_MISSING",
        "REWARD_LEDGER_ROW_MISSING",
    }
    assert rule["rule_execution"]["repository_name"] == "h07_reward_missing_repository"
    assert rule["rule_execution"]["data_contract_id"] == "H07_SYNTHETIC_V3"
    assert rule["output_requirements"]["customer_refund_decision"] == (
        "HUMAN_REVIEW_REQUIRED"
    )


def test_rule_registry_forbids_llm_and_rag_financial_actions():
    controls = load(RULE_REGISTRY)["global_controls"]
    assert controls["llm_can_recommend_rule_id"] is True
    assert controls["llm_can_execute_rule_directly"] is False
    assert controls["llm_can_generate_sql"] is False
    assert controls["llm_can_calculate_refund_amount"] is False
    assert controls["rag_can_determine_harm"] is False
    assert controls["rag_can_calculate_refund_amount"] is False
    assert controls["deterministic_rule_engine_required"] is True
    assert controls["human_review_required_before_refund"] is True
    assert controls["automatic_refund_allowed"] is False


def test_data_contract_exactly_matches_synthetic_schema():
    contract = load(DATA_CONTRACT)
    schema = load(SCHEMA_DICTIONARY)
    assert contract["data_contract_id"] == "H07_SYNTHETIC_V3"
    assert contract["base_path"] == (
        "data/demo/datasets/jb_h07_synthetic_dataset_v3"
    )
    for table_name, table_contract in contract["tables"].items():
        actual_columns = {item["column"] for item in schema[f"{table_name}.csv"]}
        assert set(table_contract["required_columns"]) <= actual_columns


def test_governance_forbids_llm_financial_actions():
    governance = load(GOVERNANCE)
    forbidden = set(governance["llm_policy"]["forbidden_actions"])
    assert {
        "generate_sql",
        "calculate_refund_amount",
        "finalize_customer_compensation",
        "override_product_policy_config",
    } <= forbidden
    assert governance["controls"]["audit_log_required"] is True
