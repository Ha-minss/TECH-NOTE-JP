"""Canonical portfolio-demo asset paths.

These constants keep the submission repository centered around one curated
demo layout under ``data/demo``.
"""

DEMO_ROOT = "data/demo"
DEMO_RULES_DIR = f"{DEMO_ROOT}/rules"
DEMO_BUNDLE_DIR = f"{DEMO_RULES_DIR}/bundles"
DEMO_BUNDLE_PATH = f"{DEMO_BUNDLE_DIR}/h07_reward_missing_mvp.json"
DEMO_RULE_REGISTRY_PATH = f"{DEMO_RULES_DIR}/rule_registry.json"
DEMO_DATA_CONTRACT_PATH = f"{DEMO_RULES_DIR}/data_contracts/h07_synthetic_v3.json"
DEMO_PRODUCT_CONFIG_PATH = (
    f"{DEMO_RULES_DIR}/product_policy_configs/"
    "JB_SMART_CASHBACK_CHECK__2022-07__v2.json"
)
DEMO_POLICY_RAG_DIR = f"{DEMO_ROOT}/policy_rag"
DEMO_POLICY_BASIS_REGISTRY_PATH = f"{DEMO_POLICY_RAG_DIR}/policy_basis_registry.jsonl"
DEMO_HARM_TYPE_BASIS_MAP_PATH = f"{DEMO_POLICY_RAG_DIR}/harm_type_basis_map.json"
DEMO_DATASET_DIR = f"{DEMO_ROOT}/datasets/jb_h07_synthetic_dataset_v3"
DEMO_AUDIT_LOG_PATH = f"{DEMO_ROOT}/audit/audit_log.jsonl"
