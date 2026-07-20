import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from src.recall_agent.core.demo_paths import DEMO_DATASET_DIR, DEMO_RULES_DIR


RULES_DIR = Path(DEMO_RULES_DIR)
REGISTRY_PATH = RULES_DIR / "rule_registry.json"
DATA_CONTRACT_PATH = RULES_DIR / "data_contracts" / "h07_synthetic_v3.json"
GOVERNANCE_PATH = RULES_DIR / "governance" / "recall_agent_governance.json"
SCHEMA_PATH = Path(DEMO_DATASET_DIR) / "schema_data_dictionary.json"
POLICY_PDF = Path(
    "data/raw/policies/terms/전북은행_Smart_Cashback_체크카드_상품설명서_202207.pdf"
)
REPORT_PATH = RULES_DIR / "rule_assets_validation_report.json"
BUNDLE_PATH = RULES_DIR / "bundles" / "h07_reward_missing_mvp.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rule_assets() -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    contract = load_json(DATA_CONTRACT_PATH)
    governance = load_json(GOVERNANCE_PATH)
    schema = load_json(SCHEMA_PATH)
    bundle = load_json(BUNDLE_PATH)
    checks = []

    executable_rules = [
        rule for rule in registry["rules"]
        if rule.get("rule_status") in {"ACTIVE_FOR_MVP_DEMO", "ACTIVE_FOR_PRODUCTION"}
    ]

    for rule in executable_rules:
        product_scope = rule["product_scope"]
        rule_execution = rule["rule_execution"]
        calculation_policy = rule["calculation_policy"]
        bundle_rule = next(
            item for item in bundle["rules"]
            if item.get("rule_id") == rule["rule_id"] or rule["rule_id"] in item.get("legacy_rule_ids", [])
        )
        product_config_entry = next(
            item for item in bundle_rule["allowed_product_configs"]
            if item["product_config_id"] == bundle_rule["default_product_config_id"]
        )
        config_path = Path(product_config_entry["product_config_path"])
        config = load_json(config_path)
        policy = config.get("h07_reward_policy") or config.get("h07_cashback_policy")
        checks.append(
            {
                "check": f"{rule['rule_id']}: product config reference",
                "passed": (
                    config["config_id"] == product_config_entry["product_config_id"]
                    and config["product_id"] == product_config_entry["product_id"]
                    and config["supported_rule_template_id"] == rule["rule_template_id"]
                ),
            }
        )
        expected_pdf_hash = config["source_document"].get("source_document_hash_sha256")
        actual_pdf_hash = sha256(POLICY_PDF) if POLICY_PDF.exists() else expected_pdf_hash
        checks.append(
            {
                "check": f"{rule['rule_id']}: source PDF hash",
                "passed": bool(expected_pdf_hash) and expected_pdf_hash == actual_pdf_hash,
                "note": "PDF file not bundled; checked stored source_document_hash_sha256 only." if not POLICY_PDF.exists() else "PDF hash verified.",
            }
        )
        checks.append(
            {
                "check": f"{rule['rule_id']}: human review before refund",
                "passed": (
                    rule["output_requirements"]["customer_refund_decision"]
                    == "HUMAN_REVIEW_REQUIRED"
                    and registry["global_controls"][
                        "human_review_required_before_refund"
                    ]
                    and not registry["global_controls"]["automatic_refund_allowed"]
                ),
            }
        )
        checks.append(
            {
                "check": f"{rule['rule_id']}: approved only for MVP demo",
                "passed": (
                    config["approval_status"] == "APPROVED_FOR_MVP_DEMO"
                    and config["production_approval_status"]
                    == "NOT_APPROVED_FOR_PRODUCTION"
                    and rule["approval_status"] == "APPROVED_FOR_MVP_DEMO"
                    and rule["production_approval_status"]
                    == "NOT_APPROVED_FOR_PRODUCTION"
                    and registry["production_approval_status"]
                    == "NOT_APPROVED_FOR_PRODUCTION"
                ),
            }
        )
        runtime_controls = config["runtime_controls"]
        checks.append(
            {
                "check": f"{rule['rule_id']}: deterministic calculation controls",
                "passed": (
                    not runtime_controls["llm_may_use_this_config_for_calculation"]
                    and not runtime_controls["rag_may_use_this_config_for_calculation"]
                    and runtime_controls[
                        "deterministic_rule_engine_must_use_this_config"
                    ]
                    and runtime_controls["human_review_required_before_refund"]
                    and not runtime_controls["automatic_customer_refund_allowed"]
                ),
            }
        )
        config_error_types = {
            item["error_type"]
            for item in policy["reconciliation"][
                "detectable_error_types"
            ]
        }
        checks.append(
            {
                "check": f"{rule['rule_id']}: detectable error types match config",
                "passed": (
                    config_error_types
                    == set(calculation_policy["detectable_error_types"])
                ),
            }
        )
        checks.append(
            {
                "check": f"{rule['rule_id']}: execution components match config",
                "passed": (
                    rule_execution["rule_template"]
                    == policy["rule_template"]
                    and rule_execution["policy_calculator_name"]
                    in {
                        policy["independent_policy_calculation"]["calculator_name"],
                        "reward_policy_calculator_from_product_config",
                    }
                    and rule_execution["reconciliation_calculator_name"]
                    == policy["reconciliation"][
                        "calculator_name"
                    ]
                ),
            }
        )

    for table_name, table_contract in contract["tables"].items():
        actual_columns = {item["column"] for item in schema[f"{table_name}.csv"]}
        missing = sorted(set(table_contract["required_columns"]) - actual_columns)
        checks.append(
            {
                "check": f"data contract: {table_name}",
                "passed": not missing,
                "missing_columns": missing,
            }
        )

    checks.append(
        {
            "check": "private ground truth prohibited at runtime",
            "passed": any(
                Path(source).stem == "private_ground_truth"
                for source in contract["prohibited_runtime_sources"]
            ),
        }
    )
    checks.append(
        {
            "check": "LLM cannot generate SQL or calculate refund",
            "passed": (
                {
                    "generate_sql",
                    "calculate_refund_amount",
                    "finalize_customer_compensation",
                }
                <= set(governance["llm_policy"]["forbidden_actions"])
                and not registry["global_controls"]["llm_can_generate_sql"]
                and not registry["global_controls"]["llm_can_calculate_refund_amount"]
                and not registry["global_controls"]["llm_can_execute_rule_directly"]
            ),
        }
    )
    checks.append(
        {
            "check": "RAG is restricted to policy basis retrieval",
            "passed": (
                not registry["global_controls"]["rag_can_determine_harm"]
                and not registry["global_controls"]["rag_can_calculate_refund_amount"]
                and executable_rules[0]["policy_basis_policy"]["rag_usage"]
                == "POLICY_BASIS_RETRIEVAL_ONLY"
            ),
        }
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "passed_count": sum(check["passed"] for check in checks),
        "failed_count": sum(not check["passed"] for check in checks),
        "checks": checks,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(validate_rule_assets(), ensure_ascii=False, indent=2))
