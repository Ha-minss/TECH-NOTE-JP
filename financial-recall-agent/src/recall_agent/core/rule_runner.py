"""Product-independent execution pipeline for approved rule templates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.recall_agent.core.artifact_hash import resolve_project_path
from src.recall_agent.core.audit_log import append_audit_log, build_audit_record
from src.recall_agent.core.bundle_loader import (
    get_bundle_product_config,
    get_bundle_rule,
    load_bundle,
    verify_bundle_static_artifacts,
)
from src.recall_agent.core.data_contract_validator import validate_data_contract
from src.recall_agent.core.demo_paths import DEMO_AUDIT_LOG_PATH
from src.recall_agent.core.models import (
    ApprovedSql,
    ExecutionContext,
    ExecutionRequest,
    ProductConfigRef,
    RuleDefinition,
    RuleReport,
)
from src.recall_agent.core.plugins import PluginRegistry
from src.recall_agent.core.registry_loader import (
    get_rule,
    get_rule_template,
    get_rule_template_id,
    load_json,
    load_rule_registry,
    validate_rule_for_execution,
)


PLACEHOLDER_RULE_STATUSES = {"DRAFT", "PLACEHOLDER_NOT_EXECUTABLE"}


def _dev_mode_enabled(request: ExecutionRequest) -> bool:
    if request.dev_mode is not None:
        return request.dev_mode
    return os.getenv("RECALL_AGENT_DEV_MODE", "").lower() in {"1", "true", "yes", "y"}


def _resolve_runtime_paths(
    request: ExecutionRequest,
    bundle: dict[str, Any],
    bundle_rule: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "rule_registry_path": str(bundle["rule_registry_path"]),
        "dataset_base_path": str(bundle["dataset_base_path"]),
        "sql_dir": str(bundle_rule["sql_dir"]),
        "policy_basis_index_path": str(bundle.get("policy_basis_registry_path") or ""),
    }
    overrides = {
        "rule_registry_path": request.rule_registry_path,
        "dataset_base_path": request.dataset_base_path,
        "sql_dir": request.sql_dir,
        "policy_basis_index_path": request.policy_basis_index_path,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if str(value) != paths[key] and not _dev_mode_enabled(request):
            raise ValueError(
                f"Runtime path override for {key} is not allowed outside developer mode."
            )
        paths[key] = str(value)
    return paths


def prepare_execution_context(
    request: ExecutionRequest,
    plugins: PluginRegistry,
) -> tuple[ExecutionContext, dict[str, Any]]:
    bundle = load_bundle(request.bundle_path)
    bundle_rule = get_bundle_rule(bundle, request.rule_id)
    config_ref_raw = get_bundle_product_config(
        bundle_rule,
        request.product_config_id or bundle_rule.get("default_product_config_id"),
    )
    config_id = str(config_ref_raw["product_config_id"])
    paths = _resolve_runtime_paths(request, bundle, bundle_rule)
    verification = verify_bundle_static_artifacts(
        bundle,
        rule_id=request.rule_id,
        product_config_id=config_id,
    )

    registry = load_rule_registry(paths["rule_registry_path"])
    rule_raw = get_rule(registry, request.rule_id)
    if rule_raw.get("rule_status") in PLACEHOLDER_RULE_STATUSES:
        raise ValueError(f"Rule {request.rule_id} is a placeholder/draft and cannot be executed.")
    validate_rule_for_execution(registry, rule_raw)

    rule_template = get_rule_template(rule_raw)
    plugin = plugins.require(rule_template)
    config_path = str(config_ref_raw["product_config_path"])
    product_config = load_json(config_path)
    config_validation = plugin.validate_config(
        config=product_config,
        rule=rule_raw,
        registry=registry,
    )

    contract_validation = None
    contract_path = bundle_rule.get("data_contract_path")
    if contract_path:
        contract_validation = validate_data_contract(
            contract=load_json(contract_path),
            dataset_base_path=paths["dataset_base_path"],
            expected_data_contract_id=rule_raw.get("rule_execution", {}).get("data_contract_id"),
        )

    sql_files = tuple(
        ApprovedSql(
            name=str(item.get("name") or Path(str(item["path"])).name),
            path=resolve_project_path(item["path"]),
            sha256=str(item["sha256"]),
        )
        for item in bundle_rule.get("approved_sql") or []
    )
    context = ExecutionContext(
        registry=registry,
        rule=RuleDefinition(
            rule_id=str(rule_raw["rule_id"]),
            rule_template=rule_template,
            rule_template_id=get_rule_template_id(rule_raw),
            raw=rule_raw,
        ),
        product_config=product_config,
        product_config_ref=ProductConfigRef(
            config_id=config_id,
            product_id=str(config_ref_raw.get("product_id") or product_config["product_id"]),
            path=resolve_project_path(config_path),
            policy_version=config_ref_raw.get("product_policy_version") or product_config.get("policy_version"),
        ),
        dataset_path=resolve_project_path(paths["dataset_base_path"]),
        sql_dir=resolve_project_path(paths["sql_dir"]),
        sql_files=sql_files,
        policy_basis_path=(
            resolve_project_path(paths["policy_basis_index_path"])
            if paths["policy_basis_index_path"]
            else None
        ),
        bundle_id=str(bundle["bundle_id"]),
        artifact_verification=verification,
        product_config_validation=config_validation,
        data_contract_validation=contract_validation,
    )
    return context, bundle


def execute_rule(
    request: ExecutionRequest,
    plugins: PluginRegistry,
) -> RuleReport:
    context, bundle = prepare_execution_context(request, plugins)
    handler = plugins.require(context.rule.rule_template).create_handler()
    report = handler.run(request, context)

    report.setdefault("audit", {})
    report["audit"].update(
        {
            "bundle_id": context.bundle_id,
            "bundle_hash": context.artifact_verification.get("bundle_hash"),
            "registry_hash": context.artifact_verification.get("registry_hash"),
            "product_config_hash": context.artifact_verification.get("config_hash"),
            "allowed_product_config_hashes": context.artifact_verification.get("allowed_product_config_hashes") or [],
            "selected_product_config_id": context.product_config_ref.config_id,
            "data_contract_hash": context.artifact_verification.get("data_contract_hash"),
            "sql_hashes": context.artifact_verification.get("sql_hashes") or [],
            "product_config_validation": dict(context.product_config_validation),
            "data_contract_validation": context.data_contract_validation,
            "runtime_path_override_allowed": _dev_mode_enabled(request),
        }
    )
    audit_record = build_audit_record(
        report=report,
        bundle_id=context.bundle_id,
        artifact_verification=dict(context.artifact_verification),
        product_config_validation=dict(context.product_config_validation),
        data_contract_validation=(
            dict(context.data_contract_validation)
            if context.data_contract_validation is not None
            else None
        ),
    )
    audit_path = request.audit_log_path or bundle.get(
        "audit_log_path", DEMO_AUDIT_LOG_PATH
    )
    log_path = append_audit_log(audit_record, path=audit_path)
    report["audit"]["audit_log_path"] = str(log_path)
    report["audit"]["audit_record"] = audit_record
    return report
