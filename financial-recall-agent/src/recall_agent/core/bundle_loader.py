"""Approved bundle loader for Financial Recall rule execution.

A bundle freezes the registry, dataset, allowed product configs, data contract
and SQL hashes used by a demo/PoC run. Normal application execution should
select a bundle and product config, not arbitrary filesystem paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.recall_agent.core.artifact_hash import (
    PROJECT_ROOT,
    resolve_project_path,
    sha256_file,
    sha256_json_canonical,
    sha256_json_file,
)
from src.recall_agent.core.demo_paths import DEMO_BUNDLE_DIR


DEFAULT_BUNDLE_DIR = DEMO_BUNDLE_DIR
APPROVED_BUNDLE_STATUSES = {"APPROVED_FOR_MVP_DEMO", "APPROVED_FOR_PRODUCTION"}


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Bundle JSON not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    if bundle.get("bundle_type") != "RECALL_RULE_BUNDLE":
        raise ValueError(
            f"Invalid bundle_type={bundle.get('bundle_type')!r}; "
            "expected 'RECALL_RULE_BUNDLE'."
        )
    if bundle.get("bundle_status") not in APPROVED_BUNDLE_STATUSES:
        raise ValueError(
            f"Bundle {bundle.get('bundle_id')} is not approved for execution: "
            f"bundle_status={bundle.get('bundle_status')!r}"
        )
    return bundle


def bundle_fingerprint(bundle: dict[str, Any]) -> str:
    """Return a canonical hash of the bundle content.

    Excludes optional stored hash fields so the bundle can be fingerprinted before
    or after publication without self-reference issues.
    """
    clean = {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_hash", "bundle_sha256_canonical"}
    }
    return sha256_json_canonical(clean)


def list_approved_bundles(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> list[dict[str, Any]]:
    """List approved bundle metadata for UI selection."""
    base = resolve_project_path(bundle_dir)
    if not base.exists():
        return []
    bundles: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.json")):
        try:
            bundle = load_bundle(path)
        except Exception:
            continue
        bundles.append(
            {
                "bundle_id": bundle.get("bundle_id"),
                "bundle_name": bundle.get("bundle_name") or bundle.get("bundle_id"),
                "bundle_status": bundle.get("bundle_status"),
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return bundles


def _rule_ids_for(bundle_rule: dict[str, Any]) -> set[str]:
    ids = {str(bundle_rule.get("rule_id"))}
    ids.update(str(x) for x in bundle_rule.get("legacy_rule_ids", []) if x)
    return ids


def _index_rules(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for rule in bundle.get("rules", []):
        for rid in _rule_ids_for(rule):
            indexed[rid] = rule
    return indexed


def get_bundle_rule(bundle: dict[str, Any], rule_id: str) -> dict[str, Any]:
    rules = _index_rules(bundle)
    if rule_id not in rules:
        raise ValueError(
            f"Rule {rule_id!r} is not part of approved bundle "
            f"{bundle.get('bundle_id')!r}. Available: {sorted(rules)}"
        )
    return rules[rule_id]


def get_bundle_product_config(
    bundle_rule: dict[str, Any],
    product_config_id: str | None = None,
) -> dict[str, Any]:
    """Select a product config instance allowed by a generic rule template.

    The old MVP shape had exactly one product_config_path at the rule level.
    The generalized shape has allowed_product_configs. Support both so legacy
    tests/direct calls keep working, while normal execution uses the new list.
    """
    allowed = bundle_rule.get("allowed_product_configs") or []
    if not allowed and bundle_rule.get("product_config_path"):
        allowed = [
            {
                "product_config_id": bundle_rule.get("product_config_id"),
                "product_id": bundle_rule.get("product_id"),
                "product_config_path": bundle_rule.get("product_config_path"),
                "product_config_sha256_canonical": bundle_rule.get("product_config_sha256_canonical"),
                "product_policy_version": bundle_rule.get("product_policy_version"),
            }
        ]

    if not allowed:
        raise ValueError(
            f"Bundle rule {bundle_rule.get('rule_id')} has no allowed_product_configs."
        )

    selected_id = product_config_id or bundle_rule.get("default_product_config_id")
    if selected_id:
        for config in allowed:
            if config.get("product_config_id") == selected_id:
                return config
        available = [config.get("product_config_id") for config in allowed]
        raise ValueError(
            f"product_config_id={selected_id!r} is not allowed for rule "
            f"{bundle_rule.get('rule_id')}. Available: {available}"
        )

    if len(allowed) == 1:
        return allowed[0]

    raise ValueError(
        f"Multiple product configs are allowed for rule {bundle_rule.get('rule_id')}; "
        "pass product_config_id explicitly."
    )


def list_allowed_product_configs(bundle_rule: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = bundle_rule.get("allowed_product_configs") or []
    if not allowed and bundle_rule.get("product_config_path"):
        allowed = [get_bundle_product_config(bundle_rule)]
    return allowed


def verify_bundle_static_artifacts(
    bundle: dict[str, Any],
    *,
    rule_id: str | None = None,
    product_config_id: str | None = None,
) -> dict[str, Any]:
    """Verify artifact hashes recorded in a bundle.

    Checks files whose hashes should not change during execution: registry, data
    contract, selected/allowed product config and approved SQL files.
    """
    verification: dict[str, Any] = {
        "bundle_id": bundle.get("bundle_id"),
        "bundle_hash": bundle_fingerprint(bundle),
        "registry_hash": None,
        "data_contract_hash": None,
        "config_hash": None,
        "selected_product_config_id": product_config_id,
        "allowed_product_config_hashes": [],
        "sql_hashes": [],
    }

    registry_path = bundle.get("rule_registry_path")
    expected_registry_hash = bundle.get("rule_registry_sha256_canonical")
    if registry_path and expected_registry_hash:
        actual = sha256_json_file(registry_path)
        if actual != expected_registry_hash:
            raise ValueError(
                "Rule Registry hash mismatch. "
                f"expected={expected_registry_hash}, actual={actual}, path={registry_path}"
            )
        verification["registry_hash"] = actual

    selected_rules = bundle.get("rules", [])
    if rule_id is not None:
        selected_rules = [get_bundle_rule(bundle, rule_id)]

    for bundle_rule in selected_rules:
        data_contract_path = bundle_rule.get("data_contract_path")
        expected_data_contract_hash = bundle_rule.get("data_contract_sha256_canonical")
        if data_contract_path and expected_data_contract_hash:
            actual = sha256_json_file(data_contract_path)
            if actual != expected_data_contract_hash:
                raise ValueError(
                    "Data Contract hash mismatch. "
                    f"expected={expected_data_contract_hash}, actual={actual}, "
                    f"path={data_contract_path}"
                )
            verification["data_contract_hash"] = actual

        config_entries: list[dict[str, Any]]
        if product_config_id:
            config_entries = [get_bundle_product_config(bundle_rule, product_config_id)]
        else:
            config_entries = list_allowed_product_configs(bundle_rule)

        for config in config_entries:
            config_path = config.get("product_config_path") or config.get("path")
            expected_config_hash = config.get("product_config_sha256_canonical")
            if config_path and expected_config_hash:
                actual = sha256_json_file(config_path)
                if actual != expected_config_hash:
                    raise ValueError(
                        "Product Config hash mismatch. "
                        f"expected={expected_config_hash}, actual={actual}, path={config_path}"
                    )
                item = {
                    "product_config_id": config.get("product_config_id"),
                    "product_id": config.get("product_id"),
                    "path": config_path,
                    "sha256": actual,
                }
                verification["allowed_product_config_hashes"].append(item)
                # config_hash remains the selected/default config hash for legacy consumers.
                if (
                    product_config_id is None
                    and config.get("product_config_id") == bundle_rule.get("default_product_config_id")
                ) or config.get("product_config_id") == product_config_id or verification["config_hash"] is None:
                    verification["config_hash"] = actual
                    verification["selected_product_config_id"] = config.get("product_config_id")

        for sql in bundle_rule.get("approved_sql", []):
            sql_path = sql.get("path")
            expected_sql_hash = sql.get("sha256")
            if not sql_path or not expected_sql_hash:
                raise ValueError(f"Approved SQL entry is incomplete: {sql}")
            actual = sha256_file(sql_path)
            if actual != expected_sql_hash:
                raise ValueError(
                    "Approved SQL hash mismatch. "
                    f"expected={expected_sql_hash}, actual={actual}, path={sql_path}"
                )
            verification["sql_hashes"].append(
                {
                    "name": sql.get("name") or Path(sql_path).name,
                    "path": sql_path,
                    "sha256": actual,
                }
            )

    return verification
