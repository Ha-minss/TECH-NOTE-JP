"""Policy basis retriever for H01-H08.

This is the RAG entry point used by the generic rule runner.

Important design rule:
RAG retrieves policy/product basis only.
RAG must never calculate harm, refund amounts, SQL, or final customer compensation.

The current implementation is a deterministic basis-id registry lookup with
a small lexical search fallback. It is intentionally safe and dependency-light.
Later, the same public functions can call FAISS/BM25/RRF indexes without
changing the rule runner or handlers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.recall_agent.core.artifact_hash import resolve_project_path
from src.recall_agent.core.demo_paths import (
    DEMO_HARM_TYPE_BASIS_MAP_PATH,
    DEMO_POLICY_BASIS_REGISTRY_PATH,
)


DEFAULT_POLICY_BASIS_REGISTRY_PATH = DEMO_POLICY_BASIS_REGISTRY_PATH
DEFAULT_HARM_TYPE_BASIS_MAP_PATH = DEMO_HARM_TYPE_BASIS_MAP_PATH

RAG_USAGE = "POLICY_BASIS_RETRIEVAL_ONLY"


class PolicyBasisError(RuntimeError):
    pass



def load_harm_type_basis_map(
    path: str | Path = DEFAULT_HARM_TYPE_BASIS_MAP_PATH,
) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Harm type basis map not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [f"H{i:02d}" for i in range(1, 9) if f"H{i:02d}" not in data.get("harm_types", {})]
    if missing:
        raise PolicyBasisError(f"Harm type basis map is incomplete. Missing: {missing}")

    return data


def load_policy_basis_registry(
    path: str | Path = DEFAULT_POLICY_BASIS_REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Policy basis registry not found: {resolved}")

    registry: dict[str, dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            basis_id = row.get("basis_id")
            if not basis_id:
                raise PolicyBasisError(f"Missing basis_id at {resolved}:{lineno}")
            registry[str(basis_id)] = row

    return registry


def basis_ids_for_harm_type(
    harm_type: str,
    *,
    harm_type_basis_map_path: str | Path = DEFAULT_HARM_TYPE_BASIS_MAP_PATH,
) -> list[str]:
    """Return required basis IDs for H01-H08."""
    data = load_harm_type_basis_map(harm_type_basis_map_path)
    harm_type_key = harm_type.upper()
    if harm_type_key not in data["harm_types"]:
        raise PolicyBasisError(f"Unknown harm_type: {harm_type}. Expected H01-H08.")
    return list(data["harm_types"][harm_type_key]["required_policy_basis_ids"])


def basis_ids_from_rule_and_config(
    rule: dict[str, Any],
    product_config: dict[str, Any] | None = None,
    *,
    harm_type_basis_map_path: str | Path = DEFAULT_HARM_TYPE_BASIS_MAP_PATH,
) -> list[str]:
    """Collect required basis IDs in this order:
    1. explicit rule policy_basis_policy.required_policy_basis_ids
    2. rule calculation_policy.required_policy_basis_ids
    3. product_config required_policy_basis_ids
    4. product_config H07 h07_reward_policy/h07_cashback_policy.required_policy_basis_ids
    5. harm_type map H01-H08
    """
    ids = (
        rule.get("policy_basis_policy", {}).get("required_policy_basis_ids")
        or rule.get("calculation_policy", {}).get("required_policy_basis_ids")
        or rule.get("required_policy_basis_ids")
    )
    if ids:
        return list(ids)

    if product_config:
        ids = product_config.get("required_policy_basis_ids")
        if ids:
            return list(ids)

        # Generalized H07 config shape; h07_cashback_policy is legacy alias.
        ids = (
            product_config.get("h07_reward_policy", {}).get("required_policy_basis_ids")
            or product_config.get("h07_cashback_policy", {}).get("required_policy_basis_ids")
        )
        if ids:
            return list(ids)

    harm_type = rule.get("harm_type")
    if harm_type:
        return basis_ids_for_harm_type(str(harm_type), harm_type_basis_map_path=harm_type_basis_map_path)

    raise PolicyBasisError(
        "Cannot determine required policy basis IDs. "
        "Provide rule.required_policy_basis_ids, product_config.required_policy_basis_ids, or rule.harm_type."
    )


def _missing_basis_record(basis_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "POLICY_BASIS",
        "basis_id": basis_id,
        "title": basis_id,
        "retrieval_status": "MISSING_BASIS",
        "source_status": "MISSING_SOURCE",
        "evidence_level": "MISSING_SOURCE",
        "usable_as_demo_basis": False,
        "usable_as_real_policy_evidence": False,
        "requires_human_review": True,
        "usage": RAG_USAGE,
        "source_text": None,
        "note": "No basis record found. Do not use this as evidence. Human review is required.",
    }


def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("retrieval_status", "FOUND")
    out.setdefault("usage", RAG_USAGE)
    out.setdefault("requires_human_review", True)
    out.setdefault("usable_as_demo_basis", False)
    out.setdefault("usable_as_real_policy_evidence", False)

    # Explicit safety flags for downstream UI/audit.
    out["may_determine_harm"] = False
    out["may_calculate_refund"] = False
    out["may_generate_sql"] = False

    if not out.get("source_text"):
        out["retrieval_status"] = "SOURCE_REQUIRED_NOT_CONNECTED"
        out["requires_human_review"] = True

    return out


def retrieve_policy_basis(
    basis_ids: list[str],
    *,
    product_config: dict[str, Any] | None = None,  # kept for handler compatibility
    policy_basis_index_path: str | Path | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve basis records by basis_id.

    This is deterministic and safe. It does not perform calculations.

    Args:
        basis_ids: Required basis IDs from rule/product config/harm-type map.
        product_config: Accepted for compatibility; not required by this registry implementation.
        policy_basis_index_path: Optional JSONL registry path. Defaults to data/demo/policy_rag/policy_basis_registry.jsonl.
        strict: If True, raise when a basis is missing or source text is not connected.
    """
    registry_path = policy_basis_index_path or DEFAULT_POLICY_BASIS_REGISTRY_PATH
    registry = load_policy_basis_registry(registry_path)

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for basis_id in basis_ids:
        row = registry.get(str(basis_id))
        if row is None:
            missing = _missing_basis_record(str(basis_id))
            results.append(missing)
            errors.append(f"{basis_id}: missing basis record")
            continue

        normalized = _normalize_record(row)
        results.append(normalized)

        if strict and normalized.get("retrieval_status") != "FOUND":
            errors.append(f"{basis_id}: {normalized.get('retrieval_status')}")

    if strict and errors:
        raise PolicyBasisError("Policy basis retrieval failed in strict mode: " + "; ".join(errors))

    return results


_TOKEN_RE = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2}


def search_policy_basis(
    query: str,
    *,
    harm_type: str | None = None,
    top_k: int = 5,
    policy_basis_index_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Search the approved policy basis registry with deterministic lexical matching.

    Search results are evidence candidates only. They never determine harm,
    calculate refunds, or generate SQL.
    """
    registry = load_policy_basis_registry(policy_basis_index_path or DEFAULT_POLICY_BASIS_REGISTRY_PATH)
    query_tokens = _tokens(query)
    scored: list[tuple[int, dict[str, Any]]] = []

    for row in registry.values():
        if harm_type and str(row.get("harm_type", "")).upper() != harm_type.upper():
            continue

        haystack = " ".join(
            [
                str(row.get("basis_id", "")),
                str(row.get("title", "")),
                str(row.get("source_text", "")),
                " ".join(map(str, row.get("keywords", []))),
            ]
        )
        row_tokens = _tokens(haystack)
        score = len(query_tokens & row_tokens)

        # Prefer connected sources over placeholders when tie happens.
        if row.get("source_text"):
            score += 1

        if score > 0:
            normalized = _normalize_record(row)
            normalized["retrieval_backend"] = "POLICY_BASIS_REGISTRY_LEXICAL_FALLBACK"
            scored.append((score, normalized))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def validate_h01_h08_coverage(
    *,
    harm_type_basis_map_path: str | Path = DEFAULT_HARM_TYPE_BASIS_MAP_PATH,
    policy_basis_index_path: str | Path = DEFAULT_POLICY_BASIS_REGISTRY_PATH,
) -> dict[str, Any]:
    """Validate that every H01-H08 required basis ID has a registry record."""
    basis_map = load_harm_type_basis_map(harm_type_basis_map_path)
    registry = load_policy_basis_registry(policy_basis_index_path)

    result: dict[str, Any] = {
        "covered_harm_types": [],
        "missing_records": [],
        "source_required_not_connected": [],
        "connected_demo_or_source_records": [],
    }

    for harm_type, spec in basis_map["harm_types"].items():
        result["covered_harm_types"].append(harm_type)
        for basis_id in spec["required_policy_basis_ids"]:
            row = registry.get(basis_id)
            if row is None:
                result["missing_records"].append({"harm_type": harm_type, "basis_id": basis_id})
            elif row.get("source_status") == "SOURCE_REQUIRED_NOT_CONNECTED":
                result["source_required_not_connected"].append({"harm_type": harm_type, "basis_id": basis_id})
            else:
                result["connected_demo_or_source_records"].append({"harm_type": harm_type, "basis_id": basis_id})

    result["is_complete_for_id_lookup"] = len(result["missing_records"]) == 0
    result["is_complete_for_real_evidence"] = (
        len(result["missing_records"]) == 0
        and len(result["source_required_not_connected"]) == 0
    )
    return result


def main() -> None:
    coverage = validate_h01_h08_coverage()
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


