"""Pure incident-scope analysis for reward-missing reconciliation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IncidentAnalysis:
    complaint: dict[str, Any]
    complainant_customer_id: str
    complainant_rows: pd.DataFrame
    complainant_affected: pd.DataFrame
    same_scope: pd.DataFrame
    affected: pd.DataFrame
    unreported_affected: pd.DataFrame
    normal_exclusions: pd.DataFrame
    requires_review: pd.DataFrame
    incident_scope: dict[str, Any] | None


def _value(row: pd.Series, key: str) -> Any:
    value = row.get(key)
    return None if pd.isna(value) else value


def _incident_scope(rows: pd.DataFrame) -> dict[str, Any] | None:
    if rows.empty:
        return None
    candidates = rows
    if "reward_batch_id" in rows.columns and rows["reward_batch_id"].notna().any():
        candidates = rows[rows["reward_batch_id"].notna()]
    row = candidates.iloc[0]
    return {
        "reward_batch_id": _value(row, "reward_batch_id"),
        "processing_route": _value(row, "processing_route"),
        "product_id": _value(row, "product_id"),
        "product_config_id": _value(row, "product_config_id"),
        "purchase_month": _value(row, "purchase_month"),
        "scope_method": "SAME_REWARD_BATCH_ROUTE_PRODUCT_CONFIG_MONTH",
    }


def _same_scope(rows: pd.DataFrame, scope: dict[str, Any] | None) -> pd.DataFrame:
    if not scope:
        return rows.iloc[0:0].copy()
    result = rows.copy()
    for key in ("product_id", "product_config_id", "purchase_month", "processing_route"):
        value = scope.get(key)
        if value is not None and key in result.columns:
            result = result[result[key].astype(str) == str(value)]
    batch_id = scope.get("reward_batch_id")
    if batch_id is not None and "reward_batch_id" in result.columns:
        result = result[
            result["reward_batch_id"].astype(str).eq(str(batch_id))
            | result["reward_batch_id"].isna()
        ]
    return result


def analyze_incident(
    rows: pd.DataFrame,
    complaint: dict[str, Any],
) -> IncidentAnalysis:
    customer_id = str(complaint["customer_id"])
    complainant_rows = rows[rows["customer_id"].astype(str) == customer_id].copy()
    complainant_affected = complainant_rows[
        complainant_rows["is_affected"].fillna(False)
    ].copy()
    scope = _incident_scope(complainant_affected)
    same_scope = _same_scope(rows, scope)
    affected = same_scope[same_scope["is_affected"].fillna(False)].copy()
    if complainant_affected.empty:
        affected = complainant_affected.copy()
    unreported = affected[affected["customer_id"].astype(str) != customer_id].copy()
    normal = complainant_rows[
        complainant_rows.get(
            "is_normal_exclusion", pd.Series(False, index=complainant_rows.index)
        ).fillna(False)
    ].copy()
    review = same_scope[
        same_scope.get(
            "requires_human_review", pd.Series(False, index=same_scope.index)
        ).fillna(False)
    ].copy()
    return IncidentAnalysis(
        complaint=complaint,
        complainant_customer_id=customer_id,
        complainant_rows=complainant_rows,
        complainant_affected=complainant_affected,
        same_scope=same_scope,
        affected=affected,
        unreported_affected=unreported,
        normal_exclusions=normal,
        requires_review=review,
        incident_scope=scope,
    )
