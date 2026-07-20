"""Typed execution contracts shared by the rule engine core and templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ApprovedSql:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ProductConfigRef:
    config_id: str
    product_id: str
    path: Path
    policy_version: str | None = None


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    rule_template: str
    rule_template_id: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionRequest:
    rule_id: str
    complaint_id: str
    bundle_path: Path
    product_config_id: str | None = None
    rule_registry_path: str | None = None
    dataset_base_path: str | None = None
    sql_dir: str | None = None
    policy_basis_index_path: str | None = None
    max_customer_rows: int = 50
    audit_log_path: str | None = None
    dev_mode: bool | None = None


@dataclass(frozen=True)
class ExecutionContext:
    registry: Mapping[str, Any]
    rule: RuleDefinition
    product_config: Mapping[str, Any]
    product_config_ref: ProductConfigRef
    dataset_path: Path
    sql_dir: Path
    sql_files: tuple[ApprovedSql, ...]
    policy_basis_path: Path | None
    bundle_id: str
    artifact_verification: Mapping[str, Any]
    product_config_validation: Mapping[str, Any]
    data_contract_validation: Mapping[str, Any] | None


class RuleReport(TypedDict, total=False):
    execution_id: str
    executed_at: str
    runner_type: str
    handler_name: str
    rule_id: str
    rule_version: str
    rule_template_id: str
    rule_template: str
    harm_type: str
    product_id: str
    product_config_id: str
    product_policy_version: str
    data_contract_id: str
    complaint: JsonObject
    complainant_confirmed: bool
    complainant_customer_id: str
    complainant_harm_amount: int
    affected_customer_count: int
    affected_transaction_count: int
    unreported_customer_count: int
    unreported_transaction_count: int
    total_harm_amount: int
    error_type_counts: dict[str, int]
    decision: JsonObject
    policy_basis: list[JsonObject]
    audit: JsonObject


class AuditRecord(TypedDict, total=False):
    run_id: str
    timestamp: str
    bundle_id: str
    rule_id: str
    rule_version: str
    rule_template_id: str
    product_id: str
    product_config_id: str
    complaint_id: str
    input_snapshot_hash: str
    review_status: str
    automatic_refund_allowed: bool


class RuleHandler(Protocol):
    def run(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> RuleReport: ...


class TemplatePlugin(Protocol):
    rule_template: str

    def create_handler(self) -> RuleHandler: ...

    def validate_config(
        self,
        *,
        config: JsonObject,
        rule: JsonObject,
        registry: JsonObject,
        execution_date: str | None = None,
    ) -> JsonObject: ...
