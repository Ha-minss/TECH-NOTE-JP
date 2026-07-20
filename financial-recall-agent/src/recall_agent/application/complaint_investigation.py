from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.recall_agent.application.complaint_router import ComplaintRouter
from src.recall_agent.application.router_schema import RouterRoute
from src.recall_agent.interfaces.cli.h07_reward_missing_demo import run_h07_demo


SUPPORTED_H07_PRODUCTS = {
    "JB_SMART_CASHBACK_CHECK",
}


@dataclass(frozen=True)
class ProductVerificationResult:
    customer_id: str | None
    claimed_product_id: str | None
    active_product_ids: list[str]
    supported_h07_product_ids: list[str]
    passed: bool
    reason_code: str
    reason: str


@dataclass(frozen=True)
class RuleExecutionResult:
    attempted: bool
    service: str | None
    report: dict[str, Any] | None
    reason: str


@dataclass(frozen=True)
class ComplaintInvestigationResult:
    complaint_id: str
    customer_id: str | None
    complaint_text: str
    router_result: dict[str, Any]
    router_schema_valid: bool
    product_verification: dict[str, Any]
    should_run_h07_rule: bool
    rule_execution: dict[str, Any]


def none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_complaint_by_id(complaints_path: Path, complaint_id: str) -> dict[str, Any]:
    rows = load_csv_rows(complaints_path)
    for row in rows:
        if row.get("complaint_id") == complaint_id:
            return row

    available = [row.get("complaint_id") for row in rows[:10]]
    raise ValueError(
        f"complaint_id not found: {complaint_id}. "
        f"First available complaint_ids: {available}"
    )


def normalize_complaint_record(row: Mapping[str, Any]) -> dict[str, Any]:
    complaint_text = (
        none_if_blank(row.get("complaint_text"))
        or none_if_blank(row.get("narrative_ko"))
        or none_if_blank(row.get("narrative"))
        or ""
    )

    complaint_id = none_if_blank(row.get("complaint_id"))
    customer_id = none_if_blank(row.get("customer_id"))

    return {
        "case_id": complaint_id,
        "complaint_id": complaint_id,
        "customer_id": customer_id,
        "channel": none_if_blank(row.get("channel")),
        "product_id_claimed": none_if_blank(row.get("product_id_claimed")),
        "complaint_text": complaint_text,
        "llm_allowed_input_fields": [
            "channel",
            "product_id_claimed",
            "complaint_text",
        ],
    }


def verify_customer_product(
    *,
    customer_id: str | None,
    claimed_product_id: str | None,
    card_contracts_path: Path,
    supported_h07_products: set[str] | None = None,
) -> ProductVerificationResult:
    supported = supported_h07_products or SUPPORTED_H07_PRODUCTS

    if not customer_id:
        return ProductVerificationResult(
            customer_id=None,
            claimed_product_id=claimed_product_id,
            active_product_ids=[],
            supported_h07_product_ids=[],
            passed=False,
            reason_code="MISSING_CUSTOMER_ID",
            reason="Cannot verify product because customer_id is missing.",
        )

    contracts = load_csv_rows(card_contracts_path)
    customer_contracts = [
        row for row in contracts if none_if_blank(row.get("customer_id")) == customer_id
    ]

    if not customer_contracts:
        return ProductVerificationResult(
            customer_id=customer_id,
            claimed_product_id=claimed_product_id,
            active_product_ids=[],
            supported_h07_product_ids=[],
            passed=False,
            reason_code="NO_CONTRACT_FOUND",
            reason="No card contract found for this customer.",
        )

    active_contracts = [
        row
        for row in customer_contracts
        if str(row.get("status", "")).strip().upper() in {"ACTIVE", "OPEN", "NORMAL"}
    ]

    active_product_ids = sorted(
        {
            str(row.get("product_id")).strip()
            for row in active_contracts
            if none_if_blank(row.get("product_id"))
        }
    )

    supported_product_ids = sorted(
        product_id for product_id in active_product_ids if product_id in supported
    )

    if supported_product_ids:
        return ProductVerificationResult(
            customer_id=customer_id,
            claimed_product_id=claimed_product_id,
            active_product_ids=active_product_ids,
            supported_h07_product_ids=supported_product_ids,
            passed=True,
            reason_code="PASS_SUPPORTED_H07_PRODUCT",
            reason="Customer has an active supported H07 product contract.",
        )

    if claimed_product_id in supported:
        return ProductVerificationResult(
            customer_id=customer_id,
            claimed_product_id=claimed_product_id,
            active_product_ids=active_product_ids,
            supported_h07_product_ids=[],
            passed=False,
            reason_code="CLAIMED_SUPPORTED_PRODUCT_BUT_NO_ACTIVE_CONTRACT",
            reason=(
                "Complaint claims a supported H07 product, but no active matching "
                "contract was found in card_contracts."
            ),
        )

    return ProductVerificationResult(
        customer_id=customer_id,
        claimed_product_id=claimed_product_id,
        active_product_ids=active_product_ids,
        supported_h07_product_ids=[],
        passed=False,
        reason_code="NO_SUPPORTED_H07_PRODUCT",
        reason="Customer has no active supported H07 product contract.",
    )


def should_enter_product_verification(route: RouterRoute) -> bool:
    return route in {
        RouterRoute.H07_CANDIDATE,
        RouterRoute.NEEDS_PRODUCT_VERIFICATION,
    }


def should_run_h07_rule(
    *,
    router_route: RouterRoute,
    product_verification: ProductVerificationResult,
) -> bool:
    if router_route != RouterRoute.H07_CANDIDATE:
        return False
    return product_verification.passed


def execute_h07_rule(
    complaint_id: str | None,
    dataset_dir: Path | None = None,
) -> RuleExecutionResult:
    if not complaint_id:
        return RuleExecutionResult(
            attempted=False,
            service=None,
            report=None,
            reason="Skipped H07 rule execution because complaint_id is missing.",
        )

    report = run_h07_demo(
        complaint_id,
        dataset_dir=str(dataset_dir) if dataset_dir is not None else None,
        dev_mode=dataset_dir is not None,
    )
    return RuleExecutionResult(
        attempted=True,
        service="run_h07_demo",
        report=report,
        reason="Executed the approved H07 flow through the in-process demo service.",
    )


def investigate_complaint(
    *,
    complaint_id: str,
    router: ComplaintRouter,
    complaints_path: Path,
    card_contracts_path: Path,
    execute_rule: bool = True,
    dataset_dir: Path | None = None,
) -> ComplaintInvestigationResult:
    raw_complaint = load_complaint_by_id(complaints_path, complaint_id)
    complaint = normalize_complaint_record(raw_complaint)

    router_output = router.route(complaint)
    router_result = router_output.result

    if should_enter_product_verification(router_result.route):
        product_verification = verify_customer_product(
            customer_id=complaint.get("customer_id"),
            claimed_product_id=complaint.get("product_id_claimed"),
            card_contracts_path=card_contracts_path,
        )
    else:
        product_verification = ProductVerificationResult(
            customer_id=complaint.get("customer_id"),
            claimed_product_id=complaint.get("product_id_claimed"),
            active_product_ids=[],
            supported_h07_product_ids=[],
            passed=False,
            reason_code="ROUTER_DID_NOT_ENTER_H07_OR_VERIFICATION",
            reason="Router classified the complaint outside the H07/verification path.",
        )

    run_rule = should_run_h07_rule(
        router_route=router_result.route,
        product_verification=product_verification,
    )

    if execute_rule and run_rule:
        rule_execution = execute_h07_rule(
            complaint_id,
            dataset_dir=dataset_dir or complaints_path.parent,
        )
    else:
        rule_execution = RuleExecutionResult(
            attempted=False,
            service=None,
            report=None,
            reason=(
                "Skipped H07 rule execution because router/product verification "
                "did not pass."
            ),
        )

    return ComplaintInvestigationResult(
        complaint_id=complaint_id,
        customer_id=complaint.get("customer_id"),
        complaint_text=complaint.get("complaint_text") or "",
        router_result=router_result.model_dump(mode="json"),
        router_schema_valid=router_output.schema_valid,
        product_verification=asdict(product_verification),
        should_run_h07_rule=run_rule,
        rule_execution=asdict(rule_execution),
    )


def save_investigation_result(path: Path, result: ComplaintInvestigationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
