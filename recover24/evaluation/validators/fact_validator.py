"""Fact-preservation validator for generated financial-incident narratives."""

from __future__ import annotations

from typing import Any

from evaluation.dataset_loader import GoldCase
from evaluation.validators.claim_extractor import extract_claims_with_provider, extract_numeric_amounts_krw
from evaluation.validators.order_checker import order_score
from evaluation.validators.status_rules import judge_status_claim
from recover24.providers.base import LLMProvider


SAFETY_BLOCKING_LABELS = {"contradicted", "unsupported"}


def format_amount_krw(value: Any) -> str:
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return "금액 미확인"
    return f"{amount:,}원"


def validate_generated_record(
    case: GoldCase,
    record: dict[str, Any],
    claim_provider: LLMProvider | None = None,
) -> dict[str, Any]:
    text = join_outputs(record.get("outputs", {}))
    claims = extract_claims_with_provider(text, case, claim_provider)
    return validate_text(
        case,
        text,
        claims,
        claim_evaluation_enabled=claim_provider is not None,
    )


def validate_text(
    case: GoldCase,
    text: str,
    claims: dict[str, Any] | None = None,
    *,
    claim_evaluation_enabled: bool | None = None,
) -> dict[str, Any]:
    claims = claims or extract_claims_with_provider(text, case, None)
    if claim_evaluation_enabled is None:
        claim_evaluation_enabled = any(
            claims.get(key)
            for key in ("status_claims", "unsupported_claims", "event_order", "supported_fact_ids")
        )
    facts = case.structured_facts
    output_amounts = set(extract_numeric_amounts_krw(text))
    allowed_amounts = _allowed_amounts(case)
    required_amounts = _required_amounts(case)

    amount_errors: list[dict[str, Any]] = []
    for amount in required_amounts:
        if amount not in output_amounts:
            amount_errors.append({"type": "missing_required_amount", "amount_krw": amount})
    for amount in output_amounts:
        if allowed_amounts and amount not in allowed_amounts:
            amount_errors.append({"type": "unsupported_amount", "amount_krw": amount})

    required_results = _score_required_facts(case, text, claims)
    status_results = _score_status_claims(facts, claims)
    order = order_score(case.expected_event_order, list(claims.get("event_order", [])))
    unsupported_claims = list(claims.get("unsupported_claims", []))

    blocking_errors = []
    blocking_errors.extend(amount_errors)
    blocking_errors.extend([r for r in status_results if r["label"] in SAFETY_BLOCKING_LABELS])
    blocking_errors.extend({"type": "unsupported_claim", **u} for u in unsupported_claims)

    return {
        "safe_to_use": len(blocking_errors) == 0,
        "amounts": {
            "output_amounts_krw": sorted(output_amounts),
            "allowed_amounts_krw": sorted(allowed_amounts),
            "required_amounts_krw": sorted(required_amounts),
            "errors": amount_errors,
        },
        "required_facts": required_results,
        "status_claims": status_results,
        "unsupported_claims": unsupported_claims,
        "event_order": order,
        "blocking_errors": blocking_errors,
        "metric_availability": {
            "status_claims": claim_evaluation_enabled,
            "unsupported_claims": claim_evaluation_enabled,
            "event_order": claim_evaluation_enabled and bool(claims.get("event_order")),
        },
    }


def join_outputs(outputs: dict[str, Any]) -> str:
    return "\n".join(str(value or "") for value in outputs.values()).strip()


def _allowed_amounts(case: GoldCase) -> set[int]:
    amounts: set[int] = set()
    for key, value in case.structured_facts.items():
        if key.endswith("amount_krw") or key == "amount_krw":
            try:
                amounts.add(int(value))
            except (TypeError, ValueError):
                pass
    amounts.update(extract_numeric_amounts_krw(case.raw_statement))
    return amounts


def _required_amounts(case: GoldCase) -> set[int]:
    amounts: set[int] = set()
    facts = case.structured_facts
    for field in case.required_fact_ids:
        if field.endswith("amount_krw") or field == "amount_krw":
            value = facts.get(field)
            try:
                amounts.add(int(value))
            except (TypeError, ValueError):
                pass
    return amounts


def _score_required_facts(case: GoldCase, text: str, claims: dict[str, Any]) -> dict[str, Any]:
    included: list[str] = []
    missing: list[str] = []
    supported_by_claim_extractor = set(claims.get("supported_fact_ids", []))

    for field in case.required_fact_ids:
        if field in supported_by_claim_extractor or _field_present_by_exact_value(case, field, text):
            included.append(field)
        else:
            missing.append(field)
    total = len(case.required_fact_ids)
    return {
        "included": included,
        "missing": missing,
        "total": total,
        "score": len(included) / total if total else 1.0,
    }


def _field_present_by_exact_value(case: GoldCase, field: str, text: str) -> bool:
    facts = case.structured_facts
    if field not in facts:
        return False
    value = facts.get(field)
    if value in (None, "", "unknown"):
        return False
    if field.endswith("amount_krw") or field == "amount_krw":
        try:
            return int(value) in set(extract_numeric_amounts_krw(text))
        except (TypeError, ValueError):
            return False
    aliases = case.fact_aliases.get(field, [])
    if aliases and any(alias and alias in text for alias in aliases):
        return True
    if isinstance(value, str) and value and value in text:
        return True
    if isinstance(value, bool):
        # Boolean facts usually need a claim extractor or aliases to be fairly scored.
        return field in case.fact_aliases and any(alias in text for alias in case.fact_aliases[field])
    return False


def _score_status_claims(facts: dict[str, Any], claims: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for claim in claims.get("status_claims", []):
        field = str(claim.get("field", ""))
        claimed = str(claim.get("claimed_value", ""))
        evidence = str(claim.get("evidence_text", ""))
        judgement = judge_status_claim(field, claimed, facts, evidence)
        results.append({
            "field": judgement.field,
            "gold_value": judgement.gold_value,
            "claimed_value": judgement.claimed_value,
            "label": judgement.label,
            "evidence_text": judgement.evidence_text,
        })
    return results
