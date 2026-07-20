"""Build MVP A/B/C evaluation datasets from existing Recover24 gold cases.

Outputs:
- evaluation/gold/master_cases.jsonl
- evaluation/normalization/dataset.jsonl   # A, 20 rows
- evaluation/consistency/dataset.jsonl     # C, 20 rows
- evaluation/narrative/dataset.jsonl       # B, 20 rows

This script derives small MVP datasets from the existing clean gold cases.
It is not a full benchmark generator. It creates 20 rows per track for a
portfolio-grade MVP evaluation flow.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from evaluation.normalization.renderer_check import render_expected_fields


ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "evaluation"


def main() -> None:
    gold_cases = load_existing_gold_cases()
    if len(gold_cases) < 5:
        raise RuntimeError(f"Not enough gold cases found: {len(gold_cases)}")

    selected = select_master_cases(gold_cases, limit=20)

    gold_dir = EVAL_DIR / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(gold_dir / "master_cases.jsonl", selected)

    normalization_rows = build_normalization_rows(selected, count=20)
    consistency_rows = build_consistency_rows(selected, count=20)
    narrative_rows = build_narrative_rows(selected, count=20)

    backup_and_write(EVAL_DIR / "normalization" / "dataset.jsonl", normalization_rows)
    backup_and_write(EVAL_DIR / "consistency" / "dataset.jsonl", consistency_rows)
    backup_and_write(EVAL_DIR / "narrative" / "dataset.jsonl", narrative_rows)

    stats = {
        "master_cases": len(selected),
        "normalization_cases": len(normalization_rows),
        "consistency_cases": len(consistency_rows),
        "narrative_cases": len(narrative_rows),
        "source": "existing evaluation gold cases",
    }
    (gold_dir / "dataset_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def load_existing_gold_cases() -> list[dict[str, Any]]:
    candidates = [
        EVAL_DIR / "dataset" / "dev.jsonl",
        EVAL_DIR / "dataset" / "test.jsonl",
        EVAL_DIR / "dataset" / "challenge.jsonl",
        EVAL_DIR / "legacy" / "dataset" / "dev.jsonl",
        EVAL_DIR / "legacy" / "dataset" / "test.jsonl",
        EVAL_DIR / "legacy" / "dataset" / "challenge.jsonl",
    ]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Prefer challenge rows first so MVP consistency cases include risky/conflict-style cases.
    ordered = sorted(
        [path for path in candidates if path.exists()],
        key=lambda path: (0 if "challenge" in path.name else 1 if "dev" in path.name else 2, str(path)),
    )

    for path in ordered:
        for row in read_jsonl(path):
            case_id = str(row.get("case_id", ""))
            if not case_id or case_id in seen:
                continue
            if not isinstance(row.get("structured_facts"), dict):
                continue
            rows.append(row)
            seen.add(case_id)

    return rows


def select_master_cases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected = rows[:limit]
    normalized: list[dict[str, Any]] = []

    for idx, row in enumerate(selected, start=1):
        out = dict(row)
        out["parent_gold_case_id"] = row.get("case_id", f"GOLD_{idx:03d}")
        out["gold_index"] = idx
        normalized.append(out)

    return normalized


def build_normalization_rows(master_cases: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx in range(count):
        case = master_cases[idx % len(master_cases)]
        facts = case.get("structured_facts", {})
        amount = known_amount(facts, fallback=1_000_000 + idx * 100_000)
        date = synthetic_date(idx)

        amount_input = amount_variant(amount, idx)
        date_input = date_variant(date, idx)

        input_payload: dict[str, Any] = {
            "damage_amount": amount_input,
            "incident_date": date_input,
        }

        expected_canonical: dict[str, Any] = {
            "damage_amount_krw": amount,
            "incident_date": date,
        }

        # Add statuses when they are known and supported by the normalizer.
        police = normalized_status(facts.get("police_status"), {"reported", "not_reported", "planned"})
        if police:
            input_payload["police_status"] = status_input("police_status", police)
            expected_canonical["police_status"] = police

        freeze = normalized_status(facts.get("freeze_status"), {"requested", "attempted_but_failed", "completed"})
        if freeze:
            input_payload["freeze_status"] = status_input("freeze_status", freeze)
            expected_canonical["freeze_status"] = freeze

        refund = normalized_status(facts.get("refund_status"), {"not_started", "requested", "completed"})
        if refund:
            input_payload["refund_status"] = status_input("refund_status", refund)
            expected_canonical["refund_status"] = refund

        expected_rendered = render_expected_fields(expected_canonical)

        rows.append(
            {
                "case_id": f"NORM_MVP_{idx + 1:03d}",
                "parent_gold_case_id": case["parent_gold_case_id"],
                "input": input_payload,
                "required_fields": ["damage_amount_krw", "incident_date"],
                "expected_canonical": expected_canonical,
                "expected_rendered": expected_rendered,
            }
        )

    return rows


def build_consistency_rows(master_cases: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    patterns = [
        "no_conflict_amount",
        "amount_conflict",
        "police_conflict",
        "police_enrichment",
        "freeze_conflict",
        "refund_conflict",
        "date_conflict",
        "recipient_account_conflict",
        "no_conflict_status",
        "multi_conflict",
    ]

    for idx in range(count):
        case = master_cases[idx % len(master_cases)]
        facts = case.get("structured_facts", {})
        amount = known_amount(facts, fallback=2_000_000 + idx * 100_000)
        date = synthetic_date(idx)
        pattern = patterns[idx % len(patterns)]

        form_facts: dict[str, Any]
        statement_facts: dict[str, Any]
        expected_conflict_fields: list[str]
        raw_statement: str

        if pattern == "no_conflict_amount":
            form_facts = {"damage_amount_krw": amount}
            statement_facts = {"damage_amount_krw": amount}
            expected_conflict_fields = []
            raw_statement = f"총 {amount // 10000}만원을 송금했습니다."

        elif pattern == "amount_conflict":
            other = alter_amount(amount)
            form_facts = {"damage_amount_krw": amount}
            statement_facts = {"damage_amount_krw": other}
            expected_conflict_fields = ["damage_amount_krw"]
            raw_statement = f"총 {other // 10000}만원을 송금했습니다."

        elif pattern == "police_conflict":
            form_facts = {"police_status": "not_reported"}
            statement_facts = {"police_status": "reported"}
            expected_conflict_fields = ["police_status"]
            raw_statement = "피해 사실을 알게 된 뒤 경찰에 신고했습니다."

        elif pattern == "police_enrichment":
            form_facts = {"police_status": "unknown"}
            statement_facts = {"police_status": "reported"}
            expected_conflict_fields = []
            raw_statement = "피해 사실을 알게 된 뒤 경찰에 신고했습니다."

        elif pattern == "freeze_conflict":
            form_facts = {"freeze_status": "completed"}
            statement_facts = {"freeze_status": "attempted_but_failed"}
            expected_conflict_fields = ["freeze_status"]
            raw_statement = "은행에 지급정지를 시도했으나 실패했습니다."

        elif pattern == "refund_conflict":
            form_facts = {"refund_status": "not_started"}
            statement_facts = {"refund_status": "completed"}
            expected_conflict_fields = ["refund_status"]
            raw_statement = "피해금 환급은 완료되었습니다."

        elif pattern == "date_conflict":
            other_date = next_date(date)
            form_facts = {"incident_date": date}
            statement_facts = {"incident_date": other_date}
            expected_conflict_fields = ["incident_date"]
            raw_statement = f"{other_date.replace('-', '.')}에 피해가 발생했습니다."

        elif pattern == "recipient_account_conflict":
            form_facts = {"recipient_account_known": True}
            statement_facts = {"recipient_account_known": False}
            expected_conflict_fields = ["recipient_account_known"]
            raw_statement = "수취 계좌는 모릅니다."

        elif pattern == "no_conflict_status":
            form_facts = {
                "damage_amount_krw": amount,
                "police_status": "reported",
                "freeze_status": "requested",
            }
            statement_facts = {
                "damage_amount_krw": amount,
                "police_status": "reported",
                "freeze_status": "requested",
            }
            expected_conflict_fields = []
            raw_statement = f"총 {amount // 10000}만원 피해를 입었고 경찰에 신고했으며 지급정지를 요청했습니다."

        else:  # multi_conflict
            other = alter_amount(amount)
            form_facts = {
                "damage_amount_krw": amount,
                "police_status": "not_reported",
            }
            statement_facts = {
                "damage_amount_krw": other,
                "police_status": "reported",
            }
            expected_conflict_fields = ["damage_amount_krw", "police_status"]
            raw_statement = f"총 {other // 10000}만원을 송금했고 경찰에 신고했습니다."

        rows.append(
            {
                "case_id": f"CONS_MVP_{idx + 1:03d}",
                "parent_gold_case_id": case["parent_gold_case_id"],
                "pattern": pattern,
                "form_facts": form_facts,
                "statement_facts": statement_facts,
                "expected_statement_facts": statement_facts,
                "raw_statement": raw_statement,
                "expected_conflict_fields": expected_conflict_fields,
                "expected_can_generate_document": len(expected_conflict_fields) == 0,
            }
        )

    return rows


def build_narrative_rows(master_cases: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx in range(count):
        case = master_cases[idx % len(master_cases)]
        facts = dict(case.get("structured_facts", {}))
        amount = known_amount(facts, fallback=3_000_000 + idx * 100_000)
        facts.setdefault("amount_krw", amount)

        required_elements = required_elements_from_case(case, amount=amount)
        generated_text = safe_generated_text(required_elements, amount=amount)

        # Keep a couple of blocked rows so narrative runner proves it skips unsafe cases.
        can_generate_document = idx not in {8, 17}

        rows.append(
            {
                "case_id": f"NARR_MVP_{idx + 1:03d}",
                "parent_gold_case_id": case["parent_gold_case_id"],
                "can_generate_document": can_generate_document,
                "canonical_case": facts,
                "raw_statement": case.get("raw_statement", ""),
                "generated_text": generated_text,
                "required_elements": required_elements,
            }
        )

    return rows


def required_elements_from_case(case: dict[str, Any], *, amount: int) -> list[dict[str, Any]]:
    required_facts = case.get("required_facts")
    elements: list[dict[str, Any]] = []

    if isinstance(required_facts, list) and required_facts:
        for fact in required_facts[:6]:
            if not isinstance(fact, dict):
                continue

            fact_id = str(fact.get("fact_id") or fact.get("field") or f"REQ_{len(elements)+1}")
            field = str(fact.get("field") or fact_id)
            canonical = str(fact.get("canonical") or field)
            acceptable = fact.get("acceptable_mentions", [])
            if not isinstance(acceptable, list):
                acceptable = []

            expected = [str(item) for item in acceptable if str(item).strip()]
            if field in {"amount_krw", "amount_status"} or "AMOUNT" in fact_id:
                expected.insert(0, f"{amount:,}원")
                expected.append(f"{amount // 10000}만원")

            if not expected:
                expected = [canonical]

            elements.append(
                {
                    "id": fact_id,
                    "field": field,
                    "description": canonical,
                    "expected": dedupe(expected),
                }
            )

    if not elements:
        elements = [
            {
                "id": "amount",
                "field": "damage_amount_krw",
                "description": "피해금액을 보존해야 함",
                "expected": [f"{amount:,}원", f"{amount // 10000}만원"],
            }
        ]

    # Guarantee amount is always evaluated.
    if not any("amount" in item["id"].lower() or item.get("field") in {"amount_krw", "damage_amount_krw"} for item in elements):
        elements.append(
            {
                "id": "amount",
                "field": "damage_amount_krw",
                "description": "피해금액을 보존해야 함",
                "expected": [f"{amount:,}원", f"{amount // 10000}만원"],
            }
        )

    return elements[:7]


def safe_generated_text(required_elements: list[dict[str, Any]], *, amount: int) -> str:
    mentions: list[str] = []
    amount_inserted = False

    for element in required_elements:
        expected = [str(x) for x in element.get("expected", []) if str(x).strip()]
        if not expected:
            continue
        token = expected[0]
        mentions.append(token)
        if token == f"{amount:,}원":
            amount_inserted = True

    if not amount_inserted:
        mentions.append(f"{amount:,}원")

    return "은행 제출용 요약: " + ", ".join(dedupe(mentions)) + " 사실을 확인해야 합니다."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def backup_and_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    write_jsonl(path, rows)


def known_amount(facts: dict[str, Any], *, fallback: int) -> int:
    value = facts.get("amount_krw")
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value >= 10_000:
        return value
    if isinstance(value, float) and value >= 10_000:
        return int(value)
    return fallback


def amount_variant(amount: int, idx: int) -> str:
    variants = [
        f"{amount // 10000}만원",
        f"{amount:,}원",
        str(amount),
        f"{amount / 10000:.0f}만원",
    ]
    return variants[idx % len(variants)]


def date_variant(date: str, idx: int) -> str:
    year, month, day = date.split("-")
    variants = [
        date,
        f"{year}.{int(month)}.{int(day)}",
        f"{year}년 {int(month)}월 {int(day)}일",
        f"{year}/{int(month)}/{int(day)}",
    ]
    return variants[idx % len(variants)]


def synthetic_date(idx: int) -> str:
    month = (idx % 12) + 1
    day = (idx % 24) + 1
    return f"2025-{month:02d}-{day:02d}"


def next_date(date: str) -> str:
    year, month, day = [int(part) for part in date.split("-")]
    day = 1 if day >= 28 else day + 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def alter_amount(amount: int) -> int:
    if amount >= 30_000_000:
        return amount - 5_000_000
    return amount + 12_500_000


def normalized_status(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value in allowed:
        return value
    return None


def status_input(field: str, value: str) -> str:
    mapping = {
        ("police_status", "reported"): "경찰 신고",
        ("police_status", "not_reported"): "미신고",
        ("police_status", "planned"): "신고 예정",
        ("freeze_status", "requested"): "지급정지 요청",
        ("freeze_status", "attempted_but_failed"): "시도했으나 실패",
        ("freeze_status", "completed"): "완료",
        ("refund_status", "not_started"): "미진행",
        ("refund_status", "requested"): "신청",
        ("refund_status", "completed"): "완료",
    }
    return mapping[(field, value)]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


if __name__ == "__main__":
    main()
