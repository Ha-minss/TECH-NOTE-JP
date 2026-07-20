from __future__ import annotations

from pathlib import Path

from evaluation.consistency.extractor import extract_statement_facts
from evaluation.consistency.runner import load_cases, run


def test_extracts_amount_police_and_freeze_facts_from_statement():
    result = extract_statement_facts(
        "총 1750만원을 송금했고 경찰에 신고했지만 지급정지는 시도했으나 실패했습니다."
    )

    assert result["damage_amount_krw"] == 17_500_000
    assert result["police_status"] == "reported"
    assert result["freeze_status"] == "attempted_but_failed"


def test_extracts_dates_and_unknown_recipient_account():
    result = extract_statement_facts(
        "2025-12-11에 사기를 인지했고 2025-12-12에 송금했습니다. 상대 계좌번호는 모릅니다."
    )

    assert result["incident_date"] == "2025-12-11"
    assert result["transfer_date"] == "2025-12-12"
    assert result["recipient_account_known"] is False


def test_consistency_runner_reports_perfect_extractor_metrics():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "consistency" / "dataset.jsonl"

    summary = run(load_cases(dataset_path), evaluate_extractor=True)

    assert summary["extractor_accuracy"] == 1.0
    assert summary["field_recall"] == 1.0
    assert summary["field_precision"] == 1.0
