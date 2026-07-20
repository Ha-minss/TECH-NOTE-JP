from __future__ import annotations

from pathlib import Path

from evaluation.consistency.conflict_checker import check_consistency
from evaluation.consistency.runner import load_cases, run


def test_amount_conflict_is_blocking():
    result = check_consistency(
        form_facts={"damage_amount_krw": 17_500_000},
        statement_facts={"damage_amount_krw": 30_000_000},
    )

    assert result["can_generate_document"] is False
    assert result["requires_human_review"] is True
    assert result["conflicts"] == [
        {
            "field": "damage_amount_krw",
            "form_value": 17_500_000,
            "statement_value": 30_000_000,
            "severity": "blocking",
            "message": "Field 'damage_amount_krw' conflicts between form and statement.",
        }
    ]


def test_unknown_statement_value_does_not_conflict():
    result = check_consistency(
        form_facts={"damage_amount_krw": 17_500_000},
        statement_facts={"damage_amount_krw": "unknown"},
    )

    assert result["conflicts"] == []
    assert result["enrichments"] == []
    assert result["can_generate_document"] is True
    assert result["requires_human_review"] is False


def test_unknown_plus_reported_is_enrichment_not_blocking():
    result = check_consistency(
        form_facts={"police_status": "unknown"},
        statement_facts={"police_status": "reported"},
    )

    assert result["conflicts"] == []
    assert result["enrichments"] == [
        {
            "field": "police_status",
            "statement_value": "reported",
            "message": "Field 'police_status' was enriched from the statement.",
        }
    ]
    assert result["can_generate_document"] is True
    assert result["requires_human_review"] is False


def test_not_reported_plus_reported_is_blocking():
    result = check_consistency(
        form_facts={"police_status": "not_reported"},
        statement_facts={"police_status": "reported"},
    )

    assert result["can_generate_document"] is False
    assert result["conflicts"][0]["field"] == "police_status"
    assert result["conflicts"][0]["severity"] == "blocking"


def test_normal_case_can_generate_document():
    result = check_consistency(
        form_facts={
            "damage_amount_krw": 17_500_000,
            "police_status": "reported",
            "freeze_status": "requested",
        },
        statement_facts={
            "damage_amount_krw": 17_500_000,
            "police_status": "reported",
            "freeze_status": "requested",
        },
    )

    assert result["conflicts"] == []
    assert result["can_generate_document"] is True
    assert result["requires_human_review"] is False


def test_dataset_runner_reports_perfect_blocking_accuracy():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "consistency" / "dataset.jsonl"

    cases = load_cases(dataset_path)
    summary = run(cases)

    assert len(cases) >= 10
    assert summary["blocking_accuracy"] == 1.0
