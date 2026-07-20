from __future__ import annotations

from pathlib import Path

from evaluation.normalization.normalizer import normalize_case
from evaluation.normalization.renderer_check import render_expected_fields
from evaluation.normalization.runner import load_cases, run


def test_normalize_amount_and_date_to_canonical_values():
    result = normalize_case(
        {
            "damage_amount": "1750만원",
            "incident_date": "2025.12.11",
        },
        required_fields=["damage_amount_krw", "incident_date"],
    )

    assert result["canonical"] == {
        "damage_amount_krw": 17_500_000,
        "incident_date": "2025-12-11",
    }
    assert result["missing_required_fields"] == []
    assert result["blocks_on_missing_required"] is False


def test_unknown_and_empty_values_are_preserved_as_unknown():
    result = normalize_case(
        {
            "damage_amount": "",
            "incident_date": "unknown",
        },
        required_fields=["damage_amount_krw", "incident_date"],
    )

    assert result["canonical"]["damage_amount_krw"] == "unknown"
    assert result["canonical"]["incident_date"] == "unknown"
    assert set(result["missing_required_fields"]) == {"damage_amount_krw", "incident_date"}
    assert result["blocks_on_missing_required"] is True


def test_renderer_check_formats_amount_and_date_for_document_slots():
    rendered = render_expected_fields(
        {
            "damage_amount_krw": 17_500_000,
            "incident_date": "2025-12-11",
        }
    )

    assert rendered["totalDamageAmountLabel"] == "17,500,000원"
    assert rendered["reportedAt"] == "2025년 12월 11일"


def test_normalization_dataset_runner_reports_perfect_accuracy():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "normalization" / "dataset.jsonl"

    summary = run(load_cases(dataset_path))

    assert summary["cases"] >= 8
    assert summary["canonical_accuracy"] == 1.0
    assert summary["rendered_accuracy"] == 1.0
