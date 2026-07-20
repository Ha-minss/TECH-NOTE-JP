from __future__ import annotations

from pathlib import Path

from evaluation.narrative.checklist import evaluate_narrative
from evaluation.narrative.runner import load_cases, run
from evaluation.run_all import run_all


def test_narrative_checklist_detects_missing_required_element():
    result = evaluate_narrative(
        canonical_case={
            "damage_amount_krw": 17_500_000,
            "fraud_type": "loan_scam",
            "police_status": "reported",
        },
        generated_text="대출사기 피해를 입었고 경찰에 신고했습니다.",
        required_elements=[
            {"id": "fraud_type", "expected": ["대출사기"]},
            {"id": "amount", "expected": ["17,500,000원", "1750만원"]},
            {"id": "police_status", "expected": ["경찰", "신고"]},
        ],
    )

    assert result["passed"] is False
    assert "amount" in result["missing_elements"]


def test_narrative_runner_skips_blocked_cases_and_scores_eligible_cases():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "narrative" / "dataset.jsonl"

    summary = run(load_cases(dataset_path))

    assert summary["eligible_cases"] >= 2
    assert summary["skipped_cases"] >= 1
    assert summary["checklist_accuracy"] == 1.0


def test_run_all_executes_three_evaluation_tracks():
    project_root = Path(__file__).parents[1]

    summary = run_all(project_root=project_root)

    assert set(summary) == {"normalization", "consistency", "narrative"}
    assert summary["normalization"]["canonical_accuracy"] == 1.0
    assert summary["consistency"]["blocking_accuracy"] == 1.0
    assert summary["narrative"]["checklist_accuracy"] == 1.0
