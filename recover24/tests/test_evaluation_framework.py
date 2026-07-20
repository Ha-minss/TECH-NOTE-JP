from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset_loader import GoldCase, load_jsonl
from evaluation.generators.hybrid_generator import HybridGenerator
from evaluation.run_eval import _generate
from evaluation.run_policy import RunPolicy, resolve_run_policy
from evaluation.scoring import summarize
from evaluation.validators.fact_validator import validate_text


def _case(**overrides) -> GoldCase:
    data = {
        "case_id": "CASE_TEST",
        "difficulty": "challenge",
        "case_type": "loan_scam",
        "structured_facts": {
            "amount_krw": 3_000_000,
            "freeze_status": "attempted_but_failed",
            "police_status": "reported",
            "refund_status": "unknown",
            "fraud_type": "loan_scam",
        },
        "raw_statement": "300만원을 송금했고 지급정지를 시도했지만 완료하지 못했습니다.",
        "required_fact_ids": ["amount_krw", "freeze_status", "fraud_type"],
        "expected_event_order": ["contact", "transfer", "freeze_attempt_failed"],
        "input_conflicts": [],
        "fact_aliases": {
            "freeze_status": ["지급정지를 시도했지만 완료하지 못"],
            "fraud_type": ["대출 사기"],
        },
    }
    data.update(overrides)
    return GoldCase.from_dict(data)


def _record(validation: dict, *, method: str = "gemma") -> dict:
    return {
        "case_id": "CASE_TEST",
        "method": method,
        "outputs": {},
        "meta": {"latency_sec": 0.1, "llm_calls": 1},
        "validation": validation,
    }


def test_unmeasured_claim_metrics_are_none_not_zero():
    validation = {
        "safe_to_use": True,
        "amounts": {"errors": [], "required_amounts_krw": [3_000_000]},
        "required_facts": {"score": 1.0},
        "status_claims": [],
        "unsupported_claims": [],
        "event_order": {"score": None},
        "metric_availability": {
            "status_claims": False,
            "unsupported_claims": False,
            "event_order": False,
        },
    }

    result = summarize([_record(validation)])["methods"]["gemma"]

    assert result["status_contradiction_rate"] is None
    assert result["unsupported_claim_case_rate"] is None
    assert result["event_order_accuracy"] is None


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("gemma", "dry_run_gemma"),
        ("gemma_validator", "dry_run_gemma_validator"),
        ("gemma_validator_fallback", "dry_run_gemma_validator_fallback"),
    ],
)
def test_provider_free_generation_is_labeled_dry_run(method: str, expected: str):
    record = _generate(method, _case(), None, None, max_retries=0)
    assert record["method"] == expected


def test_run_policy_defaults_are_mode_specific(tmp_path: Path):
    project = tmp_path

    dev = resolve_run_policy("dev", project_root=project)
    challenge = resolve_run_policy("challenge", project_root=project)
    final = resolve_run_policy("final", project_root=project)

    assert dev.dataset == project / "evaluation/dataset/dev.jsonl"
    assert dev.output_dir == project / "evaluation/runs/dev"
    assert challenge.dataset == project / "evaluation/dataset/challenge.jsonl"
    assert challenge.output_dir == project / "evaluation/runs/challenge"
    assert final.dataset == project / "evaluation/dataset/test.jsonl"
    assert final.output_dir == project / "evaluation/final_submission"


def test_final_policy_requires_real_generation_and_claim_providers(tmp_path: Path):
    policy = resolve_run_policy("final", project_root=tmp_path)

    with pytest.raises(ValueError, match="generation provider"):
        policy.validate(provider_kind="none", claim_provider_kind="gemma")

    with pytest.raises(ValueError, match="claim provider"):
        policy.validate(provider_kind="gemma", claim_provider_kind="none")


def test_final_policy_refuses_existing_lock_without_force(tmp_path: Path):
    policy = RunPolicy(
        mode="final",
        dataset=tmp_path / "test.jsonl",
        output_dir=tmp_path / "final",
        final=True,
    )
    policy.output_dir.mkdir()
    (policy.output_dir / "FINAL_EVAL_LOCK.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="locked"):
        policy.validate(provider_kind="gemma", claim_provider_kind="gemma")

    policy.validate(provider_kind="gemma", claim_provider_kind="gemma", force_final=True)


def test_changed_amount_is_blocked():
    result = validate_text(_case(), "피해금액은 500만원입니다.")
    assert result["safe_to_use"] is False
    assert {error["type"] for error in result["amounts"]["errors"]} == {
        "missing_required_amount",
        "unsupported_amount",
    }


def test_status_claims_distinguish_supported_contradicted_and_unsupported():
    claims = {
        "status_claims": [
            {"field": "police_status", "claimed_value": "reported", "evidence_text": "신고함"},
            {"field": "freeze_status", "claimed_value": "completed", "evidence_text": "완료함"},
            {"field": "refund_status", "claimed_value": "completed", "evidence_text": "환급됨"},
        ],
        "unsupported_claims": [],
        "event_order": [],
        "supported_fact_ids": [],
    }
    result = validate_text(_case(), "3,000,000원", claims)
    labels = {item["field"]: item["label"] for item in result["status_claims"]}

    assert labels == {
        "police_status": "supported",
        "freeze_status": "contradicted",
        "refund_status": "contradicted",
    }
    assert result["safe_to_use"] is False


def test_unsupported_claim_blocks_output():
    claims = {
        "status_claims": [],
        "unsupported_claims": [{"claim": "피해금 전액 환급", "reason": "근거 없음"}],
        "event_order": [],
        "supported_fact_ids": [],
    }
    result = validate_text(_case(), "3,000,000원", claims)
    assert result["safe_to_use"] is False
    assert result["blocking_errors"][0]["type"] == "unsupported_claim"


class _UnsafeProvider:
    def generate_json(self, prompt: str):
        return {
            "incident_circumstances": "피해금은 500만원입니다.",
            "post_action": "지급정지를 완료했습니다.",
            "staff_summary": "환급 완료",
        }


def test_hybrid_falls_back_when_generated_amount_is_unsafe():
    record = HybridGenerator(_UnsafeProvider(), None, max_retries=0).generate(_case())
    assert record["meta"]["fallback_used"] is True
    assert record["meta"]["blocked_by_validator"] is True
    assert record["method"] == "gemma_validator_fallback"


def test_challenge_dataset_contains_declared_conflicts():
    path = Path(__file__).parents[1] / "evaluation/dataset/challenge.jsonl"
    conflicts = [
        conflict
        for case in load_jsonl(path)
        for conflict in case.input_conflicts
    ]
    conflict_types = {item["type"] for item in conflicts}

    assert {"amount_mismatch", "police_status_mismatch", "freeze_status_mismatch"} <= conflict_types


def test_final_artifacts_include_manifest_report_failures_and_lock(tmp_path: Path):
    from evaluation.reporting import write_final_artifacts

    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(_case().to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "final"
    rows = [
        _record(
            {
                "safe_to_use": False,
                "amounts": {"errors": [{"type": "unsupported_amount"}], "required_amounts_krw": [3_000_000]},
                "required_facts": {"score": 0.5, "missing": ["freeze_status"]},
                "status_claims": [],
                "unsupported_claims": [],
                "event_order": {"score": None},
                "blocking_errors": [{"type": "unsupported_amount"}],
                "metric_availability": {
                    "status_claims": False,
                    "unsupported_claims": False,
                    "event_order": False,
                },
            }
        )
    ]
    summary = summarize(rows)

    written = write_final_artifacts(
        output_dir=output,
        dataset_path=dataset,
        rows=rows,
        summary=summary,
        run_config={
            "mode": "final",
            "provider": "gemma",
            "claim_provider": "gemma",
            "methods": ["gemma"],
            "max_retries": 1,
        },
    )

    assert set(written) == {
        "portfolio_report",
        "summary",
        "details",
        "failure_cases",
        "manifest",
        "lock",
    }
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"]
    assert manifest["prompt_sha256"]
    assert "6-case test split" in (output / "portfolio_report.md").read_text(encoding="utf-8")
    assert "CASE_TEST" in (output / "failure_cases.md").read_text(encoding="utf-8")

