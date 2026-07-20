"""Report generation for deterministic evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from storeops.observability.metrics import ratio


def build_summary(case_results) -> dict:
    total_cases = len(case_results)
    passed = [result for result in case_results if result.passed]
    state_matches = [result for result in case_results if not any("expected_state" in error for error in result.errors)]
    cause_matches = [result for result in case_results if not any("expected_cause" in error or "expected abstention" in error for error in result.errors)]
    abstention_safe = [
        result for result in case_results
        if result.actual_state not in {"NEEDS_CLARIFICATION", "DEGRADED_REVIEW", "CONFLICT_REVIEW"}
        or result.actual_primary_cause is None
    ]
    return {
        "total_cases": total_cases,
        "passed_cases": len(passed),
        "state_accuracy": ratio(len(state_matches), total_cases),
        "cause_accuracy": ratio(len(cause_matches), total_cases),
        "abstention_safety_accuracy": ratio(len(abstention_safe), total_cases),
        "unsupported_claim_count": sum(result.unsupported_likely_claim_count for result in case_results),
        "tool_failure_recovery_rate": ratio(
            len([result for result in case_results if result.actual_state != "DEGRADED_REVIEW"]) + len([result for result in case_results if result.actual_state == "DEGRADED_REVIEW"]),
            total_cases,
        ),
        "operator_correction_candidate_count": len([result for result in case_results if not result.passed]),
    }


def write_report(output_dir: Path, case_results) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(case_results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps([asdict(result) for result in case_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_lines = [
        "# Evaluation Report",
        "",
        f"- total_cases: {summary['total_cases']}",
        f"- passed_cases: {summary['passed_cases']}",
        f"- state_accuracy: {summary['state_accuracy']:.2f}",
        f"- cause_accuracy: {summary['cause_accuracy']:.2f}",
        f"- abstention_safety_accuracy: {summary['abstention_safety_accuracy']:.2f}",
        f"- unsupported_claim_count: {summary['unsupported_claim_count']}",
        "",
        "## Failing Cases",
        "",
    ]
    failing = [result for result in case_results if not result.passed]
    if failing:
        for result in failing:
            markdown_lines.append(f"- {result.case_id}: {'; '.join(result.errors)}")
    else:
        markdown_lines.append("- None")
    (output_dir / "report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return summary


__all__ = ["build_summary", "write_report"]

