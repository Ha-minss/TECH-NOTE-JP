"""Report and provenance artifacts for Recover24 evaluation runs."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.generators.gemma_generator import _build_prompt


def write_final_artifacts(
    *,
    output_dir: str | Path,
    dataset_path: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_path)

    summary_path = out / "eval_summary.json"
    details_path = out / "eval_details.jsonl"
    report_path = out / "portfolio_report.md"
    failures_path = out / "failure_cases.md"
    manifest_path = out / "run_manifest.json"
    lock_path = out / "FINAL_EVAL_LOCK.json"

    summary_path.write_text(_json(summary), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": _file_sha256(dataset),
        "prompt_sha256": hashlib.sha256(
            inspect.getsource(_build_prompt).encode("utf-8")
        ).hexdigest(),
        "case_count": len({row.get("case_id") for row in rows}),
        "record_count": len(rows),
        "git_revision": _git_revision(dataset.parent),
        **run_config,
    }
    manifest_path.write_text(_json(manifest), encoding="utf-8")
    failures_path.write_text(_failure_report(rows), encoding="utf-8")
    report_path.write_text(
        _portfolio_report(summary, manifest, rows),
        encoding="utf-8",
    )
    lock_path.write_text(
        _json(
            {
                "created_at": manifest["created_at"],
                "dataset_sha256": manifest["dataset_sha256"],
                "prompt_sha256": manifest["prompt_sha256"],
                "warning": "Do not tune prompts from this final test result.",
            }
        ),
        encoding="utf-8",
    )
    return {
        "portfolio_report": report_path,
        "summary": summary_path,
        "details": details_path,
        "failure_cases": failures_path,
        "manifest": manifest_path,
        "lock": lock_path,
    }


def _portfolio_report(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Recover24 Final LLM Evaluation",
        "",
        "This report is the locked portfolio evaluation for the 6-case test split.",
        "The small sample demonstrates engineering behavior; it does not establish statistical certainty.",
        "",
        "## Experiment",
        "",
        "- Baseline: deterministic template",
        "- Candidate: Gemma generation",
        "- Safety layer: fact/status validator",
        "- Final method: Gemma + validator + template fallback",
        f"- Dataset SHA-256: `{manifest['dataset_sha256']}`",
        f"- Prompt SHA-256: `{manifest['prompt_sha256']}`",
        "",
        "## Results",
        "",
        "| Method | Cases | Important facts | Amount preservation | Status contradiction | Unsupported claims | Safe output | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, data in summary.get("methods", {}).items():
        lines.append(
            f"| {method} | {data['cases']} | {_pct(data.get('important_fact_inclusion'))} | "
            f"{_pct(data.get('amount_preservation'))} | {_pct(data.get('status_contradiction_rate'))} | "
            f"{_pct(data.get('unsupported_claim_case_rate'))} | {_pct(data.get('safe_output_rate'))} | "
            f"{_pct(data.get('fallback_rate'))} |"
        )
    failed = sum(not row.get("validation", {}).get("safe_to_use", False) for row in rows)
    lines.extend(
        [
            "",
            "## Failure Review",
            "",
            f"- Unsafe intermediate/final records: {failed}",
            "- See `failure_cases.md` for case-level evidence.",
            "",
            "## Limitations",
            "",
            "- The final split contains only six cases.",
            "- Claim extraction quality must be checked against human annotations before stronger safety claims.",
            "- Results apply only to the recorded model endpoint, prompt hash, and dataset hash.",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_report(rows: list[dict[str, Any]]) -> str:
    lines = ["# Recover24 Evaluation Failures", ""]
    failures = [
        row
        for row in rows
        if not row.get("validation", {}).get("safe_to_use", False)
        or row.get("validation", {}).get("required_facts", {}).get("missing")
    ]
    if not failures:
        return "# Recover24 Evaluation Failures\n\nNo failures recorded.\n"
    for row in failures:
        validation = row.get("validation", {})
        lines.extend(
            [
                f"## {row.get('case_id')} — {row.get('method')}",
                "",
                f"- Safe to use: {validation.get('safe_to_use')}",
                f"- Missing facts: {validation.get('required_facts', {}).get('missing', [])}",
                f"- Blocking errors: {validation.get('blocking_errors', [])}",
                "",
            ]
        )
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
