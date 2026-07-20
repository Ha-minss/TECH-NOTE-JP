"""Command-line workflow for Recover24 dev, challenge, and final evaluation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.dataset_loader import load_jsonl
from evaluation.generators.gemma_generator import GemmaGenerator
from evaluation.generators.hybrid_generator import GemmaValidatorGenerator, HybridGenerator
from evaluation.generators.template_generator import TemplateGenerator
from evaluation.reporting import write_final_artifacts
from evaluation.run_policy import RunPolicy, resolve_run_policy
from evaluation.scoring import summarize
from evaluation.validators.fact_validator import validate_generated_record
from recover24.providers.base import LLMProvider
from recover24.providers.gemma_colab import GemmaColabProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("template", "gemma", "gemma_validator", "gemma_validator_fallback")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    policy = resolve_run_policy(
        args.mode,
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        output_dir=args.output_dir,
    )
    try:
        policy.validate(
            provider_kind=args.provider,
            claim_provider_kind=args.claim_provider,
            force_final=args.force_final,
        )
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc

    provider = _make_provider(args.provider, args.gemma_url, args.timeout)
    claim_provider = _make_provider(args.claim_provider, args.gemma_url, args.timeout)
    methods = _parse_methods(args.methods)
    rows = run_evaluation(
        policy=policy,
        methods=methods,
        provider=provider,
        claim_provider=claim_provider,
        max_retries=args.max_retries,
    )
    summary = summarize(rows)
    written = write_run_artifacts(
        policy=policy,
        rows=rows,
        summary=summary,
        run_config={
            "mode": policy.mode,
            "provider": args.provider,
            "claim_provider": args.claim_provider,
            "methods": methods,
            "max_retries": args.max_retries,
            "gemma_url_host": _safe_host(args.gemma_url or os.getenv("RECOVER24_GEMMA_COLAB_URL")),
        },
    )
    for path in written.values():
        print(f"Wrote {path}")


def run_evaluation(
    *,
    policy: RunPolicy,
    methods: list[str],
    provider: LLMProvider | None,
    claim_provider: LLMProvider | None,
    max_retries: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in load_jsonl(policy.dataset):
        for method in methods:
            record = generate_record(method, case, provider, claim_provider, max_retries)
            validation = record.get("meta", {}).get("validation")
            if validation is None:
                validation = validate_generated_record(case, record, claim_provider=claim_provider)
            rows.append(
                {
                    "case_id": case.case_id,
                    "difficulty": case.difficulty,
                    "method": record["method"],
                    "outputs": record["outputs"],
                    "meta": record.get("meta", {}),
                    "validation": validation,
                }
            )
    return rows


def generate_record(
    method: str,
    case,
    provider: LLMProvider | None,
    claim_provider: LLMProvider | None,
    max_retries: int,
) -> dict[str, Any]:
    if method == "template":
        return TemplateGenerator().generate(case)
    if method == "gemma":
        record = GemmaGenerator(provider).generate(case)
    elif method == "gemma_validator":
        record = GemmaValidatorGenerator(provider, claim_provider).generate(case)
    elif method == "gemma_validator_fallback":
        record = HybridGenerator(provider, claim_provider, max_retries=max_retries).generate(case)
    else:
        raise ValueError(method)
    if provider is None:
        record["method"] = f"dry_run_{method}"
        record.setdefault("meta", {})["dry_run"] = True
    return record


def write_run_artifacts(
    *,
    policy: RunPolicy,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Path]:
    if policy.final:
        return write_final_artifacts(
            output_dir=policy.output_dir,
            dataset_path=policy.dataset,
            rows=rows,
            summary=summary,
            run_config=run_config,
        )

    out = policy.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "eval_summary.json"
    details_path = out / "eval_details.jsonl"
    report_path = out / "eval_report.md"
    manifest_path = out / "run_manifest.json"
    summary_path.write_text(_json(summary), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(
        render_report(summary, policy=policy, run_config=run_config),
        encoding="utf-8",
    )
    manifest_path.write_text(
        _json(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(policy.dataset),
                "case_count": len({row["case_id"] for row in rows}),
                **run_config,
            }
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "details": details_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def render_report(
    summary: dict[str, Any],
    *,
    policy: RunPolicy,
    run_config: dict[str, Any],
) -> str:
    lines = [
        "# Recover24 LLM Evaluation Report",
        "",
        f"Mode: `{policy.mode}`",
        f"Dataset: `{policy.dataset}`",
        f"Generation provider: `{run_config.get('provider')}`",
        f"Claim provider: `{run_config.get('claim_provider')}`",
        "",
    ]
    if run_config.get("provider") in (None, "", "none"):
        lines.extend(
            [
                "> **DRY RUN — NOT A GEMMA PERFORMANCE RESULT.**",
                "",
            ]
        )
    lines.extend(
        [
            "| Method | Cases | Important facts | Amount preservation | Status contradiction | Unsupported claim cases | Event order | Safe output | Fallback | Avg latency | Avg LLM calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, data in summary.get("methods", {}).items():
        lines.append(
            f"| {method} | {data['cases']} | {_pct(data.get('important_fact_inclusion'))} | "
            f"{_pct(data.get('amount_preservation'))} | {_pct(data.get('status_contradiction_rate'))} | "
            f"{_pct(data.get('unsupported_claim_case_rate'))} | {_pct(data.get('event_order_accuracy'))} | "
            f"{_pct(data.get('safe_output_rate'))} | {_pct(data.get('fallback_rate'))} | "
            f"{data.get('avg_latency_sec', 0):.3f}s | {data.get('avg_llm_calls', 0):.2f} |"
        )
    lines.extend(
        [
            "",
            "Unmeasured metrics are displayed as `n/a`, never as zero.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "challenge", "final"], default="dev")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--provider", choices=["none", "gemma"], default="none")
    parser.add_argument("--claim-provider", choices=["none", "gemma"], default="none")
    parser.add_argument("--gemma-url", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--force-final", action="store_true")
    return parser.parse_args(argv)


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise SystemExit(f"Unknown methods: {invalid}. Allowed: {METHODS}")
    return methods


def _make_provider(
    kind: str | None,
    gemma_url: str | None,
    timeout: int,
) -> LLMProvider | None:
    if kind in (None, "", "none"):
        return None
    if kind == "gemma":
        return GemmaColabProvider(
            base_url=gemma_url or os.getenv("RECOVER24_GEMMA_COLAB_URL"),
            timeout_seconds=timeout,
        )
    raise SystemExit(f"Unknown provider: {kind}")


def _safe_host(url: str | None) -> str | None:
    if not url:
        return None
    return url.split("://", 1)[-1].split("/", 1)[0]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
