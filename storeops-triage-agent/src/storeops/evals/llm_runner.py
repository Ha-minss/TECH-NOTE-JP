from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from storeops.evals.datasets import GoldenCase, load_golden_cases
from storeops.evals.llm_evaluator import LLMEvaluator, build_llm_summary
from storeops.llm.client import ScriptedLLMClient


@dataclass(frozen=True)
class LLMEvaluationRunReport:
    output_dir: Path
    summary: dict[str, Any]


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "eval_reports" / "llm" / "latest"


def default_script_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "legacy_s1_s7"
        / "data"
        / "llm"
        / "scripted_responses"
        / "offline_payment_ops_s1_s7.json"
    )


def build_scripted_client(
    cases: list[GoldenCase],
    *,
    script_path: Path | str | None = None,
) -> ScriptedLLMClient:
    path = Path(script_path) if script_path is not None else default_script_path()
    scenario_scripts = json.loads(path.read_text(encoding="utf-8"))
    prompt_queues: dict[str, list[dict[str, Any]]] = {
        "case_parser": [],
        "checklist_extractor": [],
        "clarification": [],
        "merchant_response": [],
    }
    for case in cases:
        script_key = case.script_key or case.fixture_key
        try:
            scenario_script = scenario_scripts[script_key]
        except KeyError as exc:
            raise KeyError(f"No scripted LLM responses for {script_key}") from exc
        for prompt_name, queue in prompt_queues.items():
            response = scenario_script.get(prompt_name)
            if response is not None:
                queue.append(response)
    return ScriptedLLMClient(prompt_queues)


def build_client(provider: str, *, cases: list[GoldenCase], config: str | None = None):
    if provider == "scripted":
        return build_scripted_client(cases)
    if provider == "live":
        from storeops.llm.providers.live import LiveLLMClient

        return LiveLLMClient.from_sources(config_path=config)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def run_llm_evaluation(
    output_dir: Path | str | None = None,
    *,
    provider: str = "scripted",
    config: str | None = None,
    fixture_key: str | None = None,
    dataset_path: Path | str | None = None,
    fixture_db_path: Path | str | None = None,
) -> LLMEvaluationRunReport:
    cases = load_golden_cases(dataset_path)
    if fixture_key is not None:
        cases = [
            case
            for case in cases
            if case.fixture_key == fixture_key or case.case_id == fixture_key
        ]
        if not cases:
            raise ValueError(f"No golden cases matched fixture_key or case_id: {fixture_key}")

    client = build_client(provider, cases=cases, config=config)
    evaluator = LLMEvaluator(
        client=client,
        model_name=f"{provider}-eval",
        fixture_db_path=fixture_db_path,
    )
    case_results = [evaluator.evaluate_case(case) for case in cases]
    summary = write_llm_report(
        Path(output_dir) if output_dir is not None else default_output_dir(),
        case_results,
    )
    return LLMEvaluationRunReport(
        output_dir=Path(output_dir) if output_dir is not None else default_output_dir(),
        summary=summary,
    )


def write_llm_report(output_dir: Path, case_results) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_llm_summary(case_results)
    cases_payload = [asdict(result) for result in case_results]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# LLM Evaluation Report",
        "",
        f"- total_cases: {summary['total_cases']}",
        f"- passed_cases: {summary['passed_cases']}",
        f"- state_accuracy: {summary['state_accuracy']:.2f}",
        f"- cause_accuracy: {summary['cause_accuracy']:.2f}",
        f"- required_tool_recall: {summary['required_tool_recall']:.2f}",
        f"- forbidden_action_safety: {summary['forbidden_action_safety']:.2f}",
        f"- evidence_citation_coverage: {summary['evidence_citation_coverage']:.2f}",
        f"- abstention_safety_accuracy: {summary['abstention_safety_accuracy']:.2f}",
        f"- clarification_safety: {summary['clarification_safety']:.2f}",
        f"- merchant_response_safety: {summary['merchant_response_safety']:.2f}",
        f"- llm_trace_coverage: {summary['llm_trace_coverage']:.2f}",
        f"- fallback_rate: {summary['fallback_rate']:.2f}",
        f"- unsupported_claim_count: {summary['unsupported_claim_count']}",
        "",
        "## Failing Cases",
        "",
    ]
    failing = [result for result in case_results if not result.passed]
    if failing:
        for result in failing:
            lines.append(f"- {result.case_id}: {'; '.join(result.failure_reasons)}")
    else:
        lines.append("- None")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scripted LLM StoreOps evaluation.")
    parser.add_argument("--provider", choices=["scripted", "live"], default="scripted")
    parser.add_argument("--config", help="Optional local provider config path.")
    parser.add_argument("--output-dir", help="Override eval report output directory.")
    parser.add_argument("--dataset", help="Override golden dataset JSON path.")
    parser.add_argument("--fixture-db", help="Open an existing SQLite fixture DB instead of seeded demo fixtures.")
    parser.add_argument(
        "--fixture-key",
        help="Run only matching fixture_key or case_id, for example S1 or GOLD-S1-001.",
    )
    args = parser.parse_args(argv)
    report = run_llm_evaluation(
        output_dir=args.output_dir,
        provider=args.provider,
        config=args.config,
        fixture_key=args.fixture_key,
        dataset_path=args.dataset,
        fixture_db_path=args.fixture_db,
    )
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


__all__ = [
    "LLMEvaluationRunReport",
    "build_client",
    "build_scripted_client",
    "default_output_dir",
    "default_script_path",
    "run_llm_evaluation",
    "write_llm_report",
]


if __name__ == "__main__":
    main()

