"""Run consistency evaluation.

Default mode uses dataset statement_facts.
Optional modes:
- extractor_mode="rule": run local rule extractor on raw_statement.
- extractor_mode="llm": run LLM extractor on raw_statement, fallback to rule on failure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from recover24.providers.base import LLMProvider
from recover24.providers.gemma_colab import GemmaColabProvider
from recover24.providers.deepseek import DeepSeekProvider

from .conflict_checker import check_consistency
from .extractor import extract_statement_facts as extract_statement_facts_rule
from .llm_extractor import extract_statement_facts_llm_with_meta
from .metrics import summarize_extractor_results, summarize_results


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(
    cases: list[dict[str, Any]],
    *,
    evaluate_extractor: bool = False,
    extractor_mode: str = "gold",
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    rows = []
    extractor_rows = []

    for case in cases:
        statement_facts = case["statement_facts"]
        extractor_meta = {"extractor": "gold", "fallback_used": False}

        if evaluate_extractor or extractor_mode != "gold":
            if extractor_mode == "llm":
                statement_facts, extractor_meta = extract_statement_facts_llm_with_meta(
                    case.get("raw_statement", ""),
                    provider,
                    fallback_to_rule=True,
                )
            elif extractor_mode in {"gold", "rule"}:
                statement_facts = extract_statement_facts_rule(case.get("raw_statement", ""))
                extractor_meta = {"extractor": "rule", "fallback_used": False}
            else:
                raise ValueError(f"Unknown extractor_mode: {extractor_mode}")

            extractor_rows.append(
                {
                    "case_id": case["case_id"],
                    "expected_statement_facts": case.get("expected_statement_facts", case["statement_facts"]),
                    "predicted_statement_facts": statement_facts,
                    "extractor_meta": extractor_meta,
                }
            )

        result = check_consistency(
            form_facts=case["form_facts"],
            statement_facts=statement_facts,
        )

        rows.append(
            {
                "case_id": case["case_id"],
                "expected_conflict_fields": list(case.get("expected_conflict_fields", [])),
                "expected_can_generate_document": case["expected_can_generate_document"],
                "predicted_conflict_fields": [item["field"] for item in result["conflicts"]],
                "statement_facts": statement_facts,
                "extractor_meta": extractor_meta,
                "result": result,
            }
        )

    summary = summarize_results(rows)

    if evaluate_extractor or extractor_mode != "gold":
        summary.update(summarize_extractor_results(extractor_rows))
        summary["extractor_mode"] = extractor_mode
        summary["extractor_fallback_count"] = sum(
            1 for row in extractor_rows if row.get("extractor_meta", {}).get("fallback_used")
        )

    summary["cases"] = len(rows)
    summary["rows"] = rows
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--extractor", choices=["gold", "rule", "llm"], default="gold")
    parser.add_argument("--provider", choices=["none", "gemma", "deepseek"], default="none")
    parser.add_argument("--gemma-url", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset) if args.dataset else Path(__file__).with_name("dataset.jsonl")
    provider: LLMProvider | None = None

    if args.provider == "gemma":
        provider = GemmaColabProvider(
            base_url=args.gemma_url or os.getenv("RECOVER24_GEMMA_COLAB_URL"),
            timeout_seconds=args.timeout,
        )
    elif args.provider == "deepseek":
        provider = DeepSeekProvider(timeout_seconds=args.timeout)

    summary = run(
        load_cases(dataset_path),
        evaluate_extractor=args.extractor != "gold",
        extractor_mode=args.extractor,
        provider=provider,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
