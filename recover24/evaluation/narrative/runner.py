"""Run narrative evaluation.

Default mode is deterministic checklist.
Optional judge_mode="llm" uses an LLM judge for semantic required-element coverage.
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

from .checklist import evaluate_narrative
from .judge import evaluate_narrative_with_llm
from .metrics import summarize_results


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
    judge_mode: str = "checklist",
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    rows = []
    skipped_cases = 0

    for case in cases:
        if not case.get("can_generate_document", True):
            skipped_cases += 1
            continue

        if judge_mode == "llm":
            result = evaluate_narrative_with_llm(
                canonical_case=case["canonical_case"],
                generated_text=case["generated_text"],
                required_elements=case["required_elements"],
                provider=provider,
                fallback_to_checklist=True,
            )
        elif judge_mode == "checklist":
            result = evaluate_narrative(
                canonical_case=case["canonical_case"],
                generated_text=case["generated_text"],
                required_elements=case["required_elements"],
            )
            result.update({"judge_available": False, "judge_method": "checklist", "fallback_used": False})
        else:
            raise ValueError(f"Unknown judge_mode: {judge_mode}")

        rows.append(
            {
                "case_id": case["case_id"],
                "required_elements": case["required_elements"],
                "result": result,
            }
        )

    summary = summarize_results(rows)
    summary["judge_mode"] = judge_mode
    summary["judge_fallback_count"] = sum(1 for row in rows if row["result"].get("fallback_used"))
    summary["eligible_cases"] = len(rows)
    summary["skipped_cases"] = skipped_cases
    summary["rows"] = rows
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--judge", choices=["checklist", "llm"], default="checklist")
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

    summary = run(load_cases(dataset_path), judge_mode=args.judge, provider=provider)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
