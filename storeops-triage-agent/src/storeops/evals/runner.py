"""Run the deterministic evaluation suite."""

from __future__ import annotations

import argparse
import json

from dataclasses import dataclass
from pathlib import Path

from storeops.evals.datasets import load_golden_cases
from storeops.evals.deterministic import DeterministicEvaluator, default_fixture_db_path
from storeops.evals.report import write_report


@dataclass(frozen=True)
class EvaluationRunReport:
    output_dir: Path
    summary: dict


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "experiments" / "eval_runs" / "deterministic" / "latest"


def run_full_evaluation(
    output_dir: Path | str | None = None,
    *,
    dataset_path: Path | str | None = None,
    fixture_db_path: Path | str | None = None,
) -> EvaluationRunReport:
    target_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    cases = load_golden_cases(dataset_path)
    evaluator = DeterministicEvaluator.from_fixture_db(
        fixture_db_path if fixture_db_path is not None else default_fixture_db_path()
    )
    case_results = [evaluator.evaluate_case(case) for case in cases]
    summary = write_report(target_dir, case_results)
    return EvaluationRunReport(output_dir=target_dir, summary=summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic StoreOps evaluation.")
    parser.add_argument("--output-dir", help="Override eval report output directory.")
    parser.add_argument("--dataset", help="Override golden dataset JSON path.")
    parser.add_argument("--fixture-db", help="Open an existing SQLite fixture DB instead of seeded demo fixtures.")
    args = parser.parse_args(argv)
    report = run_full_evaluation(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        fixture_db_path=args.fixture_db,
    )
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))


__all__ = ["EvaluationRunReport", "default_output_dir", "main", "run_full_evaluation"]


if __name__ == "__main__":
    main()
