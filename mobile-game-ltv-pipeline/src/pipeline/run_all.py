from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[pipeline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_all(project_root: Path, zip_path: Path, python: str = sys.executable) -> None:
    """Run the portfolio final 2-stage LTV pipeline only.

    Steps:
    1. raw data validation
    2. feature build
    3. model input validation
    4. final two-stage model train/refit
    5. test-like prediction artifact generation
    6. business analysis report generation
    7. model card generation
    """
    _run([python, "src/pipeline/validate_raw_data.py", "--project-root", str(project_root), "--zip-path", str(zip_path)])
    _run([python, "src/pipeline/build_features.py", "--output-root", str(project_root), "--zip-path", str(zip_path)])
    _run([python, "src/pipeline/validate_model_input.py", "--feature-root", str(project_root)])
    _run([python, "src/pipeline/train_final_model.py", "--project-root", str(project_root), "--zip-path", str(zip_path)])
    _run([python, "src/pipeline/predict_submission.py", "--project-root", str(project_root), "--zip-path", str(zip_path)])
    _run([python, "src/pipeline/build_business_report.py", "--project-root", str(project_root), "--zip-path", str(zip_path)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    args = parser.parse_args()
    run_all(Path(args.project_root).resolve(), Path(args.zip_path))


if __name__ == "__main__":
    main()
