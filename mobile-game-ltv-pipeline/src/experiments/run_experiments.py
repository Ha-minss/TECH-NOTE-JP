from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENT_COMMANDS = [
    ["src/experiments/train_baseline.py"],
    ["src/experiments/train_linear_model.py"],
    ["src/experiments/train_xgboost_model.py"],
    ["src/experiments/run_feature_ablation.py"],
    ["src/experiments/train_two_stage_model.py"],
    ["src/experiments/rolling_time_validation.py"],
    ["src/experiments/tune_xgboost_optuna.py"],
]

ZIP_REQUIRED = {
    "src/experiments/run_feature_ablation.py",
    "src/experiments/train_two_stage_model.py",
    "src/experiments/rolling_time_validation.py",
    "src/experiments/tune_xgboost_optuna.py",
}

EXPERIMENT_REPORTS = [
    "baseline_results.md",
    "linear_model_results.md",
    "xgboost_model_results.md",
    "feature_engineering_results.md",
    "two_stage_model_results.md",
    "rolling_validation_results.md",
    "optuna_tuning_results.md",
]

EXPERIMENT_PREFIXES = (
    "baseline_",
    "linear_",
    "xgboost_",
    "two_stage_",
    "feature_engineering_",
    "rolling_validation_",
    "optuna_",
)


def organize_experiment_outputs(project_root: Path) -> None:
    reports_dir = project_root / "reports"
    report_target = reports_dir / "experiments"
    data_target = project_root / "data" / "experiments"
    report_target.mkdir(parents=True, exist_ok=True)
    data_target.mkdir(parents=True, exist_ok=True)

    for name in EXPERIMENT_REPORTS:
        src = reports_dir / name
        if src.exists():
            shutil.move(str(src), str(report_target / name))

    processed_dir = project_root / "data" / "processed"
    if processed_dir.exists():
        for path in list(processed_dir.iterdir()):
            if path.is_file() and path.name.startswith(EXPERIMENT_PREFIXES):
                target = data_target / path.name
                if target.exists():
                    target.unlink()
                shutil.move(str(path), str(target))


def run_experiments(project_root: Path, zip_path: Path, python: str = sys.executable) -> None:
    for command in EXPERIMENT_COMMANDS:
        script = command[0]
        cmd = [python, script, "--project-root", str(project_root)]
        if script in ZIP_REQUIRED:
            cmd.extend(["--zip-path", str(zip_path)])
        print("[experiments]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    organize_experiment_outputs(project_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    args = parser.parse_args()
    run_experiments(Path(args.project_root).resolve(), Path(args.zip_path))


if __name__ == "__main__":
    main()
