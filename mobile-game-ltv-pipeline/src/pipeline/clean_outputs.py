from __future__ import annotations

from pathlib import Path

GENERATED_PATTERNS = [
    "data/processed/final_test_context_predictions.*",
    "data/processed/final_test_user_predictions.*",
    "data/processed/final_prediction_summary.json",
    "models/final_two_stage_stage1.joblib",
    "models/final_two_stage_stage2.joblib",
    "models/final_preprocessor.joblib",
    "models/final_model_metadata.json",
    "reports/final_prediction_report.md",
]


def main() -> None:
    root = Path.cwd()
    removed = []
    for pattern in GENERATED_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print("Removed generated final-pipeline files:")
    for path in removed:
        print(f"- {path}")


if __name__ == "__main__":
    main()
