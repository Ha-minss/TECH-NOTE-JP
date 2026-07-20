from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.predict_submission import FINAL_MODEL, fit_full_two_stage_model, prepare_full_train_test_features
from experiments.run_feature_ablation import build_time_bucket_features_from_zip
from experiments.tune_xgboost_optuna import build_param_dict_from_best


def train_final_model(project_root: Path, zip_path: Path) -> dict[str, object]:
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_model_input = pd.read_parquet(processed_dir / "train_model_input.parquet")
    test_model_input = pd.read_parquet(processed_dir / "test_model_input.parquet")
    params_payload = json.loads((processed_dir / "final_model_params.json").read_text(encoding="utf-8"))
    params = build_param_dict_from_best(params_payload["best_params"][FINAL_MODEL])
    train_bucket = build_time_bucket_features_from_zip(zip_path, "train.csv")
    test_bucket = build_time_bucket_features_from_zip(zip_path, "test.csv")

    train_features, test_features, feature_columns, preprocessor = prepare_full_train_test_features(
        train_model_input,
        test_model_input,
        train_bucket,
        test_bucket,
    )
    _, classifier, regressor, preprocessor = fit_full_two_stage_model(train_features, test_features, feature_columns, preprocessor, params)

    joblib.dump(classifier, models_dir / "final_two_stage_stage1.joblib")
    joblib.dump(regressor, models_dir / "final_two_stage_stage2.joblib")
    joblib.dump(preprocessor, models_dir / "final_preprocessor.joblib")
    metadata = {
        "model": FINAL_MODEL,
        "purpose": "portfolio production-style final refit",
        "train_rows": int(len(train_model_input)),
        "test_feature_rows_seen_for_schema": int(len(test_model_input)),
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "target": "ltv_d8_d180",
        "test_metrics": "not_calculated_no_test_target",
        "params": params,
    }
    (models_dir / "final_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--zip-path", default=str(Path.home() / "Downloads" / "mobile-game-ltv-forecasting-challenge.zip"))
    args = parser.parse_args()
    train_final_model(Path(args.project_root), Path(args.zip_path))


if __name__ == "__main__":
    main()
