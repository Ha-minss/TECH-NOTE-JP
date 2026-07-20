import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PipelineMakefileTests(unittest.TestCase):
    def test_makefile_exposes_final_pipeline_targets(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        for target in ["all", "validate", "features", "train", "predict", "business", "test", "experiments", "clean"]:
            self.assertIn(f"{target}:", makefile)
        all_line = next(line for line in makefile.splitlines() if line.startswith("all:"))
        self.assertIn("validate", all_line)
        self.assertIn("features", all_line)
        self.assertIn("train", all_line)
        self.assertIn("predict", all_line)
        self.assertIn("business", all_line)
        self.assertNotIn("experiments", all_line)

    def test_pipeline_run_all_documents_final_steps_without_experiments(self):
        runner = (ROOT / "src" / "pipeline" / "run_all.py").read_text(encoding="utf-8")

        for step in [
            "raw data validation",
            "feature build",
            "model input validation",
            "final two-stage model train/refit",
            "test-like prediction artifact",
            "business analysis report",
            "model card",
        ]:
            self.assertIn(step, runner)
        self.assertNotIn("tune_xgboost_optuna", runner)
        self.assertNotIn("train_baseline", runner)

    def test_readme_positions_make_all_as_main_and_experiments_as_optional(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("make all", readme)
        self.assertIn("make experiments", readme)
        self.assertIn("optional", readme.lower())
        self.assertIn("production-style", readme.lower())
        self.assertIn("Model Selection Evidence", readme)
        self.assertNotIn("make kaggle-submission", readme)


if __name__ == "__main__":
    unittest.main()
