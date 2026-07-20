import json
import tempfile
import unittest
from pathlib import Path

from storeops.evals.datasets import default_dataset_path, load_golden_cases
from storeops.evals.deterministic import DeterministicEvaluator, default_fixture_db_path
from storeops.evals.llm_runner import run_llm_evaluation
from storeops.evals.runner import run_full_evaluation
from storeops.infra.database import open_database


class SubmissionPipelineTests(unittest.TestCase):
    def test_default_dataset_and_fixture_are_synthetic_50(self):
        self.assertEqual(default_dataset_path().name, "offline_payment_ops_cases_50.json")
        self.assertEqual(default_fixture_db_path().name, "offline_payment_ops_synthetic_50.sqlite3")

        cases = load_golden_cases()

        self.assertEqual(len(cases), 50)
        self.assertEqual(cases[0].fixture_key, "SYN-001")
        self.assertEqual(cases[0].script_key, "S1")

    def test_fixture_db_contains_synthetic_store_mapping(self):
        connection = open_database(default_fixture_db_path())
        try:
            row = connection.execute(
                "SELECT store_id FROM scenario_stores WHERE scenario_id = ?",
                ("SYN-001",),
            ).fetchone()
        finally:
            connection.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["store_id"], "STR-SYN-001")

    def test_deterministic_evaluation_runs_against_submission_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_full_evaluation(output_dir=Path(tmpdir) / "deterministic")

            self.assertEqual(report.summary["total_cases"], 50)
            self.assertEqual(report.summary["passed_cases"], 38)
            self.assertEqual(report.summary["unsupported_claim_count"], 0)

    def test_scripted_llm_smoke_uses_script_key_with_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_llm_evaluation(
                output_dir=Path(tmpdir) / "llm_smoke",
                provider="scripted",
                fixture_key="SYN-001",
            )
            summary_path = Path(tmpdir) / "llm_smoke" / "summary.json"

            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_cases"], 1)
            self.assertEqual(summary["state_accuracy"], 1.0)
            self.assertEqual(report.summary["cause_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
