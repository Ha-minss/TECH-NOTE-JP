from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.duckdb
pytest.importorskip("duckdb")

from src.recall_agent.application import run_rule
from src.recall_agent.interfaces.cli.h07_reward_missing_demo import (
    H07_BUNDLE_PATH,
    H07_PRODUCT_CONFIG_ID,
)


def _pick_complaint_id() -> str:
    complaints = pd.read_csv(
        Path("data/demo/datasets/jb_h07_synthetic_dataset_v3")
        / "complaints.csv"
    )
    return str(complaints.iloc[0]["complaint_id"])


def test_generic_run_rule_dispatches_to_reward_handler(tmp_path):
    report = run_rule(
        rule_id="H07-REWARD-MISSING-TEMPLATE",
        complaint_id=_pick_complaint_id(),
        product_config_id=H07_PRODUCT_CONFIG_ID,
        bundle_path=H07_BUNDLE_PATH,
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )

    assert report["runner_type"] == "GENERIC_RUN_RULE"
    assert report["handler_name"] == "RewardRecallHandler"
    assert report["rule_template_id"] == "H07_REWARD_MISSING"
    assert report["decision"]["automatic_refund_allowed"] is False
    assert report["audit"]["used_private_ground_truth"] is False



def test_policy_basis_is_not_hardcoded_in_runner(tmp_path):
    report = run_rule(
        rule_id="H07-REWARD-MISSING-TEMPLATE",
        complaint_id=_pick_complaint_id(),
        product_config_id=H07_PRODUCT_CONFIG_ID,
        bundle_path=H07_BUNDLE_PATH,
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )

    basis_ids = {row["basis_id"] for row in report["policy_basis"]}
    assert {
        "H07-BASIS-RATE",
        "H07-BASIS-MONTHLY-CAP",
        "H07-BASIS-EXCLUSION",
        "H07-BASIS-PAYMENT-DATE",
    } <= basis_ids



