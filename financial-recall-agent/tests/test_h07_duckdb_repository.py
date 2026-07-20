from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.duckdb
pytest.importorskip("duckdb")

from src.recall_agent.composition import PLUGIN_REGISTRY
from src.recall_agent.core.models import ExecutionRequest
from src.recall_agent.core.rule_runner import prepare_execution_context
from src.recall_agent.interfaces.cli.h07_reward_missing_demo import (
    H07_BUNDLE_PATH,
    H07_PRODUCT_CONFIG_ID,
    H07_RULE_ID,
)
from src.recall_agent.templates.reward_missing.repository import build_reconciled_result


def _result():
    context, _ = prepare_execution_context(
        ExecutionRequest(
            rule_id=H07_RULE_ID,
            complaint_id="EVAL_BASE_0001",
            product_config_id=H07_PRODUCT_CONFIG_ID,
            bundle_path=Path(H07_BUNDLE_PATH),
        ),
        PLUGIN_REGISTRY,
    )
    return build_reconciled_result(context)


def test_repository_detects_all_injected_error_types():
    df = _result()
    joined = "|".join(df["detected_error_types"].dropna().astype(str))
    assert "EXPECTED_REWARD_ROW_MISSING" in joined
    assert "REWARD_LEDGER_ROW_MISSING" in joined
    assert "POLICY_CALCULATION_ERROR" in joined


def test_repository_preserves_card_purchase_rows():
    purchases = pd.read_csv(
        Path("data/demo/datasets/jb_h07_synthetic_dataset_v3")
        / "card_purchases.csv"
    )
    expected = len(purchases[purchases["product_id"] == "JB_SMART_CASHBACK_CHECK"])
    assert len(_result()) == expected


def test_repository_preserves_missing_expected_and_ledger_rows():
    df = _result()
    assert df["reward_id"].isna().any()
    assert df["ledger_id"].isna().any()
    assert df["is_expected_reward_row_missing"].fillna(False).any()
    assert df["is_reward_ledger_row_missing"].fillna(False).any()


