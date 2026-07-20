"""DuckDB data access for the generic reward-missing template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.recall_agent.core.artifact_hash import resolve_project_path, sha256_file
from src.recall_agent.core.models import ExecutionContext
from src.recall_agent.templates.reward_missing.config_validator import reward_policy


RUNTIME_TABLES = {
    "customers": "customers.csv",
    "card_contracts": "card_contracts.csv",
    "merchant_master": "merchant_master.csv",
    "card_purchases": "card_purchases.csv",
    "expected_rewards": "expected_rewards.csv",
    "reward_ledger": "reward_ledger.csv",
    "reward_batch_logs": "reward_batch_logs.csv",
    "complaints": "complaints.csv",
}


def load_complaint(dataset_path: Path, complaint_id: str) -> dict[str, Any]:
    path = dataset_path / "complaints.csv"
    complaints = pd.read_csv(path)
    matched = complaints[complaints["complaint_id"].astype(str) == str(complaint_id)]
    if matched.empty:
        raise ValueError(f"Complaint not found: {complaint_id}")
    row = matched.iloc[0].to_dict()
    return {key: None if pd.isna(value) else value for key, value in row.items()}


def _register_views(con: duckdb.DuckDBPyConnection, dataset_path: Path) -> None:
    for table_name, file_name in RUNTIME_TABLES.items():
        csv_path = dataset_path / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV missing for {table_name}: {csv_path}")
        escaped = str(csv_path).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {table_name} AS "
            f"SELECT * FROM read_csv_auto('{escaped}', header=true, ignore_errors=false)"
        )


def _create_policy_tables(
    con: duckdb.DuckDBPyConnection,
    context: ExecutionContext,
) -> None:
    policy = reward_policy(dict(context.product_config))
    eligibility = policy["eligibility"]
    cap = policy["monthly_cap"]
    schedule = policy["payment_schedule"]
    con.execute(
        "CREATE OR REPLACE TEMP TABLE runtime_params("
        "product_id VARCHAR, product_config_id VARCHAR, rule_template_id VARCHAR)"
    )
    con.execute(
        "INSERT INTO runtime_params VALUES (?, ?, ?)",
        [
            context.product_config_ref.product_id,
            context.product_config_ref.config_id,
            context.rule.rule_template_id,
        ],
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE runtime_policy("
        "minimum_eligible_amount_krw BIGINT, monthly_cap_amount_krw BIGINT, payment_day INTEGER)"
    )
    con.execute(
        "INSERT INTO runtime_policy VALUES (?, ?, ?)",
        [
            int(eligibility["minimum_eligible_amount_krw"]),
            int(cap["cap_amount_krw"]),
            int(schedule["payment_day"]),
        ],
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE rate_table("
        "min_amount_inclusive BIGINT, max_amount_exclusive BIGINT, cashback_rate DOUBLE)"
    )
    for row in policy["rate_table"]:
        con.execute(
            "INSERT INTO rate_table VALUES (?, ?, ?)",
            [
                int(row["min_amount_inclusive"]),
                None if row["max_amount_exclusive"] is None else int(row["max_amount_exclusive"]),
                float(row["cashback_rate"]),
            ],
        )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE excluded_merchant_categories(merchant_category VARCHAR)"
    )
    for category in eligibility["excluded_merchant_categories"]:
        con.execute("INSERT INTO excluded_merchant_categories VALUES (?)", [str(category)])
    con.execute(
        "CREATE OR REPLACE TEMP TABLE eligible_transaction_statuses(status VARCHAR)"
    )
    for status in eligibility["eligible_transaction_statuses"]:
        con.execute("INSERT INTO eligible_transaction_statuses VALUES (?)", [str(status)])


def _execute_sql(con: duckdb.DuckDBPyConnection, context: ExecutionContext) -> None:
    if not context.sql_files:
        raise ValueError("Approved SQL list is required.")
    for sql in context.sql_files:
        path = resolve_project_path(sql.path)
        actual_hash = sha256_file(path)
        if actual_hash != sql.sha256:
            raise ValueError(
                f"Approved SQL hash mismatch before execution: {path}"
            )
        con.execute(path.read_text(encoding="utf-8"))


def build_reconciled_result(context: ExecutionContext) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    try:
        _register_views(con, context.dataset_path)
        _create_policy_tables(con, context)
        _execute_sql(con, context)
        return con.execute("SELECT * FROM h07_reconciled_result").fetchdf()
    finally:
        con.close()
