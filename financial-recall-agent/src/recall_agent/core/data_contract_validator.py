"""Runtime Data Contract validation for demo CSV adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.recall_agent.core.artifact_hash import resolve_project_path


class DataContractValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractValidationError(message)


def _read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def _validate_pk(path: Path, table_name: str, pk_columns: list[str]) -> int:
    if not pk_columns:
        return 0
    frame = pd.read_csv(path, usecols=pk_columns)
    duplicate_count = int(frame.duplicated(subset=pk_columns).sum())
    _require(duplicate_count == 0, f"Primary key duplicate rows found in {table_name}: {duplicate_count}")
    return duplicate_count


def _validate_non_negative_amounts(path: Path, table_name: str, columns: list[str]) -> list[str]:
    amount_columns = [
        col for col in columns
        if "amount" in col.lower() and "original" not in col.lower()
    ]
    checked: list[str] = []
    if not amount_columns:
        return checked
    frame = pd.read_csv(path, usecols=amount_columns)
    for col in amount_columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            min_value = numeric.min()
            _require(min_value >= 0, f"Negative amount detected in {table_name}.{col}: {min_value}")
            checked.append(col)
    return checked


def validate_data_contract(
    *,
    contract: dict[str, Any],
    dataset_base_path: str | Path,
    expected_data_contract_id: str | None = None,
) -> dict[str, Any]:
    """Validate required CSV files, columns, primary keys and basic amount sanity."""
    if expected_data_contract_id is not None:
        _require(
            contract.get("data_contract_id") == expected_data_contract_id,
            "Data contract id does not match rule execution contract.",
        )

    base_path = resolve_project_path(dataset_base_path)
    _require(base_path.exists(), f"Dataset base path not found: {base_path}")

    prohibited = set(contract.get("prohibited_runtime_sources", []))
    tables = contract.get("tables") or {}
    _require(bool(tables), "Data contract has no tables.")

    table_results: list[dict[str, Any]] = []
    registered_files: set[str] = set()
    for table_name, table_contract in tables.items():
        file_name = table_contract.get("file_name")
        _require(bool(file_name), f"Data contract table {table_name} has no file_name.")
        _require(file_name not in prohibited, f"Prohibited source is registered as runtime table: {file_name}")
        registered_files.add(str(file_name))

        csv_path = base_path / str(file_name)
        _require(csv_path.exists(), f"Required CSV missing for {table_name}: {csv_path}")

        actual_columns = set(_read_header(csv_path))
        required_columns = set(table_contract.get("required_columns") or [])
        missing = sorted(required_columns - actual_columns)
        _require(not missing, f"Required columns missing in {table_name}: {missing}")

        pk_columns = [str(col) for col in table_contract.get("primary_key") or []]
        _validate_pk(csv_path, str(table_name), pk_columns)
        amount_columns_checked = _validate_non_negative_amounts(
            csv_path,
            str(table_name),
            sorted(actual_columns),
        )
        table_results.append(
            {
                "table": table_name,
                "file_name": file_name,
                "required_columns_count": len(required_columns),
                "primary_key": pk_columns,
                "amount_columns_checked": amount_columns_checked,
            }
        )

    registered_prohibited = sorted(prohibited & registered_files)
    _require(not registered_prohibited, f"Prohibited runtime files registered: {registered_prohibited}")

    return {
        "data_contract_id": contract.get("data_contract_id"),
        "dataset_base_path": str(base_path),
        "validated_table_count": len(table_results),
        "tables": table_results,
        "prohibited_runtime_sources": sorted(prohibited),
        "registered_prohibited_sources": registered_prohibited,
    }
