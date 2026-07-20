"""Golden dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    scenario_family: str
    merchant_message: str
    fixture_key: str
    expected_state: str
    expected_primary_cause: str | None
    acceptable_alternatives: list[str] = field(default_factory=list)
    required_evidence_ids: list[str] = field(default_factory=list)
    required_tool_names: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    notes: str = ""
    script_key: str | None = None


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "golden" / "offline_payment_ops_cases_50.json"


def load_golden_cases(path: Path | str | None = None) -> list[GoldenCase]:
    dataset_path = Path(path) if path is not None else default_dataset_path()
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [GoldenCase(**raw_case) for raw_case in raw_cases]


__all__ = ["GoldenCase", "default_dataset_path", "load_golden_cases"]
