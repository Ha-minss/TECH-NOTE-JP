"""JSONL dataset utilities for Recover24 narrative evaluation.

The evaluation dataset is deliberately fact-first:
- store verified structured facts
- store the user's free-form statement
- store required fact ids and event order
- do NOT store case-specific banned sentences

Forbidden/contradicted claims are derived dynamically by validators from
structured_facts and extracted output claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    difficulty: str
    case_type: str
    structured_facts: dict[str, Any]
    raw_statement: str
    required_fact_ids: list[str]
    expected_event_order: list[str]
    input_conflicts: list[dict[str, Any]]
    fact_aliases: dict[str, list[str]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldCase":
        required = ["case_id", "structured_facts", "raw_statement"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Gold case is missing required keys: {missing}")
        return cls(
            case_id=str(data["case_id"]),
            difficulty=str(data.get("difficulty", "unknown")),
            case_type=str(data.get("case_type", "unknown")),
            structured_facts=dict(data.get("structured_facts", {})),
            raw_statement=str(data.get("raw_statement", "")),
            required_fact_ids=list(data.get("required_fact_ids", [])),
            expected_event_order=list(data.get("expected_event_order", [])),
            input_conflicts=list(data.get("input_conflicts", [])),
            fact_aliases={k: list(v) for k, v in dict(data.get("fact_aliases", {})).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "difficulty": self.difficulty,
            "case_type": self.case_type,
            "structured_facts": self.structured_facts,
            "raw_statement": self.raw_statement,
            "required_fact_ids": self.required_fact_ids,
            "expected_event_order": self.expected_event_order,
            "input_conflicts": self.input_conflicts,
            "fact_aliases": self.fact_aliases,
        }


def load_jsonl(path: str | Path) -> list[GoldCase]:
    cases: list[GoldCase] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            cases.append(GoldCase.from_dict(data))
    return cases


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
