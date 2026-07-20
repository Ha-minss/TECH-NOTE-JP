from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .models import ProviderRecord, Recommendation
from .utils import read_jsonl, write_jsonl


class DirectoryRepository:
    """Interface used by the MVP and by future production DB connectors."""

    def load_records(self) -> List[ProviderRecord]:
        raise NotImplementedError

    def write_recommendations(self, recommendations: Iterable[Recommendation]) -> None:
        raise NotImplementedError


class JsonlDirectoryRepository(DirectoryRepository):
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_records(self) -> List[ProviderRecord]:
        rows = read_jsonl(self.input_path)
        return [ProviderRecord(**row) for row in rows]

    def write_recommendations(self, recommendations: Iterable[Recommendation]) -> None:
        write_jsonl(self.output_path, recommendations)


class CsvDirectoryRepository(DirectoryRepository):
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_records(self) -> List[ProviderRecord]:
        records: List[ProviderRecord] = []
        with open(self.input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(ProviderRecord(**row))
        return records

    def write_recommendations(self, recommendations: Iterable[Recommendation]) -> None:
        rows = [r.model_dump() for r in recommendations]
        fieldnames = [
            "provider_id", "npi", "change_detected", "overall_confidence",
            "recommended_action", "reason", "audit_id", "changes"
        ]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row["changes"] = str(row.get("changes", []))
                writer.writerow({k: row.get(k, "") for k in fieldnames})


class PostgresDirectoryRepository(DirectoryRepository):
    """
    Production placeholder.
    Replace JsonlDirectoryRepository with this connector after HealthLynked provides
    a read-only replica or staging DB credentials.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def load_records(self) -> List[ProviderRecord]:
        raise NotImplementedError("Implement SELECT from HealthLynked read-only/staging DB.")

    def write_recommendations(self, recommendations: Iterable[Recommendation]) -> None:
        raise NotImplementedError("Implement INSERT into update_candidates/review_queue/audit tables.")
