"""Hash utilities for approved Financial Recall runtime artifacts.

All execution-critical artifacts are addressed by explicit SHA-256 hashes.
JSON artifacts are hashed in canonical form so whitespace/key order changes do not
accidentally change the approval fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, *, project_root: str | Path | None = None) -> Path:
    """Resolve project-relative paths used by bundles and registry assets."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(project_root).expanduser().resolve() if project_root else PROJECT_ROOT
    return (root / candidate).resolve()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, project_root: str | Path | None = None) -> str:
    """Return the SHA-256 of a raw file's bytes."""
    resolved = resolve_project_path(path, project_root=project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact not found for hashing: {resolved}")
    return sha256_bytes(resolved.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data in a stable canonical representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json_canonical(value: Any) -> str:
    """Return SHA-256 over canonical JSON bytes."""
    return sha256_bytes(canonical_json_bytes(value))


def sha256_json_file(path: str | Path, *, project_root: str | Path | None = None) -> str:
    """Return canonical JSON SHA-256 for a JSON file."""
    resolved = resolve_project_path(path, project_root=project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"JSON artifact not found for hashing: {resolved}")
    return sha256_json_canonical(json.loads(resolved.read_text(encoding="utf-8")))
