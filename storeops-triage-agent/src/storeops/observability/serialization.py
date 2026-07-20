"""JSON serialization helpers for trace records."""

from __future__ import annotations

import json


def trace_to_json(trace_record) -> str:
    return json.dumps(trace_record.as_dict(), ensure_ascii=False, indent=2)


__all__ = ["trace_to_json"]
