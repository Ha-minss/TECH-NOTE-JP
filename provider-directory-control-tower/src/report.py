from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .models import AuditEvent, EvidenceItem, Recommendation
from .utils import write_json, write_jsonl


def export_recommendations(output_dir: str | Path, recommendations: List[Recommendation]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "recommendations.jsonl", recommendations)
    write_json(out / "recommendations_pretty.json", [r.model_dump() for r in recommendations], pretty=True)

    with open(out / "submission.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "provider_id", "npi", "change_detected", "overall_confidence",
            "recommended_action", "reason", "num_changes", "changed_fields", "audit_id"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in recommendations:
            writer.writerow({
                "provider_id": r.provider_id,
                "npi": r.npi,
                "change_detected": r.change_detected,
                "overall_confidence": r.overall_confidence,
                "recommended_action": r.recommended_action,
                "reason": r.reason,
                "num_changes": len(r.changes),
                "changed_fields": "|".join(c.field for c in r.changes),
                "audit_id": r.audit_id,
            })


def export_evidence(output_dir: str | Path, evidence: Iterable[EvidenceItem]) -> None:
    write_jsonl(Path(output_dir) / "evidence_packets.jsonl", evidence)


def export_audit(output_dir: str | Path, audit_log: Iterable[AuditEvent]) -> None:
    write_jsonl(Path(output_dir) / "audit_log.jsonl", audit_log)


def export_executive_summary(output_dir: str | Path, recommendations: List[Recommendation]) -> None:
    out = Path(output_dir)
    action_counts = {}
    for r in recommendations:
        action_counts[r.recommended_action] = action_counts.get(r.recommended_action, 0) + 1

    text = f"""# Provider Directory Control Tower - MVP Run Summary

## What this prototype demonstrates

This MVP takes provider records and returns structured validation recommendations using NPI Registry and CMS public data: field-level changes, old/new values, confidence scores, supporting sources, overall confidence, recommended action, reason, and audit trail.

## Run results

- Provider records processed: {len(recommendations)}
- Action counts: {action_counts}

## Safety design

- Final decisions are deterministic: source reliability, source agreement, entity match, recency, and field safety are scored separately.
- Conflicting sources or high-risk fields are routed to human review.
- CMS FFS provider name and specialty are confirmation context, not automatic rewrite sources.
- CMS Revoked signals are forced to human review.
- Deactivation or inactive-provider signals should never be auto-applied without review.

## Production integration plan

The MVP uses a JSONL repository as a stand-in for an operational provider directory store. In production, the repository can be replaced by a read-only database connector, while recommendations are written to staging tables such as update_candidates, review_queue, outreach_queue, and audit_log.
"""
    with open(out / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write(text)


def export_connector_diagnostics(output_dir: str | Path, audit_log: Iterable[AuditEvent]) -> None:
    """Write compact connector diagnostics for the NPI and CMS evidence path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for event in audit_log:
        if event.node not in {"npi_registry_lookup", "cms_care_compare_lookup"}:
            continue
        detail = event.detail or {}
        rows.append({
            "provider_id": event.provider_id,
            "node": event.node,
            "status": event.status,
            "evidence_count": detail.get("evidence_count", 0),
            "reason": detail.get("reason", ""),
            "detail": detail,
            "timestamp": event.timestamp,
        })

    write_json(out / "connector_diagnostics.json", rows, pretty=True)

    with open(out / "connector_diagnostics.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "provider_id", "node", "status", "evidence_count", "reason",
            "endpoint_used", "query", "timestamp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            detail = row.get("detail", {}) or {}
            writer.writerow({
                "provider_id": row["provider_id"],
                "node": row["node"],
                "status": row["status"],
                "evidence_count": row["evidence_count"],
                "reason": row["reason"],
                "endpoint_used": detail.get("endpoint_used", detail.get("endpoint", "")),
                "query": detail.get("query", row["provider_id"]),
                "timestamp": row["timestamp"],
            })