# Provider Directory Control Tower - MVP Run Summary

## What this prototype demonstrates

This MVP takes provider records and returns structured validation recommendations using NPI Registry and CMS public data: field-level changes, old/new values, confidence scores, supporting sources, overall confidence, recommended action, reason, and audit trail.

## Run results

- Provider records processed: 20
- Action counts: {'no_change': 19, 'human_review': 1}

## Safety design

- Final decisions are deterministic: source reliability, source agreement, entity match, recency, and field safety are scored separately.
- Conflicting sources or high-risk fields are routed to human review.
- CMS FFS provider name and specialty are confirmation context, not automatic rewrite sources.
- CMS Revoked signals are forced to human review.
- Deactivation or inactive-provider signals should never be auto-applied without review.

## Production integration plan

The MVP uses a JSONL repository as a stand-in for an operational provider directory store. In production, the repository can be replaced by a read-only database connector, while recommendations are written to staging tables such as update_candidates, review_queue, outreach_queue, and audit_log.
