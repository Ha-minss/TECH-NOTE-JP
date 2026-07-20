from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

from .cms import CMSCareCompareClient
from .decision import build_field_changes, build_recommendation
from .models import AuditEvent, EvidenceItem, ProviderRecord, Recommendation
from .sources import NPIRegistryClient
from .utils import now_iso


class ProviderDirectoryPipeline:
    def __init__(
        self,
        config: dict,
        use_real_npi: bool = False,
        use_cms: bool = False,
    ):
        load_dotenv()
        self.config = config
        self.use_real_npi = use_real_npi
        self.use_cms = use_cms

        reliability = config.get("source_reliability", {})
        self.npi_client = NPIRegistryClient(reliability)
        self.cms_client = CMSCareCompareClient(
            reliability,
            data_url=os.getenv("CMS_DOCTORS_CLINICIANS_DATA_URL") or config.get("cms_doctors_clinicians_data_url"),
            max_rows=int(config.get("cms_max_rows_per_npi", 10)),
            max_candidate_endpoints=int(config.get("cms_max_candidate_endpoints", 12)),
            source_mode=str(config.get("cms_source_mode", "minimal")),
        )

    def process_record(self, record: ProviderRecord) -> tuple[Recommendation, List[EvidenceItem], List[AuditEvent]]:
        audit: List[AuditEvent] = []
        evidence: List[EvidenceItem] = []

        def log(node: str, status: str, detail=None):
            audit.append(AuditEvent(
                audit_id="pending",
                provider_id=record.provider_id,
                node=node,
                status=status,
                detail=detail or {},
                timestamp=now_iso(),
            ))

        log("load_record", "ok", {"provider_id": record.provider_id, "npi": record.npi})

        if self.use_real_npi and record.npi:
            try:
                npi_evidence = self.npi_client.lookup_by_npi(record.npi)
                evidence.extend(npi_evidence)
                log("npi_registry_lookup", "ok", {"evidence_count": len(npi_evidence)})
            except Exception as exc:
                log("npi_registry_lookup", "error", {"error": repr(exc)})

        if self.use_cms and record.npi:
            try:
                cms_evidence = self.cms_client.collect(record)
                evidence.extend(cms_evidence)
                cms_detail = {"evidence_count": len(cms_evidence)}
                cms_debug = getattr(self.cms_client, "last_debug", {})
                if cms_debug:
                    cms_detail.update(cms_debug)
                log("cms_care_compare_lookup", "ok", cms_detail)
            except Exception as exc:
                log("cms_care_compare_lookup", "error", {"error": repr(exc)})

        changes = build_field_changes(record, evidence, self.config)
        log("field_verification", "ok", {"changes": [c.model_dump() for c in changes]})

        recommendation = build_recommendation(record, changes, self.config)
        log("decision_engine", "ok", recommendation.model_dump())

        for ev in audit:
            ev.audit_id = recommendation.audit_id

        return recommendation, evidence, audit

    def process_batch(self, records: List[ProviderRecord]) -> tuple[List[Recommendation], List[EvidenceItem], List[AuditEvent]]:
        recommendations: List[Recommendation] = []
        all_evidence: List[EvidenceItem] = []
        audit_log: List[AuditEvent] = []

        from collections import Counter
        npi_counts = Counter((r.npi or '').strip() for r in records if (r.npi or '').strip())

        for record in records:
            rec, ev, audit = self.process_record(record)
            if record.npi and npi_counts.get(record.npi.strip(), 0) > 1:
                rec.recommended_action = "human_review"
                rec.reason = "Potential duplicate provider record: the same NPI appears multiple times in this batch. Manual merge/review recommended."
                rec.overall_confidence = min(rec.overall_confidence, 0.75)
                audit.append(AuditEvent(
                    audit_id=rec.audit_id,
                    provider_id=record.provider_id,
                    node="duplicate_npi_guard",
                    status="human_review",
                    detail={"npi": record.npi, "batch_count": npi_counts.get(record.npi.strip(), 0)},
                    timestamp=now_iso(),
                ))
            recommendations.append(rec)
            all_evidence.extend(ev)
            audit_log.extend(audit)
        return recommendations, all_evidence, audit_log