from __future__ import annotations

"""CMS provider-data connector (v08 whitelist + aggregate version).

Why this version exists
-----------------------
v06 tried to pick a CMS dataset by keyword score from data.cms.gov/data.json.
That was too broad and often selected enrollment/revalidation datasets that are
not useful for provider-directory verification.

v07 uses a safer approach:
- fixed, purpose-aligned CMS datasets for the MVP;
- dataset-specific field mappings;
- optional catalog discovery only for known titles whose UUIDs may change;
- detailed diagnostics so zero-evidence runs are explainable.

Important: CMS evidence has different roles depending on the dataset.
- Medicare FFS Public Provider Enrollment: Medicare enrollment / provider type / state evidence.
- Hospital/FQHC/Hospice/OTP datasets: organization/location evidence when NPI is present.
- Revoked Medicare Providers and Suppliers: high-risk status signal.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from .models import EvidenceItem, ProviderRecord
from .normalize import normalize_field_value, normalize_phone
from .utils import now_iso


CMS_DATA_CATALOG_URL = "https://data.cms.gov/data.json"


@dataclass(frozen=True)
class CMSDatasetConfig:
    source_name: str
    role: str
    title_contains: Sequence[str]
    endpoint: str = ""
    landing_contains: Sequence[str] = field(default_factory=tuple)
    exclude_contains: Sequence[str] = field(default_factory=tuple)
    npi_fields: Sequence[str] = ("NPI",)
    provider_name_mode: str = "none"  # none | person | direct
    direct_provider_name_fields: Sequence[str] = field(default_factory=tuple)
    org_fields: Sequence[str] = field(default_factory=tuple)
    practice_fields: Sequence[str] = field(default_factory=tuple)
    specialty_fields: Sequence[str] = field(default_factory=tuple)
    address_fields: Sequence[str] = field(default_factory=tuple)
    phone_fields: Sequence[str] = field(default_factory=tuple)
    state_fields: Sequence[str] = field(default_factory=tuple)
    revoked: bool = False
    max_rows: int = 10


# Known endpoints confirmed during the v07 CMS investigation.
# FQHC Enrollments is intentionally discovered by title because the user found
# the data dictionary but v06 matched the wrong "All Owners" dataset by name.
CMS_DATASETS: List[CMSDatasetConfig] = [
    CMSDatasetConfig(
        source_name="CMS Medicare FFS Public Provider Enrollment",
        role="medicare_enrollment_provider_type",
        title_contains=("Medicare Fee-For-Service", "Public Provider Enrollment"),
        endpoint="https://data.cms.gov/data-api/v1/dataset/2457ea29-fc82-48b0-86ec-3b0755de7515/data",
        npi_fields=("NPI",),
        provider_name_mode="person",
        org_fields=("ORG_NAME",),
        specialty_fields=("PROVIDER_TYPE_DESC",),
        state_fields=("STATE_CD",),
        max_rows=10,
    ),
    CMSDatasetConfig(
        source_name="CMS Hospital Enrollments",
        role="hospital_location",
        title_contains=("Hospital Enrollments",),
        endpoint="https://data.cms.gov/data-api/v1/dataset/f6f6505c-e8b0-4d57-b258-e2b94133aaf2/data",
        npi_fields=("NPI",),
        org_fields=("ORGANIZATION NAME",),
        practice_fields=("DOING BUSINESS AS NAME",),
        specialty_fields=("PROVIDER TYPE TEXT",),
        address_fields=("ADDRESS LINE 1", "ADDRESS LINE 2", "CITY", "STATE", "ZIP CODE"),
        state_fields=("STATE", "ENROLLMENT STATE"),
        max_rows=5,
    ),
    CMSDatasetConfig(
        source_name="CMS FQHC Enrollments",
        role="fqhc_location_phone",
        title_contains=("Federally Qualified Health Center", "Enrollments"),
        landing_contains=("federally-qualified-health-center", "enroll"),
        exclude_contains=("owner", "all-owners", "ownership"),
        npi_fields=("NPI",),
        org_fields=("ORGANIZATION NAME",),
        practice_fields=("DOING BUSINESS AS NAME",),
        specialty_fields=("PROVIDER TYPE TEXT",),
        address_fields=("ADDRESS LINE 1", "ADDRESS LINE 2", "CITY", "STATE", "ZIP CODE"),
        phone_fields=("TELEPHONE NUMBER",),
        state_fields=("STATE", "ENROLLMENT STATE"),
        max_rows=5,
    ),
    CMSDatasetConfig(
        source_name="CMS Hospice Enrollments",
        role="hospice_location",
        title_contains=("Hospice Enrollments",),
        endpoint="https://data.cms.gov/data-api/v1/dataset/25704213-e833-4b8b-9dbc-58dd17149209/data",
        npi_fields=("NPI",),
        org_fields=("ORGANIZATION NAME",),
        practice_fields=("DOING BUSINESS AS NAME",),
        specialty_fields=("PROVIDER TYPE TEXT",),
        address_fields=("ADDRESS LINE 1", "ADDRESS LINE 2", "CITY", "STATE", "ZIP CODE"),
        state_fields=("STATE", "ENROLLMENT STATE"),
        max_rows=5,
    ),
    CMSDatasetConfig(
        source_name="CMS Opioid Treatment Program Providers",
        role="otp_location_phone",
        title_contains=("Opioid Treatment Program Providers",),
        endpoint="https://data.cms.gov/data-api/v1/dataset/f1a8c197-b53d-4c24-9770-aea5d5a97dfb/data",
        npi_fields=("NPI",),
        provider_name_mode="direct",
        direct_provider_name_fields=("PROVIDER NAME",),
        practice_fields=("PROVIDER NAME",),
        address_fields=("ADDRESS LINE 1", "ADDRESS LINE 2", "CITY", "STATE", "ZIP"),
        phone_fields=("PHONE",),
        state_fields=("STATE",),
        max_rows=5,
    ),
    CMSDatasetConfig(
        source_name="CMS Revoked Medicare Providers and Suppliers",
        role="revoked_status",
        title_contains=("Revoked Medicare Providers and Suppliers",),
        endpoint="https://data.cms.gov/data-api/v1/dataset/a6496a7d-4e19-479a-a9ad-d4c0a49e07c3/data",
        npi_fields=("NPI",),
        provider_name_mode="person",
        org_fields=("ORG_NAME",),
        specialty_fields=("PROVIDER_TYPE_DESC",),
        state_fields=("STATE_CD",),
        revoked=True,
        max_rows=10,
    ),
]


class CMSCareCompareClient:
    """Purpose-aligned CMS provider-data connector keyed by NPI."""

    DATA_CATALOG_URL = CMS_DATA_CATALOG_URL

    def __init__(
        self,
        source_reliability: Dict[str, float],
        data_url: Optional[str] = None,
        timeout: int = 20,
        max_rows: int = 10,
        max_candidate_endpoints: int = 12,
        include_extended_sources: bool = False,
        source_mode: str = "minimal",
    ):
        self.source_reliability = source_reliability
        # Kept for backward compatibility with older config/env. In v07 this is
        # optional and is not used as a generic search seed.
        self.data_url = (data_url or "").strip()
        self.timeout = timeout
        self.max_rows = max_rows
        self.max_candidate_endpoints = max_candidate_endpoints
        self.include_extended_sources = include_extended_sources
        self.source_mode = (source_mode or "minimal").lower().strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ProviderDirectoryControlTower/0.8 (+https://example.org/provider-directory-mvp)",
            "Accept": "application/json,text/plain,*/*",
        })
        self.last_debug: Dict[str, Any] = {}
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        self._resolved_endpoint_cache: Dict[str, Dict[str, str]] = {}

    def collect(self, record: ProviderRecord) -> List[EvidenceItem]:
        npi = re.sub(r"\D", "", record.npi or "")
        debug: Dict[str, Any] = {
            "connector": "CMS Provider Data Whitelist",
            "provider_id": record.provider_id,
            "npi": npi,
            "configured_data_url": self.data_url,
            "source_mode": self.source_mode,
            "strategy": "v08_whitelist_aggregate_dataset_configs",
            "dataset_runs": [],
            "evidence_count": 0,
            "reason": "",
        }
        self.last_debug = debug

        if len(npi) != 10:
            debug["reason"] = "invalid_or_missing_npi"
            return []

        evidence: List[EvidenceItem] = []
        for cfg in self._active_dataset_configs():
            run_debug = self._run_dataset(cfg, npi)
            debug["dataset_runs"].append(run_debug)
            evidence.extend(run_debug.pop("_evidence", []))

        debug["evidence_count"] = len(evidence)
        debug["sources_with_rows"] = [r["source_name"] for r in debug["dataset_runs"] if r.get("rows_found", 0) > 0]
        debug["sources_with_evidence"] = [r["source_name"] for r in debug["dataset_runs"] if r.get("evidence_count", 0) > 0]

        if evidence:
            debug["reason"] = "ok"
        elif any(r.get("status") == "error" for r in debug["dataset_runs"]):
            debug["reason"] = "cms_partial_or_network_error"
        elif any(r.get("rows_found", 0) > 0 for r in debug["dataset_runs"]):
            debug["reason"] = "cms_rows_found_but_no_supported_fields_extracted"
        else:
            debug["reason"] = "no_cms_rows_found_for_npi_in_whitelist"
        return evidence

    def _active_dataset_configs(self) -> List[CMSDatasetConfig]:
        """Cost-aware CMS source selector.

        At million-record scale, querying every CMS dataset for every provider is
        expensive/slow. Minimal mode uses the two highest-leverage individual
        provider sources: FFS enrollment context and Revoked risk signal. Facility
        datasets can be enabled for organization/location verification.
        """
        mode = self.source_mode
        if mode == "all":
            return CMS_DATASETS
        if mode == "ffs":
            return [c for c in CMS_DATASETS if c.role == "medicare_enrollment_provider_type"]
        if mode == "revoked":
            return [c for c in CMS_DATASETS if c.role == "revoked_status"]
        if mode == "facility":
            return [c for c in CMS_DATASETS if c.role in {"hospital_location", "fqhc_location_phone", "hospice_location", "otp_location_phone", "revoked_status"}]
        # Default minimal: good for NPI-1 clinician records and cheap enough for
        # large batches.
        return [c for c in CMS_DATASETS if c.role in {"medicare_enrollment_provider_type", "revoked_status"}]

    def _run_dataset(self, cfg: CMSDatasetConfig, npi: str) -> Dict[str, Any]:
        run: Dict[str, Any] = {
            "source_name": cfg.source_name,
            "role": cfg.role,
            "status": "not_run",
            "endpoint": cfg.endpoint,
            "endpoint_source": "built_in" if cfg.endpoint else "catalog_lookup",
            "query_attempts": [],
            "rows_found": 0,
            "evidence_count": 0,
            "reason": "",
            "_evidence": [],
        }

        endpoint_info = self._resolve_endpoint(cfg)
        endpoint = endpoint_info.get("url", "")
        if not endpoint:
            run.update({
                "status": "skipped",
                "reason": "cms_endpoint_not_found_for_whitelisted_dataset",
                "endpoint_lookup": endpoint_info,
            })
            return run

        run["endpoint"] = endpoint
        run["endpoint_source"] = endpoint_info.get("source", run["endpoint_source"])
        run["dataset_title"] = endpoint_info.get("title", "")
        run["landingPage"] = endpoint_info.get("landingPage", "")

        rows = self._query_by_npi(endpoint, npi, cfg, run)
        run["rows_found"] = len(rows)
        if not rows:
            run["status"] = "ok"
            run["reason"] = "no_rows_for_npi"
            return run

        # v08: FFS and Revoked need special handling.
        # FFS often has multiple rows per NPI (multiple enrollment IDs, states, specialties).
        # It should be aggregated and treated as confirmation/provider-type evidence, not
        # as a direct patient-facing name/specialty update source.
        if cfg.role == "medicare_enrollment_provider_type":
            out = self._build_ffs_aggregate_evidence(cfg, rows, npi, endpoint, run)
            run["_evidence"] = out
            run["evidence_count"] = len(out)
            run["status"] = "ok"
            run["reason"] = "ok" if out else "rows_found_but_no_supported_fields_extracted"
            return run

        # Revoked source's actionable signal is active_status=revoked. Name/specialty
        # from this dataset may be useful context but should not create rewrite changes.
        if cfg.revoked:
            out = self._build_revoked_evidence(cfg, rows, npi, endpoint, run)
            run["_evidence"] = out
            run["evidence_count"] = len(out)
            run["status"] = "ok"
            run["reason"] = "ok" if out else "rows_found_but_no_supported_fields_extracted"
            return run

        collected = now_iso()
        out: List[EvidenceItem] = []
        trust = self._trust(cfg.source_name)
        for row_index, row in enumerate(rows[: min(cfg.max_rows, self.max_rows)]):
            row_norm = self._normalise_row_keys(row)
            if row_index == 0:
                run["sample_row_keys"] = list(row.keys())
                run["sample_row_keys_normalized"] = sorted(row_norm.keys())
                run["sample_row_preview"] = {k: str(v)[:120] for k, v in list(row.items())[:30]}

            base_meta = {
                "matched_by": "npi",
                "row_index": row_index,
                "cms_dataset_role": cfg.role,
                "cms_dataset_title": run.get("dataset_title", ""),
            }

            # Always record NPI evidence for matched CMS rows.
            out.append(self._ev(
                cfg=cfg,
                field="npi",
                value=npi,
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=f"{cfg.source_name} contains a row for NPI {npi}.",
                metadata={**base_meta, "evidence_role": "entity_match"},
            ))

            provider_name = self._provider_name(row_norm, cfg)
            if provider_name:
                out.append(self._ev(
                    cfg=cfg,
                    field="provider_name",
                    value=provider_name,
                    trust=trust,
                    collected=collected,
                    source_url=endpoint,
                    text=f"{cfg.source_name} provider name: {provider_name}.",
                    metadata={**base_meta, "decision_policy": "confirm_only"},
                ))

            # Organization / DBA is safest as practice_name in the current schema.
            practice_name = self._first(row_norm, self._norm_keys(cfg.practice_fields)) or self._first(row_norm, self._norm_keys(cfg.org_fields))
            if practice_name:
                out.append(self._ev(
                    cfg=cfg,
                    field="practice_name",
                    value=practice_name,
                    trust=trust,
                    collected=collected,
                    source_url=endpoint,
                    text=f"{cfg.source_name} organization/practice name: {practice_name}.",
                    metadata=base_meta,
                ))

            specialty = self._first(row_norm, self._norm_keys(cfg.specialty_fields))
            if specialty:
                out.append(self._ev(
                    cfg=cfg,
                    field="specialty",
                    value=specialty,
                    trust=trust,
                    collected=collected,
                    source_url=endpoint,
                    text=f"{cfg.source_name} provider type/specialty: {specialty}.",
                    metadata={**base_meta, "decision_policy": "confirm_only"},
                ))

            address = self._address(row_norm, cfg)
            if address:
                out.append(self._ev(
                    cfg=cfg,
                    field="address",
                    value=address,
                    trust=trust,
                    collected=collected,
                    source_url=endpoint,
                    text=f"{cfg.source_name} location address: {address}.",
                    metadata=base_meta,
                ))

            phone_raw = self._first(row_norm, self._norm_keys(cfg.phone_fields))
            phone_norm = normalize_phone(phone_raw)
            if phone_norm:
                out.append(self._ev(
                    cfg=cfg,
                    field="phone",
                    value=phone_norm,
                    trust=trust,
                    collected=collected,
                    source_url=endpoint,
                    text=f"{cfg.source_name} phone: {phone_norm}.",
                    metadata={**base_meta, "raw_phone": phone_raw, "phone_extension": self._phone_extension(phone_raw)},
                ))

        run["_evidence"] = out
        run["evidence_count"] = len(out)
        run["status"] = "ok"
        run["reason"] = "ok" if out else "rows_found_but_no_supported_fields_extracted"
        return run

    def _build_ffs_aggregate_evidence(
        self,
        cfg: CMSDatasetConfig,
        rows: List[Dict[str, Any]],
        npi: str,
        endpoint: str,
        run: Dict[str, Any],
    ) -> List[EvidenceItem]:
        collected = now_iso()
        trust = self._trust(cfg.source_name)
        row_norms = [self._normalise_row_keys(r) for r in rows[: min(cfg.max_rows, self.max_rows)]]
        if rows:
            run["sample_row_keys"] = list(rows[0].keys())
            run["sample_row_keys_normalized"] = sorted(row_norms[0].keys()) if row_norms else []
            run["sample_row_preview"] = {k: str(v)[:120] for k, v in list(rows[0].items())[:30]}

        names = sorted({self._provider_name(r, cfg) for r in row_norms if self._provider_name(r, cfg)})
        org_names = sorted({self._first(r, self._norm_keys(cfg.org_fields)) for r in row_norms if self._first(r, self._norm_keys(cfg.org_fields))})
        specialties = sorted({self._first(r, self._norm_keys(cfg.specialty_fields)) for r in row_norms if self._first(r, self._norm_keys(cfg.specialty_fields))})
        states = sorted({self._first(r, self._norm_keys(cfg.state_fields)) for r in row_norms if self._first(r, self._norm_keys(cfg.state_fields))})
        enrollment_ids = sorted({self._first(r, ["enrlmt_id", "enrollment_id"]) for r in row_norms if self._first(r, ["enrlmt_id", "enrollment_id"])})

        base_meta = {
            "matched_by": "npi",
            "cms_dataset_role": cfg.role,
            "cms_dataset_title": run.get("dataset_title", ""),
            "aggregation": "by_npi",
            "row_count": len(rows),
            "enrollment_ids": enrollment_ids[:25],
            "states": states,
            "provider_types": specialties,
            "decision_policy": "confirm_only",
        }

        out: List[EvidenceItem] = []
        out.append(self._ev(
            cfg=cfg,
            field="npi",
            value=npi,
            trust=trust,
            collected=collected,
            source_url=endpoint,
            text=f"{cfg.source_name} contains {len(rows)} enrollment row(s) for NPI {npi}.",
            metadata={**base_meta, "evidence_role": "entity_match", "decision_policy": "entity_match"},
        ))

        # Provider name is confirmation context only. Use one canonical display value
        # but preserve all CMS names in metadata/audit.
        display_name = names[0] if names else (org_names[0] if org_names else "")
        if display_name:
            out.append(self._ev(
                cfg=cfg,
                field="provider_name",
                value=display_name,
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=f"{cfg.source_name} confirms CMS enrollment identity name(s): {', '.join((names or org_names)[:5])}.",
                metadata={**base_meta, "all_names": names, "all_org_names": org_names, "evidence_role": "identity_context"},
            ))

        # Specialty/provider type is not a directory overwrite source. It is Medicare
        # enrollment context; multiple rows per NPI are normal.
        if specialties:
            out.append(self._ev(
                cfg=cfg,
                field="specialty",
                value=" | ".join(specialties[:8]),
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=f"{cfg.source_name} enrollment provider type(s): {', '.join(specialties[:8])}.",
                metadata={**base_meta, "evidence_role": "provider_type_context"},
            ))

        # Keep state as metadata because ProviderRecord currently has no state field.
        if states:
            out.append(self._ev(
                cfg=cfg,
                field="cms_enrollment_state",
                value="|".join(states),
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=f"{cfg.source_name} enrollment state(s): {', '.join(states)}.",
                metadata={**base_meta, "evidence_role": "state_context"},
            ))

        return out

    def _build_revoked_evidence(
        self,
        cfg: CMSDatasetConfig,
        rows: List[Dict[str, Any]],
        npi: str,
        endpoint: str,
        run: Dict[str, Any],
    ) -> List[EvidenceItem]:
        collected = now_iso()
        trust = max(self._trust(cfg.source_name), 0.94)
        row_norms = [self._normalise_row_keys(r) for r in rows[: min(cfg.max_rows, self.max_rows)]]
        if rows:
            run["sample_row_keys"] = list(rows[0].keys())
            run["sample_row_keys_normalized"] = sorted(row_norms[0].keys()) if row_norms else []
            run["sample_row_preview"] = {k: str(v)[:120] for k, v in list(rows[0].items())[:30]}

        first = row_norms[0] if row_norms else {}
        reason = self._first(first, ["revocation_rsn", "revocation_reason"])
        effective = self._first(first, ["revocation_efctv_dt", "revocation_effective_date"])
        bar = self._first(first, ["reenrollment_bar_exprtn_dt", "reenrollment_bar_expiration"])
        name = self._provider_name(first, cfg) or self._first(first, self._norm_keys(cfg.org_fields))
        specialty = self._first(first, self._norm_keys(cfg.specialty_fields))

        base_meta = {
            "matched_by": "npi",
            "cms_dataset_role": cfg.role,
            "cms_dataset_title": run.get("dataset_title", ""),
            "row_count": len(rows),
            "revocation_reason": reason,
            "revocation_effective_date": effective,
            "reenrollment_bar_expiration": bar,
            "cms_name_context": name,
            "cms_provider_type_context": specialty,
            "force_review": True,
        }

        return [
            self._ev(
                cfg=cfg,
                field="npi",
                value=npi,
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=f"{cfg.source_name} contains a revoked-provider row for NPI {npi}.",
                metadata={**base_meta, "evidence_role": "entity_match", "decision_policy": "entity_match"},
            ),
            self._ev(
                cfg=cfg,
                field="active_status",
                value="revoked",
                trust=trust,
                collected=collected,
                source_url=endpoint,
                text=(
                    f"{cfg.source_name} lists this NPI as revoked"
                    + (f"; reason: {reason}" if reason else "")
                    + (f"; effective date: {effective}" if effective else "")
                    + (f"; re-enrollment bar expiration: {bar}" if bar else "")
                    + "."
                ),
                metadata=base_meta,
            ),
        ]

    def _resolve_endpoint(self, cfg: CMSDatasetConfig) -> Dict[str, str]:
        cache_key = cfg.source_name
        if cache_key in self._resolved_endpoint_cache:
            return dict(self._resolved_endpoint_cache[cache_key])

        if cfg.endpoint:
            info = {"url": cfg.endpoint, "source": "built_in", "title": cfg.source_name, "landingPage": ""}
            self._resolved_endpoint_cache[cache_key] = info
            return dict(info)

        # Discover only known whitelist titles from the CMS catalog.
        catalog = self._catalog()
        best: Dict[str, str] = {}
        best_score = -999
        for ds in catalog:
            title = str(ds.get("title", ""))
            desc = str(ds.get("description", ""))
            landing = str(ds.get("landingPage", ""))
            haystack = f"{title} {desc} {landing}".lower()
            if not all(term.lower() in haystack for term in cfg.title_contains):
                continue
            if cfg.landing_contains and not all(term.lower() in landing.lower() for term in cfg.landing_contains):
                continue
            if cfg.exclude_contains and any(term.lower() in haystack for term in cfg.exclude_contains):
                continue

            score = 100
            if "enrollment" in title.lower() or "enrollments" in title.lower():
                score += 20
            if "owner" in haystack:
                score -= 80
            for dist in ds.get("distribution", []) or []:
                access_url = str(dist.get("accessURL") or "")
                fmt = str(dist.get("format", "")).lower()
                if fmt != "api" or "/data-api/v1/dataset/" not in access_url or not access_url.endswith("/data"):
                    continue
                if "latest" in str(dist.get("description", "")).lower():
                    score += 10
                if score > best_score:
                    best_score = score
                    best = {"url": access_url, "source": "data_json_catalog_whitelist", "title": title, "landingPage": landing}

        self._resolved_endpoint_cache[cache_key] = best
        return dict(best)

    def _catalog(self) -> List[Dict[str, Any]]:
        if self._catalog_cache is not None:
            return self._catalog_cache
        try:
            response = self.session.get(self.DATA_CATALOG_URL, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            datasets = payload.get("dataset", []) or []
            self._catalog_cache = [ds for ds in datasets if isinstance(ds, dict)]
        except Exception:
            self._catalog_cache = []
        return self._catalog_cache

    def _query_by_npi(self, endpoint: str, npi: str, cfg: CMSDatasetConfig, run: Dict[str, Any]) -> List[Dict[str, Any]]:
        fields = list(cfg.npi_fields) or ["NPI"]
        # Some endpoints have normalized/lower variants; trying these is cheap and safe.
        extra_fields = [self._normalise_key(f) for f in fields]
        for field_name in list(dict.fromkeys(fields + extra_fields)):
            params = {f"filter[{field_name}]": npi, "size": str(min(cfg.max_rows, self.max_rows)), "offset": "0"}
            attempt = {"method": "filter", "field": field_name, "params": params, "status": "not_run"}
            run["query_attempts"].append(attempt)
            try:
                response = self.session.get(endpoint, params=params, timeout=self.timeout)
                attempt["status_code"] = response.status_code
                response.raise_for_status()
                rows = self._rows_from_payload(response.json())
                attempt["raw_rows"] = len(rows)
                matched = [r for r in rows if self._row_has_npi(r, npi)]
                attempt["matched_rows"] = len(matched)
                attempt["status"] = "ok"
                if matched:
                    return matched
            except Exception as exc:
                attempt["status"] = "error"
                attempt["error"] = repr(exc)

        # Keyword fallback for CMS endpoints that ignore unsupported filter fields.
        params = {"keyword": npi, "size": str(min(cfg.max_rows, self.max_rows)), "offset": "0"}
        attempt = {"method": "keyword", "params": params, "status": "not_run"}
        run["query_attempts"].append(attempt)
        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            attempt["status_code"] = response.status_code
            response.raise_for_status()
            rows = self._rows_from_payload(response.json())
            attempt["raw_rows"] = len(rows)
            matched = [r for r in rows if self._row_has_npi(r, npi)]
            attempt["matched_rows"] = len(matched)
            attempt["status"] = "ok"
            return matched
        except Exception as exc:
            attempt["status"] = "error"
            attempt["error"] = repr(exc)
            run["status"] = "error"
            return []

    def _ev(
        self,
        cfg: CMSDatasetConfig,
        field: str,
        value: str,
        trust: float,
        collected: str,
        source_url: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            source_name=cfg.source_name,
            source_type="cms_provider_data",
            source_url=source_url,
            field=field,
            value=value,
            normalized_value=normalize_field_value(field, value),
            evidence_text=text,
            source_confidence=trust,
            collected_at=collected,
            metadata=metadata or {},
        )

    def _trust(self, source_name: str) -> float:
        return float(
            self.source_reliability.get(
                source_name,
                self.source_reliability.get("CMS Care Compare", self.source_reliability.get("CMS/NPPES", 0.92)),
            )
        )

    @staticmethod
    def _rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            for key in ("data", "results", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [r for r in value if isinstance(r, dict)]
        return []

    @staticmethod
    def _normalise_key(key: str) -> str:
        key = key.strip().lower()
        key = key.replace("#", " number ")
        key = re.sub(r"[^a-z0-9]+", "_", key)
        return key.strip("_")

    def _normalise_row_keys(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {self._normalise_key(str(k)): v for k, v in row.items()}

    def _norm_keys(self, keys: Iterable[str]) -> List[str]:
        return [self._normalise_key(k) for k in keys]

    def _row_has_npi(self, row: Dict[str, Any], npi: str) -> bool:
        row_norm = self._normalise_row_keys(row)
        for key, value in row_norm.items():
            if "npi" in key or key in {"national_provider_identifier", "rndrng_npi"}:
                candidates = re.findall(r"\d{10}", str(value))
                if npi in candidates:
                    return True
                # Backward-compatible exact digit check for clean single-NPI cells.
                if re.sub(r"\D", "", str(value)) == npi:
                    return True
        return False

    @staticmethod
    def _phone_extension(phone_raw: str) -> str:
        m = re.search(r"(?:\b(?:x|ext|extension)\b\.?\s*)(\d{1,8})", str(phone_raw or ""), re.IGNORECASE)
        return m.group(1) if m else ""

    @staticmethod
    def _first(row: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _provider_name(self, row: Dict[str, Any], cfg: CMSDatasetConfig) -> str:
        if cfg.provider_name_mode == "direct":
            return self._first(row, self._norm_keys(cfg.direct_provider_name_fields))
        if cfg.provider_name_mode == "person":
            parts = [
                self._first(row, ["first_name"]),
                self._first(row, ["mdl_name", "middle_name"]),
                self._first(row, ["last_name"]),
            ]
            return " ".join(p for p in parts if p).strip()
        return ""

    def _address(self, row: Dict[str, Any], cfg: CMSDatasetConfig) -> str:
        keys = self._norm_keys(cfg.address_fields)
        if not keys:
            return ""
        parts = [self._first(row, [key]) for key in keys]
        # Avoid returning a state/zip-only pseudo-address.
        non_empty = [p for p in parts if p]
        if len(non_empty) < 2:
            return ""
        return ", ".join(non_empty)
