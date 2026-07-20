from __future__ import annotations

import re
from typing import Dict, List

import requests

from .models import EvidenceItem
from .normalize import normalize_active_status, normalize_field_value, normalize_phone
from .npi import (
    NPI_API_BASE_URL,
    RetryConfig,
    compose_address,
    first_location_address,
    get_with_retry,
    is_valid_npi_checksum,
    normalize_npi_digits,
    normalize_npi_status,
    primary_taxonomy,
    provider_display_name,
)
from .utils import now_iso


class NPIRegistryClient:
    """NPPES NPI Registry API v2.1 client. Public API; no key required.

    This client is intentionally conservative:
    - validates NPI check digit before network lookup
    - uses retry/backoff for transient HTTP failures
    - keeps NPI active/inactive status as explicit evidence
    - preserves address_2 / suite in composed addresses
    """

    BASE_URL = NPI_API_BASE_URL

    def __init__(
        self,
        source_reliability: Dict[str, float],
        timeout: int = 15,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
    ):
        self.source_reliability = source_reliability
        self.retry = RetryConfig(
            timeout_seconds=timeout,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
        )
        self.session = requests.Session()

    def lookup_by_npi(self, npi: str) -> List[EvidenceItem]:
        npi_digits = normalize_npi_digits(npi)
        if not re.fullmatch(r"\d{10}", npi_digits):
            return []
        if not is_valid_npi_checksum(npi_digits):
            return [
                EvidenceItem(
                    source_name="NPI Checksum",
                    source_type="npi_validation",
                    source_url=None,
                    field="npi",
                    value=npi_digits,
                    normalized_value=npi_digits,
                    evidence_text=f"NPI value {npi_digits} failed check digit validation before NPI Registry lookup.",
                    source_confidence=0.95,
                    collected_at=now_iso(),
                    metadata={"checksum_valid": False},
                )
            ]

        params = {"version": "2.1", "number": npi_digits}
        response = get_with_retry(self.BASE_URL, params=params, retry=self.retry, session=self.session)
        payload = response.json()
        results = payload.get("results", []) or []
        if not results:
            return []

        result = results[0]
        basic = result.get("basic", {}) or {}
        location = first_location_address(result)
        primary_tax = primary_taxonomy(result)
        status_raw = basic.get("status", "")
        active_status = normalize_npi_status(status_raw)

        evidence: List[EvidenceItem] = []
        trust = float(self.source_reliability.get("NPI Registry", 0.9))
        collected = now_iso()

        evidence.append(EvidenceItem(
            source_name="NPI Registry",
            source_type="npi_registry",
            source_url=self.BASE_URL,
            field="npi",
            value=npi_digits,
            normalized_value=npi_digits,
            evidence_text=f"NPI Registry returned a public record for NPI {npi_digits}; checksum is valid.",
            source_confidence=trust,
            collected_at=collected,
            metadata={"checksum_valid": True, "raw_status": status_raw, "active_status": active_status},
        ))

        evidence.append(EvidenceItem(
            source_name="NPI Registry",
            source_type="npi_registry",
            source_url=self.BASE_URL,
            field="active_status",
            value=active_status,
            normalized_value=normalize_active_status(active_status),
            evidence_text=f"NPI Registry status is {active_status} (raw status: {status_raw or 'blank'}).",
            source_confidence=trust,
            collected_at=collected,
            metadata={"raw_status": status_raw},
        ))

        name = provider_display_name(result)
        if name:
            evidence.append(EvidenceItem(
                source_name="NPI Registry",
                source_type="npi_registry",
                source_url=self.BASE_URL,
                field="provider_name",
                value=name,
                normalized_value=normalize_field_value("provider_name", name),
                evidence_text=f"NPI Registry basic provider name: {name}.",
                source_confidence=trust,
                collected_at=collected,
            ))

        if primary_tax and primary_tax.get("desc"):
            specialty = primary_tax.get("desc", "")
            evidence.append(EvidenceItem(
                source_name="NPI Registry",
                source_type="npi_registry",
                source_url=self.BASE_URL,
                field="specialty",
                value=specialty,
                normalized_value=normalize_field_value("specialty", specialty),
                evidence_text=f"NPI Registry primary taxonomy: {specialty}.",
                source_confidence=trust,
                collected_at=collected,
                metadata={
                    "taxonomy_code": primary_tax.get("code", ""),
                    "license": primary_tax.get("license", ""),
                    "state": primary_tax.get("state", ""),
                },
            ))

        if location:
            address = compose_address(location)
            if address:
                evidence.append(EvidenceItem(
                    source_name="NPI Registry",
                    source_type="npi_registry",
                    source_url=self.BASE_URL,
                    field="address",
                    value=address,
                    normalized_value=normalize_field_value("address", address),
                    evidence_text=f"NPI Registry practice location address: {address}.",
                    source_confidence=trust,
                    collected_at=collected,
                    metadata={
                        "address_1": location.get("address_1", ""),
                        "address_2": location.get("address_2", ""),
                        "city": location.get("city", ""),
                        "state": location.get("state", ""),
                        "postal_code": location.get("postal_code", ""),
                    },
                ))
            phone = normalize_phone(location.get("telephone_number", ""))
            if phone:
                evidence.append(EvidenceItem(
                    source_name="NPI Registry",
                    source_type="npi_registry",
                    source_url=self.BASE_URL,
                    field="phone",
                    value=phone,
                    normalized_value=normalize_field_value("phone", phone),
                    evidence_text=f"NPI Registry practice location phone: {phone}.",
                    source_confidence=trust,
                    collected_at=collected,
                ))
        return evidence