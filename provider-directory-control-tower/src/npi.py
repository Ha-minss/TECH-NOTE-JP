from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .normalize import display_phone, normalize_phone, safe_str

NPI_API_BASE_URL = "https://npiregistry.cms.hhs.gov/api/"


class NPIRegistryError(RuntimeError):
    """Raised when NPI Registry returns an unrecoverable response."""


def normalize_npi_digits(npi: str) -> str:
    return re.sub(r"\D", "", safe_str(npi))


def is_valid_npi_format(npi: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", normalize_npi_digits(npi)))


def is_valid_npi_checksum(npi: str) -> bool:
    """Validate an NPI using the official Luhn-style check digit rule.

    NPI is validated by prefixing the 10-digit NPI with the ISO issuer prefix
    80840 and running the Luhn algorithm over the resulting 15 digits.

    Examples known to validate: 1234567893, 1003000126.
    """
    digits = normalize_npi_digits(npi)
    if not re.fullmatch(r"\d{10}", digits):
        return False
    payload = "80840" + digits
    total = 0
    double = False
    for ch in reversed(payload):
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return total % 10 == 0


def npi_validation_detail(npi: str) -> Dict[str, Any]:
    digits = normalize_npi_digits(npi)
    return {
        "npi": digits,
        "format_valid": is_valid_npi_format(digits),
        "checksum_valid": is_valid_npi_checksum(digits),
    }


def normalize_npi_status(status: Any) -> str:
    """Map NPPES status variants to the internal active/inactive/unknown vocabulary."""
    s = safe_str(status).strip().lower()
    if s in {"a", "active", "enabled", "valid"}:
        return "active"
    if s in {"i", "inactive", "deactivated", "disabled", "retired"}:
        return "inactive"
    if not s:
        return "unknown"
    # Keep unknown statuses conservative.
    return "unknown"


def first_location_address(result: Dict[str, Any]) -> Dict[str, Any]:
    addresses = result.get("addresses", []) or []
    for addr in addresses:
        if safe_str(addr.get("address_purpose")).upper() == "LOCATION":
            return addr
    return addresses[0] if addresses else {}


def mailing_address(result: Dict[str, Any]) -> Dict[str, Any]:
    addresses = result.get("addresses", []) or []
    for addr in addresses:
        if safe_str(addr.get("address_purpose")).upper() == "MAILING":
            return addr
    return {}


def primary_taxonomy(result: Dict[str, Any]) -> Dict[str, Any]:
    taxes = result.get("taxonomies", []) or []
    for tax in taxes:
        if bool(tax.get("primary")):
            return tax
    return taxes[0] if taxes else {}


def provider_display_name(result: Dict[str, Any]) -> str:
    basic = result.get("basic", {}) or {}
    name_parts: List[str] = []
    if basic.get("first_name") or basic.get("last_name"):
        name_parts = [
            basic.get("first_name", ""),
            basic.get("middle_name", ""),
            basic.get("last_name", ""),
            basic.get("credential", ""),
        ]
    elif basic.get("organization_name"):
        name_parts = [basic.get("organization_name", "")]
    return " ".join(p for p in name_parts if p).strip()


def compose_address(addr: Dict[str, Any]) -> str:
    """Compose a human-readable address without losing address_2 / suite."""
    parts = [
        addr.get("address_1", ""),
        addr.get("address_2", ""),
        addr.get("city", ""),
        addr.get("state", ""),
        addr.get("postal_code", ""),
    ]
    return ", ".join(str(p).strip() for p in parts if str(p or "").strip())


def extract_all_useful_npi_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten high-value NPI fields while preserving the raw JSON elsewhere.

    This is not meant to replace raw storage. It is a compact summary that makes
    later debugging, analysis, and Kaggle writeups easier.
    """
    basic = result.get("basic", {}) or {}
    loc = first_location_address(result)
    mail = mailing_address(result)
    tax = primary_taxonomy(result)
    npi = normalize_npi_digits(str(result.get("number", "")))
    return {
        "npi": npi,
        "npi_checksum_valid": is_valid_npi_checksum(npi),
        "enumeration_type": result.get("enumeration_type", ""),
        "provider_name": provider_display_name(result),
        "first_name": basic.get("first_name", ""),
        "middle_name": basic.get("middle_name", ""),
        "last_name": basic.get("last_name", ""),
        "credential": basic.get("credential", ""),
        "organization_name": basic.get("organization_name", ""),
        "sole_proprietor": basic.get("sole_proprietor", ""),
        "sex": basic.get("sex", ""),
        "npi_raw_status": basic.get("status", ""),
        "active_status": normalize_npi_status(basic.get("status", "")),
        "enumeration_date": basic.get("enumeration_date", ""),
        "last_updated": basic.get("last_updated", ""),
        "certification_date": basic.get("certification_date", ""),
        "location_address_1": loc.get("address_1", ""),
        "location_address_2": loc.get("address_2", ""),
        "location_city": loc.get("city", ""),
        "location_state": loc.get("state", ""),
        "location_postal_code": loc.get("postal_code", ""),
        "location_country_code": loc.get("country_code", ""),
        "location_phone": display_phone(loc.get("telephone_number", "")),
        "location_fax": display_phone(loc.get("fax_number", "")),
        "mailing_address": compose_address(mail),
        "mailing_phone": display_phone(mail.get("telephone_number", "")),
        "primary_taxonomy_code": tax.get("code", ""),
        "primary_taxonomy_desc": tax.get("desc", ""),
        "primary_taxonomy_state": tax.get("state", ""),
        "primary_taxonomy_license": tax.get("license", ""),
        "taxonomy_count": len(result.get("taxonomies", []) or []),
        "identifier_count": len(result.get("identifiers", []) or []),
        "endpoint_count": len(result.get("endpoints", []) or []),
        "practice_location_count": len(result.get("practiceLocations", []) or []),
    }


def to_provider_record(result: Dict[str, Any], provider_id: str) -> Dict[str, Any]:
    summary = extract_all_useful_npi_fields(result)
    loc = first_location_address(result)
    tax = primary_taxonomy(result)
    basic = result.get("basic", {}) or {}
    address = compose_address(loc)
    return {
        "provider_id": provider_id,
        "provider_name": summary["provider_name"],
        "npi": summary["npi"],
        "specialty": tax.get("desc", ""),
        # For Type 1 individuals this is usually empty. That is OK; websites / other
        # sources should later fill affiliation/practice details.
        "practice_name": basic.get("organization_name", "") or "",
        "address": address,
        "phone": display_phone(loc.get("telephone_number", "")),
        "website": "",
        "last_verified_date": basic.get("last_updated", "") or basic.get("certification_date", "") or basic.get("enumeration_date", ""),
        "active_status": summary["active_status"],
        # Extra debug/analysis fields. ProviderRecord will ignore these during pipeline
        # execution, but the processed JSONL keeps them for reviewers and benchmark work.
        "npi_checksum_valid": summary["npi_checksum_valid"],
        "npi_raw_status": summary["npi_raw_status"],
        "npi_enumeration_type": summary["enumeration_type"],
        "address_1": loc.get("address_1", ""),
        "address_2": loc.get("address_2", ""),
        "city": loc.get("city", ""),
        "state": loc.get("state", ""),
        "postal_code": loc.get("postal_code", ""),
        "taxonomy_code": tax.get("code", ""),
        "taxonomy_license": tax.get("license", ""),
    }


@dataclass
class RetryConfig:
    max_retries: int = 4
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 20.0
    timeout_seconds: int = 30
    sleep_between_requests: float = 0.10


def get_with_retry(
    url: str,
    params: Dict[str, Any],
    retry: RetryConfig,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    sess = session or requests.Session()
    last_error: Optional[BaseException] = None
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(retry.max_retries + 1):
        try:
            response = sess.get(url, params=params, timeout=retry.timeout_seconds)
            if response.status_code in retryable_statuses:
                raise NPIRegistryError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            return response
        except Exception as exc:  # requests exceptions + retryable status wrapper
            last_error = exc
            if attempt >= retry.max_retries:
                break
            delay = min(retry.backoff_max_seconds, retry.backoff_base_seconds * (2 ** attempt))
            delay = delay * (0.75 + random.random() * 0.5)  # jitter
            time.sleep(delay)
    raise NPIRegistryError(f"NPI request failed after retries: {last_error}")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
