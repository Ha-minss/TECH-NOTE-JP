from __future__ import annotations

import re
from datetime import datetime
from rapidfuzz import fuzz


CREDENTIAL_SUFFIXES = {
    "md", "m d", "do", "d o", "dr", "doctor", "phd", "ph d", "dds", "dmd", "dpm",
    "np", "fnp", "fnp c", "aprn", "pa", "pa c", "rn", "crna", "cnm", "cns",
    "lcsw", "lmft", "psyd", "od", "facp", "facc", "facs", "faap",
}

SPECIALTY_SYNONYMS = {
    "cardiovascular disease": "cardiology",
    "cardiovascular disease cardiology": "cardiology",
    "cardiologist": "cardiology",
    "pediatric medicine": "pediatrics",
    "family practice": "family medicine",
    "general practice": "family medicine",
    "ob gyn": "obgyn",
    "obstetrics gynecology": "obgyn",
    "certified registered nurse anesthetist crna": "crna",
    "certified registered nurse anesthetist": "crna",
    "nurse practitioner": "nurse practitioner",
    "physician assistant": "physician assistant",
}


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_text(text: str) -> str:
    text = safe_str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phone(phone: str) -> str:
    """Normalize US phone numbers.

    Handles common CMS/website formats such as:
    - (239) 555-1234
    - +1 239 555 1234
    - 9087694700x2025
    - 4256737293 ext 7010

    We keep only the base 10-digit NANP phone number. Extensions are deliberately
    excluded from field matching and can be kept in metadata by a caller if needed.
    """
    raw = safe_str(phone).lower()
    # Prefer the first plausible phone-like sequence before an extension marker.
    raw = re.split(r"\b(?:x|ext|extension)\b\.?\s*", raw, maxsplit=1)[0]
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) > 10:
        # Useful for values like 9087694700x2025 where extension text was not split.
        digits = digits[:10]
    return digits


def display_phone(phone: str) -> str:
    digits = normalize_phone(phone)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return safe_str(phone)


def normalize_npi(npi: str) -> str:
    digits = re.sub(r"\D", "", safe_str(npi))
    return digits if len(digits) == 10 else digits


def is_valid_npi(npi: str) -> bool:
    """Format-only validation. Use src.npi.is_valid_npi_checksum for check digit validation."""
    return bool(re.fullmatch(r"\d{10}", normalize_npi(npi)))


def normalize_active_status(status: str) -> str:
    s = normalize_text(status)
    if s in {"a", "active", "valid", "enabled"}:
        return "active"
    if s in {"i", "inactive", "deactivated", "disabled", "retired", "no longer practicing", "revoked"}:
        return "inactive" if s != "revoked" else "revoked"
    if any(term in s for term in ["revoked", "revocation"]):
        return "revoked"
    if any(term in s for term in ["retired", "inactive", "deactivated", "no longer", "left the practice", "permanently closed"]):
        return "inactive"
    if not s:
        return "unknown"
    return s


def normalize_address(address: str) -> str:
    s = safe_str(address).lower()
    replacements = {
        "street": "st",
        "st.": "st",
        "avenue": "ave",
        "ave.": "ave",
        "road": "rd",
        "rd.": "rd",
        "drive": "dr",
        "dr.": "dr",
        "boulevard": "blvd",
        "blvd.": "blvd",
        "parkway": "pkwy",
        "pkwy.": "pkwy",
        "lane": "ln",
        "ln.": "ln",
        "court": "ct",
        "ct.": "ct",
        "suite": "ste",
        "ste.": "ste",
        "apartment": "apt",
        "floor": "fl",
        "north": "n",
        "south": "s",
        "east": "e",
        "west": "w",
    }
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [replacements.get(tok, tok) for tok in s.split()]
    # ZIP+4 and 5-digit ZIP should compare well; keep first 5 where a 9-digit ZIP leaked.
    out = " ".join(tokens)
    out = re.sub(r"\b(\d{5})\d{4}\b", r"\1", out)
    return out


def _strip_credentials_from_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if tok in CREDENTIAL_SUFFIXES:
            continue
        out.append(tok)
    # Remove trailing credential phrases after punctuation normalization.
    while out and out[-1] in CREDENTIAL_SUFFIXES:
        out.pop()
    return out


def normalize_provider_name(name: str) -> str:
    s = normalize_text(name)
    if not s:
        return ""
    # Convert M D / D O style tokens into credentials before filtering.
    s = re.sub(r"\bm\s*d\b", " md ", s)
    s = re.sub(r"\bd\s*o\b", " do ", s)
    s = re.sub(r"\bp\s*h\s*d\b", " phd ", s)
    s = re.sub(r"\bpa\s*c\b", " pa c ", s)
    tokens = _strip_credentials_from_tokens(s.split())
    # Drop common generational suffixes for fuzzy identity matching.
    tokens = [t for t in tokens if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def _name_last_token(name: str) -> str:
    tokens = normalize_provider_name(name).split()
    return tokens[-1] if tokens else ""


def provider_names_equivalent(left: str, right: str) -> bool:
    a = normalize_provider_name(left)
    b = normalize_provider_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # Last name should match for person identity. For org names this is only a
    # helper and exact/fuzzy match is still required.
    last_a = _name_last_token(left)
    last_b = _name_last_token(right)
    score = fuzz.token_set_ratio(a, b)
    if last_a and last_b and last_a == last_b and score >= 86:
        return True
    # Middle-name vs middle-initial case: ANARA KAYSER ABBAY vs ANARA K ABBAY.
    ta, tb = a.split(), b.split()
    if len(ta) >= 2 and len(tb) >= 2 and ta[0] == tb[0] and ta[-1] == tb[-1]:
        return score >= 80
    return score >= 93


def normalize_specialty(specialty: str) -> str:
    s = normalize_text(specialty)
    if not s:
        return ""
    # CMS PPEF descriptions are role-prefixed. Keep the clinical/supplier part.
    prefixes = [
        "practitioner",
        "part a provider",
        "part b supplier",
        "dme supplier physician",
        "dme supplier",
        "order and referring only",
        "mdpp supplier",
    ]
    for prefix in prefixes:
        if s.startswith(prefix + " "):
            s = s[len(prefix):].strip()
            break
    s = re.sub(r"\bformerly\b.*$", "", s).strip()
    s = s.replace(" and ", " ")
    s = re.sub(r"\((.*?)\)", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return SPECIALTY_SYNONYMS.get(s, s)


def specialty_tokens(specialty: str) -> set[str]:
    # Many provider records store specialties as comma/slash separated values.
    raw = safe_str(specialty)
    parts = re.split(r"[,;/|]+", raw)
    normalized = {normalize_specialty(p) for p in parts if normalize_specialty(p)}
    if not normalized and raw:
        normalized.add(normalize_specialty(raw))
    # Add synonym-collapsed version for full field too.
    full = normalize_specialty(raw)
    if full:
        normalized.add(full)
    return normalized


def specialties_equivalent(left: str, right: str) -> bool:
    a_set = specialty_tokens(left)
    b_set = specialty_tokens(right)
    if not a_set or not b_set:
        return False
    if a_set & b_set:
        return True
    for a in a_set:
        for b in b_set:
            if fuzz.token_set_ratio(a, b) >= 88:
                return True
            # Containment catches "Internal Medicine, Interventional Cardiology"
            # vs "Interventional Cardiology" after tokenization misses a variant.
            if a in b or b in a:
                return True
    return False


def values_equal(field: str, left: str, right: str) -> bool:
    if field == "phone":
        return normalize_phone(left) == normalize_phone(right)
    if field == "address":
        return fuzz.token_set_ratio(normalize_address(left), normalize_address(right)) >= 88
    if field == "provider_name":
        return provider_names_equivalent(left, right)
    if field == "practice_name":
        return fuzz.token_set_ratio(normalize_text(left), normalize_text(right)) >= 90
    if field == "specialty":
        return specialties_equivalent(left, right)
    if field == "active_status":
        return normalize_active_status(left) == normalize_active_status(right)
    return normalize_text(left) == normalize_text(right)


def normalize_field_value(field: str, value: str) -> str:
    if field == "phone":
        return normalize_phone(value)
    if field == "address":
        return normalize_address(value)
    if field == "provider_name":
        return normalize_provider_name(value)
    if field == "specialty":
        return normalize_specialty(value)
    if field == "npi":
        return normalize_npi(value)
    if field == "active_status":
        return normalize_active_status(value)
    return normalize_text(value)


def days_since(date_str: str) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (datetime.utcnow() - dt).days
    except Exception:
        return None
