# src/utils/evidence.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class EvidenceCheck:
    ok: bool
    label: str                # "OK" | "INSUFFICIENT_EVIDENCE"
    matched_terms: List[str]  # query terms that appeared in evidence
    coverage: float           # matched_terms / required_terms
    details: str              # short debug message


_JA_TOKEN_RE = re.compile(r"[一-龥々〆ヵヶぁ-ゖァ-ヺーa-zA-Z0-9]+")


def _simple_ja_terms(text: str) -> List[str]:
    """
    Very simple Japanese term extraction without external deps.
    - Keeps CJK/Hiragana/Katakana/Latin/num chunks.
    - Drops 1-char tokens to reduce noise.
    """
    toks = _JA_TOKEN_RE.findall(text or "")
    toks = [t.strip() for t in toks if len(t.strip()) >= 2]
    return toks


def insufficient_evidence_check(
    query: str,
    snippets: Sequence[str],
    *,
    # If provided, we ONLY check these terms (recommended for controlled tests)
    required_terms: List[str] | None = None,
    # Otherwise, we auto-pick top-N query terms as "required"
    auto_top_n: int = 4,
    # Need at least this fraction of required terms to appear in snippets
    min_coverage: float = 0.34,
    # Also require at least this many unique matched terms
    min_matched: int = 1,
    # Search only within first N chars of each snippet (speed/noise control)
    snippet_window: int = 600,
) -> EvidenceCheck:
    """
    Checks whether top retrieved snippets contain enough of the query's key terms.
    This is NOT a keyword filter for retrieval; it's a post-check to prevent nonsense answers.
    """
    q = (query or "").strip()
    if not q:
        return EvidenceCheck(
            ok=False,
            label="INSUFFICIENT_EVIDENCE",
            matched_terms=[],
            coverage=0.0,
            details="empty query",
        )

    # Determine required terms
    if required_terms is None:
        terms = _simple_ja_terms(q)
        # Dedup while preserving order
        seen = set()
        terms = [t for t in terms if not (t in seen or seen.add(t))]
        required = terms[:auto_top_n]
    else:
        required = [t.strip() for t in required_terms if t.strip()]

    if not required:
        # If we can't extract anything meaningful, don't block the answer
        return EvidenceCheck(
            ok=True,
            label="OK",
            matched_terms=[],
            coverage=1.0,
            details="no required terms (skip check)",
        )

    hay = "\n".join([(s or "")[:snippet_window] for s in snippets])
    matched = [t for t in required if t in hay]
    matched_unique = list(dict.fromkeys(matched))
    coverage = len(matched_unique) / max(1, len(required))

    ok = (coverage >= min_coverage) and (len(matched_unique) >= min_matched)
    label = "OK" if ok else "INSUFFICIENT_EVIDENCE"
    details = f"required={required} matched={matched_unique}"

    return EvidenceCheck(
        ok=ok,
        label=label,
        matched_terms=matched_unique,
        coverage=float(coverage),
        details=details,
    )
