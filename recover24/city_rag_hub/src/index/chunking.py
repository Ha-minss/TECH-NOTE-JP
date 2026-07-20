# src/index/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


ChunkMode = Literal["none", "threshold"]


@dataclass(frozen=True)
class ChunkConfig:
    mode: ChunkMode = "threshold"

    # Threshold decision is based on BODY (explanation) length
    threshold_body_chars: int = 1400  # p95 (your data)
    max_body_chars: int = 1000        # target chunk size for body
    overlap_chars: int = 100          # ~10%

    # Optional: merge chunks shorter than this (0 disables)
    min_chunk_body_chars: int = 100

    # Optional: filter records with too-short total text (0 disables)
    min_total_chars: int = 0


def _make_header(name: str, prefecture: str) -> str:
    n = (name or "").strip()
    p = (prefecture or "").strip()
    lines = []
    if n:
        lines.append(f"Name: {n}")
    if p:
        lines.append(f"Prefecture: {p}")
    return "\n".join(lines).strip()


def _split_paragraphs(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if "\n\n" in t:
        parts = [p.strip() for p in t.split("\n\n")]
    else:
        parts = [p.strip() for p in t.split("\n")]
    return [p for p in parts if p]


def _slice_with_overlap(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be < max_chars")

    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    chunks: List[str] = []
    start = 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(t):
            break
        start = end - overlap_chars
    return chunks


def _pack_paragraphs(paras: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """
    Paragraph-first packing.
    If a paragraph is longer than max_chars, it will be sliced with overlap.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        merged = "\n".join(buf).strip()
        if merged:
            chunks.append(merged)
        buf = []
        buf_len = 0

    for p in paras:
        # oversize paragraph: slice it directly
        if len(p) > max_chars:
            flush()
            chunks.extend(_slice_with_overlap(p, max_chars, overlap_chars))
            continue

        add_len = len(p) + (1 if buf else 0)  # newline cost (approx)
        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
        else:
            flush()
            buf.append(p)
            buf_len = len(p)

    flush()
    return [c for c in chunks if c.strip()]


def _merge_short_chunks(chunks: List[str], min_len: int) -> List[str]:
    """Merge chunks shorter than min_len into neighbors. If min_len <= 0, do nothing."""
    if min_len <= 0 or len(chunks) < 2:
        return chunks

    merged: List[str] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if len(cur) < min_len:
            if merged:
                merged[-1] = (merged[-1].rstrip() + "\n" + cur.lstrip()).strip()
            elif i + 1 < len(chunks):
                nxt = chunks[i + 1]
                merged.append((cur.rstrip() + "\n" + nxt.lstrip()).strip())
                i += 1
            else:
                merged.append(cur)
        else:
            merged.append(cur)
        i += 1
    return [c for c in merged if c.strip()]


def chunk_record(
    *,
    name: str,
    prefecture: str,
    explanation: str,
    cfg: ChunkConfig,
    include_labels: bool = True,
) -> List[str]:
    """
    Returns final chunk texts ready for embedding.

    - Header is always included (name/prefecture).
    - Chunking decision and slicing are based on BODY (explanation) length.
    - Optional filtering based on TOTAL length (header+body).
    """
    header = _make_header(name, prefecture)
    body = (explanation or "").strip()

    # skip empty body (header-only embeddings are often noisy)
    if not body:
        return []

    def compose(body_part: str) -> str:
        if include_labels:
            core = f"Description: {body_part}"
        else:
            core = body_part
        if header:
            return f"{header}\n{core}".strip()
        return core.strip()

    full_text = compose(body)
    if cfg.min_total_chars > 0 and len(full_text) < cfg.min_total_chars:
        return []

    if cfg.mode == "none":
        return [full_text]

    # threshold mode
    if len(body) <= cfg.threshold_body_chars:
        return [full_text]

    # chunk long body
    paras = _split_paragraphs(body)
    if paras:
        body_chunks = _pack_paragraphs(paras, cfg.max_body_chars, cfg.overlap_chars)
    else:
        body_chunks = _slice_with_overlap(body, cfg.max_body_chars, cfg.overlap_chars)

    body_chunks = _merge_short_chunks(body_chunks, cfg.min_chunk_body_chars)
    return [compose(c) for c in body_chunks if c.strip()]
