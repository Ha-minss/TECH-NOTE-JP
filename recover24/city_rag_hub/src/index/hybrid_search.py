# src/index/hybrid_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.index.bm25_store import BM25Store, BM25SearchResult
from src.index.vector_store import VectorStore


@dataclass
class HybridHit:
    doc_id: str
    score: float
    bm25_score: float
    vec_score: float
    name: str
    prefecture: str
    snippet: str
    source: str  # "bm25" | "vec" | "both"


def _normalize_doc_id(doc_id: str) -> str:
    """
    Align doc_id between BM25 ("tour_123") and Vector ("123").
    """
    s = str(doc_id)
    if s.startswith("tour_"):
        return s.replace("tour_", "", 1)
    return s


def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    a = np.asarray(scores, dtype=np.float32)
    mn = float(a.min())
    mx = float(a.max())
    if mx - mn < 1e-12:
        return [0.0 for _ in scores]
    return [float((x - mn) / (mx - mn)) for x in scores]


class HybridSearch:
    def __init__(self, bm25: BM25Store, vec: VectorStore):
        self.bm25 = bm25
        self.vec = vec

    @classmethod
    def load(
        cls,
        *,
        bm25_index_dir: str = "data/processed/index_bm25",
        vec_index_dir: str = "data/processed/index_vector_thr_p95",
    ) -> "HybridSearch":
        bm25 = BM25Store.load(bm25_index_dir)
        vec = VectorStore.load(vec_index_dir)
        return cls(bm25=bm25, vec=vec)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        bm25_k: int = 30,
        vec_k: int = 30,
        w_bm25: float = 0.6,
        w_vec: float = 0.4,
        prefecture_filter: str | None = None,   # ✅ 추가
    ) -> List[HybridHit]:
        """
        Simple hybrid:
          - get bm25_k and vec_k
          - normalize each score to 0~1 (min-max within each list)
          - final = w_bm25 * bm25_norm + w_vec * vec_norm
          - merge by normalized doc_id
        """
        bm25_res = self.bm25.search(query, topk=bm25_k)
        vec_res = self.vec.search(query, top_k=vec_k, dedup_by_doc_id=True, max_per_doc=1)

        # normalize scores (separately)
        bm25_norm = _minmax_norm([r.score for r in bm25_res])
        vec_norm = _minmax_norm([r["score"] for r in vec_res])

        merged: Dict[str, Dict[str, Any]] = {}

        # add bm25
        for r, nscore in zip(bm25_res, bm25_norm):
            doc_id = _normalize_doc_id(r.doc_id)
            if doc_id not in merged:
                merged[doc_id] = {
                    "doc_id": doc_id,
                    "bm25_score": 0.0,
                    "vec_score": 0.0,
                    "name": r.title_ja,
                    "prefecture": r.prefecture,
                    "snippet": r.snippet,
                    "has_bm25": False,
                    "has_vec": False,
                }
            merged[doc_id]["bm25_score"] = float(nscore)
            merged[doc_id]["has_bm25"] = True
            # bm25는 title/snippet가 더 믿을만할 때가 많아서 유지

        # add vector
        for r, nscore in zip(vec_res, vec_norm):
            doc_id = _normalize_doc_id(r["doc_id"])
            if doc_id not in merged:
                merged[doc_id] = {
                    "doc_id": doc_id,
                    "bm25_score": 0.0,
                    "vec_score": 0.0,
                    "name": r["name"],
                    "prefecture": r["prefecture"],
                    "snippet": r["snippet"],
                    "has_bm25": False,
                    "has_vec": False,
                }
            merged[doc_id]["vec_score"] = float(nscore)
            merged[doc_id]["has_vec"] = True
            # 이름/지역이 비어있거나 이상하면 vec쪽으로 덮어씌우지 말자
            if merged[doc_id].get("name", "") in ("", None):
                merged[doc_id]["name"] = r["name"]
            if merged[doc_id].get("prefecture", "") in ("", None):
                merged[doc_id]["prefecture"] = r["prefecture"]

        hits: List[HybridHit] = []
        for doc_id, m in merged.items():
            final = w_bm25 * float(m["bm25_score"]) + w_vec * float(m["vec_score"])
            if m["has_bm25"] and m["has_vec"]:
                source = "both"
            elif m["has_bm25"]:
                source = "bm25"
            else:
                source = "vec"

            hits.append(
                HybridHit(
                    doc_id=doc_id,
                    score=float(final),
                    bm25_score=float(m["bm25_score"]),
                    vec_score=float(m["vec_score"]),
                    name=str(m.get("name", "")),
                    prefecture=str(m.get("prefecture", "")),
                    snippet=str(m.get("snippet", "")),
                    source=source,
                )
            )

        if prefecture_filter:
            pf = prefecture_filter.strip()
            hits = [h for h in hits if (h.prefecture or "").strip() == pf]

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]
