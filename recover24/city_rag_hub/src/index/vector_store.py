# src/index/vector_store.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.index.chunking import ChunkConfig, chunk_record

# ---------------------------
# Optional deps (faiss + sentence-transformers)
# ---------------------------
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore
    _faiss_import_error = e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _st_import_error = e


ChunkMode = Literal["none", "threshold"]


def _require_deps() -> None:
    if faiss is None:
        raise ImportError(
            "faiss is required. Install one of: faiss-cpu (recommended) or faiss-gpu.\n"
            f"Original error: {_faiss_import_error}"
        )
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required.\n"
            f"Original error: {_st_import_error}"
        )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: id, prefecture, name, explanation
    needed = {"id", "prefecture", "name", "explanation"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"parquet is missing columns: {sorted(missing)}")

    # Clean NaNs
    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["prefecture"] = out["prefecture"].fillna("").astype(str)
    out["name"] = out["name"].fillna("").astype(str)
    out["explanation"] = out["explanation"].fillna("").astype(str)
    return out


def _make_chunk_cfg(chunk_mode: ChunkMode) -> ChunkConfig:
    # 너희 분포 기반 디폴트
    if chunk_mode == "none":
        return ChunkConfig(
            mode="none",
            threshold_body_chars=1400,
            max_body_chars=1000,
            overlap_chars=100,
            min_chunk_body_chars=0,  # merge off by default
            min_total_chars=0,       # don't filter by default
        )
    return ChunkConfig(
        mode="threshold",
        threshold_body_chars=1400,  # p95
        max_body_chars=1000,
        overlap_chars=100,
        min_chunk_body_chars=0,     # start simple
        min_total_chars=0,
    )


def _default_model_name() -> str:
    # Multilingual SBERT (Japanese OK). Good default for demo.
    return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class VectorStore:
    """
    FAISS-backed vector store.

    Artifacts saved in index_dir:
      - index.faiss          (FAISS index)
      - meta.parquet         (chunk metadata aligned with FAISS vector ids)
      - config.json          (build config)
    """

    def __init__(
        self,
        index,
        meta: pd.DataFrame,
        model_name: str,
        normalize_embeddings: bool = True,
    ):
        self.index = index
        self.meta = meta.reset_index(drop=True)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self._model = None  # lazy loaded

    # -------------
    # Model
    # -------------
    def _get_model(self):
        _require_deps()
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -------------
    # Build
    # -------------
    @classmethod
    def build(
        cls,
        parquet_path: str | Path,
        index_dir: str | Path,
        *,
        chunk_mode: ChunkMode = "threshold",
        model_name: Optional[str] = None,
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        include_labels: bool = True,
    ) -> "VectorStore":
        """
        Build vector index from tourism.parquet.
        - chunk_mode: "none" or "threshold"
        - model_name: sentence-transformers model
        """
        _require_deps()

        parquet_path = Path(parquet_path)
        index_dir = Path(index_dir)
        _ensure_dir(index_dir)

        df = pd.read_parquet(parquet_path)
        df = _normalize_rows(df)

        cfg = _make_chunk_cfg(chunk_mode)
        model_name = model_name or _default_model_name()

        # 1) Make chunks table
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            doc_id = str(r["id"])
            name = r["name"]
            prefecture = r["prefecture"]
            explanation = r["explanation"]

            chunk_texts = chunk_record(
                name=name,
                prefecture=prefecture,
                explanation=explanation,
                cfg=cfg,
                include_labels=include_labels,
            )
            if not chunk_texts:
                continue

            for i, txt in enumerate(chunk_texts):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}__c{i:02d}",
                        "name": name,
                        "prefecture": prefecture,
                        "text": txt,
                    }
                )

        if not rows:
            raise RuntimeError("No chunks were generated. Check input data / chunking config.")

        meta = pd.DataFrame(rows)
        texts = meta["text"].tolist()

        # 2) Embed
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        ).astype(np.float32)

        # 3) Build FAISS index (cosine similarity via inner product on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # 4) Save artifacts
        faiss.write_index(index, str(index_dir / "index.faiss"))
        meta.to_parquet(index_dir / "meta.parquet", index=False)

        config = {
            "parquet_path": str(parquet_path),
            "chunk_mode": chunk_mode,
            "chunk_config": asdict(cfg),
            "model_name": model_name,
            "normalize_embeddings": normalize_embeddings,
            "include_labels": include_labels,
            "num_vectors": int(index.ntotal),
            "dim": int(dim),
        }
        (index_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        return cls(index=index, meta=meta, model_name=model_name, normalize_embeddings=normalize_embeddings)

    # -------------
    # Load
    # -------------
    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        _require_deps()

        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "meta.parquet"
        cfg_path = index_dir / "config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"missing: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing: {meta_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing: {cfg_path}")

        index = faiss.read_index(str(index_path))
        meta = pd.read_parquet(meta_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        return cls(
            index=index,
            meta=meta,
            model_name=cfg["model_name"],
            normalize_embeddings=bool(cfg.get("normalize_embeddings", True)),
        )

    # -------------
    # Search
    # -------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        dedup_by_doc_id: bool = True,
        max_per_doc: int = 1,
        return_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of hits with fields:
          doc_id, chunk_id, name, prefecture, score, snippet (and optionally text)
        """
        if not query or not query.strip():
            return []

        model = self._get_model()
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)

        # FAISS returns (scores, ids)
        scores, ids = self.index.search(q_emb, top_k * 5 if dedup_by_doc_id else top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()

        results: List[Dict[str, Any]] = []
        per_doc_count: Dict[str, int] = {}

        for idx, score in zip(ids, scores):
            if idx < 0:
                continue
            m = self.meta.iloc[int(idx)]
            doc_id = str(m["doc_id"])

            if dedup_by_doc_id:
                c = per_doc_count.get(doc_id, 0)
                if c >= max_per_doc:
                    continue
                per_doc_count[doc_id] = c + 1

            text = str(m["text"])
            snippet = text.replace("\n", " ")[:180]
            hit = {
                "doc_id": doc_id,
                "chunk_id": str(m["chunk_id"]),
                "name": str(m["name"]),
                "prefecture": str(m["prefecture"]),
                "score": float(score),
                "snippet": snippet,
            }
            if return_text:
                hit["text"] = text

            results.append(hit)
            if len(results) >= top_k:
                break

        return results


# ---------------------------
# Tiny smoke test helper (optional)
# ---------------------------
def _demo():
    """
    Run from project root:
      python -m src.index.vector_store
    """
    base = Path("data/processed")
    parquet = base / "tourism.parquet"

    vs_none_dir = base / "index_vector_none"
    vs_thr_dir = base / "index_vector_thr_p95"

    # build once (comment out after built)
    # VectorStore.build(parquet, vs_none_dir, chunk_mode="none")
    # VectorStore.build(parquet, vs_thr_dir, chunk_mode="threshold")

    vs = VectorStore.load(vs_thr_dir)
    hits = vs.search("雨の日 室内 観光", top_k=5)
    for h in hits:
        print(h["score"], h["prefecture"], h["name"], "=>", h["snippet"])


if __name__ == "__main__":
    _demo()
