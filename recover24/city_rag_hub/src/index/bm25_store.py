# TODO
# src/index/bm25_store.py
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils.text import tokenize_ja_for_bm25

@dataclass
class BM25SearchResult:
    doc_id: str
    score: float
    title_ja: str
    prefecture: str
    snippet: str

class BM25Store:
    def __init__(self, bm25: BM25Okapi, meta: List[Dict[str, Any]]):
        self.bm25 = bm25
        self.meta = meta  # same length as corpus

    @staticmethod
    def build_from_parquet(parquet_path: str) -> "BM25Store":
        df = pd.read_parquet(parquet_path)

        required = ["id", "prefecture", "name", "explanation"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in parquet: {missing}")

        corpus_tokens: List[List[str]] = []
        meta: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            title = str(row.get("name", "") or "")
            body = str(row.get("explanation", "") or "")

            content = (title + "\n" + body).strip()
            tokens = tokenize_ja_for_bm25(content)

            if not tokens:
                continue

            raw_id = row.get("id", "")
            try:
                doc_id = f"tour_{int(raw_id)}" if pd.notna(raw_id) and str(raw_id).strip() != "" else ""
            except Exception:
                doc_id = f"tour_{str(raw_id)}"

            corpus_tokens.append(tokens)
            meta.append({
                "doc_id": doc_id,
                "prefecture": str(row.get("prefecture", "")),
                "title_ja": title,
                "body_ja": body,
            })

        bm25 = BM25Okapi(corpus_tokens)
        return BM25Store(bm25=bm25, meta=meta)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def load(index_dir: str) -> "BM25Store":
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)

        meta: List[Dict[str, Any]] = []
        with open(os.path.join(index_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))

        return BM25Store(bm25=bm25, meta=meta)

    def search(self, query: str, topk: int = 5) -> List[BM25SearchResult]:
        q_tokens = tokenize_ja_for_bm25(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)  # numpy array
        # Get topk indices
        topk = max(1, int(topk))
        idxs = scores.argsort()[::-1][:topk]

        results: List[BM25SearchResult] = []
        for i in idxs:
            m = self.meta[int(i)]
            body = m.get("body_ja", "")
            snippet = (body[:300] + "…") if len(body) > 300 else body

            results.append(BM25SearchResult(
                doc_id=m.get("doc_id", ""),
                score=float(scores[int(i)]),
                title_ja=m.get("title_ja", ""),
                prefecture=m.get("prefecture", ""),
                snippet=snippet,
            ))
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/processed/tourism.parquet")
    parser.add_argument("--index_dir", default="data/processed/index_bm25")
    parser.add_argument("--build", action="store_true", help="Build BM25 index")
    parser.add_argument("--query", type=str, default=None, help="Query string")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        store = BM25Store.build_from_parquet(args.parquet)
        store.save(args.index_dir)
        print(f"[OK] Built & saved BM25 index -> {args.index_dir}")
        return

    # query mode
    if not args.query:
        raise SystemExit("Provide --build or --query '<text>'")

    store = BM25Store.load(args.index_dir)
    results = store.search(args.query, args.topk)

    print(f"query: {args.query}")
    for r in results:
        print("-" * 60)
        print(f"doc_id: {r.doc_id}")
        print(f"score: {r.score:.4f}")
        print(f"prefecture: {r.prefecture}")
        print(f"title_ja: {r.title_ja}")
        print(f"snippet: {r.snippet[:200]}")

if __name__ == "__main__":
    main()  