import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from src.index.bm25_store import BM25Store
from src.index.vector_store import VectorStore
from src.index.hybrid_search import HybridSearch, _normalize_doc_id  # ✅ class명/정규화 함수

QUERIES_PATH = "data/processed/eval/queries.jsonl"
GOLD_PATH    = "data/processed/eval/gold.jsonl"

BM25_DIR   = "data/processed/index_bm25"
VEC_DIR    = "data/processed/index_vector_thr_p95"  # ✅ 너희 vector 기본 dir (필요시 수정)

TOPK_LIST = [1, 3, 5, 10]
OUT_PATH  = "data/processed/eval/results_eval50.json"


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_gold_map(gold_rows: List[dict]) -> Dict[str, Set[str]]:
    """
    gold.jsonl row:
      {"qid":"1_name","relevant_ids":["123"]} 또는 {"qid":"1_name","relevant_id":"123"}
    """
    m: Dict[str, Set[str]] = defaultdict(set)
    for r in gold_rows:
        qid = str(r["qid"])
        if "relevant_ids" in r:
            for _id in r["relevant_ids"]:
                m[qid].add(str(_id))
        elif "relevant_id" in r:
            m[qid].add(str(r["relevant_id"]))
        else:
            raise ValueError(f"gold row missing relevant_id(s): {r}")
    return m


def hit_at_k(pred_ids: List[str], gold_ids: Set[str], k: int) -> int:
    return 1 if set(pred_ids[:k]) & gold_ids else 0


def eval_system(name: str, search_fn, queries: List[dict], gold: Dict[str, Set[str]]) -> dict:
    totals = {k: 0 for k in TOPK_LIST}
    hits   = {k: 0 for k in TOPK_LIST}
    per_q = []

    for q in queries:
        qid = q["qid"]
        text = q["query"]
        gold_ids = gold[qid]

        pred_ids = search_fn(text, top_k=max(TOPK_LIST))  # ✅ pred_ids: List[str]
        row = {"qid": qid, "gold": sorted(list(gold_ids)), "pred": pred_ids[:10]}

        for k in TOPK_LIST:
            totals[k] += 1
            h = hit_at_k(pred_ids, gold_ids, k)
            hits[k] += h
            row[f"hit@{k}"] = h

        per_q.append(row)

    metrics = {f"hit@{k}": hits[k] / max(1, totals[k]) for k in TOPK_LIST}
    return {"system": name, "metrics": metrics, "per_query": per_q}


def main():
    queries = read_jsonl(QUERIES_PATH)
    gold_rows = read_jsonl(GOLD_PATH)
    gold = build_gold_map(gold_rows)

    # --- load ---
    bm25 = BM25Store.load(BM25_DIR)
    vec  = VectorStore.load(VEC_DIR)
    hybrid = HybridSearch(bm25=bm25, vec=vec)

    # --- wrappers: "List[str]"로 통일 ---
    def bm25_search(q: str, *, top_k: int) -> List[str]:
        res = bm25.search(q, topk=top_k)
        # BM25 doc_id가 "tour_123"일 수 있으니 normalize해서 gold("123")와 맞춤
        return [_normalize_doc_id(r.doc_id) for r in res]

    def vec_search(q: str, *, top_k: int) -> List[str]:
        res = vec.search(q, top_k=top_k, dedup_by_doc_id=True, max_per_doc=1)
        return [str(r["doc_id"]) for r in res]

    def hybrid_search(q: str, *, top_k: int) -> List[str]:
        res = hybrid.search(q, top_k=top_k)
        # HybridHit.doc_id는 이미 normalize된 형태(코드상)지만 안전하게 한 번 더
        return [_normalize_doc_id(h.doc_id) for h in res]

    results = []
    results.append(eval_system("bm25", bm25_search, queries, gold))
    results.append(eval_system("vector", vec_search, queries, gold))
    results.append(eval_system("hybrid", hybrid_search, queries, gold))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote -> {OUT_PATH}\n")
    for r in results:
        print(f"== {r['system']} ==")
        for k, v in r["metrics"].items():
            print(f"{k}: {v:.3f}")
        print("")


if __name__ == "__main__":
    main()
