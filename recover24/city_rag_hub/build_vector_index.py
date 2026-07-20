from src.index.vector_store import VectorStore

PARQUET = "data/processed/tourism.parquet"
VectorStore.build(PARQUET, "data/processed/index_vector_none", chunk_mode="none")
VectorStore.build(PARQUET, "data/processed/index_vector_thr_p95", chunk_mode="threshold")
print("DONE")
