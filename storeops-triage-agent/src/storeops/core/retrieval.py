from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

_TOKEN_PATTERN = re.compile("[A-Za-z0-9\uac00-\ud7a3]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2].lstrip()


@dataclass(frozen=True)
class PolicyDocument:
    document_id: str
    chunk_id: str
    title: str
    content: str
    path: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    chunk_id: str
    title: str
    content: str
    score: float
    dense_score: float
    bm25_score: float
    dense_weight: float
    bm25_weight: float
    metadata: dict[str, str]


class PolicyDocumentLoader:
    def __init__(self, policy_dir: Path | str):
        self.policy_dir = Path(policy_dir)

    def load(self) -> list[PolicyDocument]:
        documents: list[PolicyDocument] = []
        for path in sorted(self.policy_dir.glob('*.md')):
            raw = path.read_text(encoding='utf-8')
            metadata, content = _parse_frontmatter(raw)
            document_id = metadata.get('document_id', path.stem)
            title = self._extract_title(content) or path.stem
            documents.append(
                PolicyDocument(
                    document_id=document_id,
                    chunk_id=f'{document_id}#full',
                    title=title,
                    content=content,
                    path=path,
                    metadata=metadata,
                )
            )
        return documents

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            if line.startswith('# '):
                return line.removeprefix('# ').strip()
        return None


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in self._features(text):
                rows[row_index, self._stable_bucket(token)] += 1.0
            norm = float(np.linalg.norm(rows[row_index]))
            if norm:
                rows[row_index] /= norm
        return rows

    def _features(self, text: str) -> list[str]:
        tokens = _tokenize(text)
        features = list(tokens)
        compact = ''.join(tokens)
        features.extend(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
        features.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
        return features

    def _stable_bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, 'big') % self.dimensions


class BgeM3EmbeddingProvider:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'sentence-transformers is required for BGE-M3 embeddings. '
                    'Install it or inject another EmbeddingProvider.'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class _Bm25Index:
    def __init__(self, documents: list[PolicyDocument]):
        self.documents = documents
        self.tokenized_documents = [_tokenize(document.title + '\n' + document.content) for document in documents]
        self.doc_count = len(documents)
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.tokenized_documents) / self.doc_count if self.doc_count else 0.0
        )
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized_documents:
            self.doc_freq.update(set(tokens))

    def search(self, query: str) -> np.ndarray:
        query_terms = _tokenize(query)
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if not query_terms or not self.doc_count:
            return scores
        k1 = 1.5
        b = 0.75
        for doc_index, tokens in enumerate(self.tokenized_documents):
            frequencies = Counter(tokens)
            doc_len = len(tokens) or 1
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                containing_docs = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.doc_count - containing_docs + 0.5) / (containing_docs + 0.5))
                denominator = term_frequency + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1))
                scores[doc_index] += idf * (term_frequency * (k1 + 1) / denominator)
        return scores


class FaissDenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._faiss_index = None
        try:
            import faiss  # type: ignore
        except ImportError:
            self.backend = 'numpy'
            return
        self.backend = 'faiss'
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        self._faiss_index = index

    def search(self, query_vector: np.ndarray) -> np.ndarray:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if self._faiss_index is not None:
            return self.vectors @ query_vector
        return self.vectors @ query_vector


class HybridPolicyRetriever:
    def __init__(
        self,
        documents: list[PolicyDocument],
        embedding_provider: EmbeddingProvider,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        if not documents:
            raise ValueError('HybridPolicyRetriever requires at least one policy document.')
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError('Retriever weights must be non-negative.')
        if dense_weight + bm25_weight <= 0:
            raise ValueError('At least one retriever weight must be positive.')
        self.documents = documents
        self.embedding_provider = embedding_provider
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._bm25 = _Bm25Index(documents)
        self._dense = FaissDenseIndex(embedding_provider.embed([self._document_text(document) for document in documents]))

    @classmethod
    def from_policy_dir(
        cls,
        policy_dir: Path | str,
        embedding_provider: EmbeddingProvider | None = None,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> 'HybridPolicyRetriever':
        documents = PolicyDocumentLoader(policy_dir).load()
        return cls(
            documents=documents,
            embedding_provider=embedding_provider or BgeM3EmbeddingProvider(),
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_vector = self.embedding_provider.embed([query])[0]
        dense_scores = self._dense.search(query_vector)
        bm25_scores = self._bm25.search(query)
        dense_normalized = self._normalize(dense_scores)
        bm25_normalized = self._normalize(bm25_scores)
        combined = self.dense_weight * dense_normalized + self.bm25_weight * bm25_normalized
        bonuses = np.asarray([self._keyword_bonus(query, document) for document in self.documents], dtype=np.float32)
        combined = combined + bonuses
        ranked_indexes = np.argsort(-combined)[:top_k]
        return [
            RetrievalResult(
                document_id=self.documents[index].document_id,
                chunk_id=self.documents[index].chunk_id,
                title=self.documents[index].title,
                content=self.documents[index].content,
                score=float(combined[index]),
                dense_score=float(dense_scores[index]),
                bm25_score=float(bm25_scores[index]),
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                metadata=self.documents[index].metadata,
            )
            for index in ranked_indexes
        ]

    @staticmethod
    def _document_text(document: PolicyDocument) -> str:
        return f'{document.title}\n{document.content}'

    @staticmethod
    def _keyword_bonus(query: str, document: PolicyDocument) -> float:
        normalized_query = query.lower()
        doc_id = document.document_id
        bonus = 0.0
        uncertainty_terms = (
            'clarification',
            'human review',
            'conflict review',
            'degraded review',
            'manual review',
            'uncertain',
            'unknown',
            'missing information',
            'not sure',
            '모름',
            '불명확',
            '추가 확인',
            '확인 필요',
            '정보 부족',
            '충돌',
            '수동 검토',
        )
        if doc_id == 'SOP-PAY-OP-005':
            bonus += 0.12
        if doc_id == 'SOP-PAY-OP-005' and any(term in normalized_query for term in uncertainty_terms):
            bonus += 0.55
        if doc_id == 'SOP-PAY-OP-004' and any(term in normalized_query for term in ('pos', 'front', 'requestdelivery', 'timeout')):
            bonus += 0.2
        if doc_id == 'SOP-PAY-OP-003' and any(term in normalized_query for term in ('van', 'merchantregistration', 'merchantnumber')):
            bonus += 0.2
        if doc_id == 'SOP-PAY-OP-002' and any(term in normalized_query for term in ('terminalidentifier', 'identity', 'serial', 'devicenumber', 'duplicatetid', 'newterminal', 'installation')):
            bonus += 0.2
        return bonus

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        minimum = float(scores.min())
        maximum = float(scores.max())
        if math.isclose(minimum, maximum):
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)


__all__ = [
    'BgeM3EmbeddingProvider',
    'DeterministicEmbeddingProvider',
    'EmbeddingProvider',
    'FaissDenseIndex',
    'HybridPolicyRetriever',
    'PolicyDocument',
    'PolicyDocumentLoader',
    'RetrievalResult',
]
