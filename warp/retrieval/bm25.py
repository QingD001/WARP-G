"""可复现、支持区域过滤的 Okapi BM25 实现。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from warp.models import Document, SearchResult
from warp.utils import tokenize


class BM25Retriever:
    """Dependency-free Okapi BM25 with optional region filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.documents: list[Document] = []
        self.term_freqs: list[Counter[str]] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, documents: list[Document]) -> "BM25Retriever":
        """构建词频、倒排表、IDF 和平均文档长度统计。"""
        self.documents = list(documents)
        self.term_freqs = [Counter(tokenize(doc.content)) for doc in documents]
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, frequencies in enumerate(self.term_freqs):
            for term, frequency in frequencies.items():
                postings[term].append((index, frequency))
        self.postings = dict(postings)
        n = len(documents)
        self.idf = {term: math.log(1.0 + (n - len(rows) + 0.5) / (len(rows) + 0.5))
                    for term, rows in self.postings.items()}
        self.avgdl = sum(sum(freq.values()) for freq in self.term_freqs) / max(n, 1)
        return self

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """按 BM25 得分检索；doc_ids 用于图区域内部的受限 seed 检索。"""
        if k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        allowed = None if doc_ids is None else set(doc_ids)
        for term, qtf in Counter(tokenize(query)).items():
            idf = self.idf.get(term, 0.0)
            for index, frequency in self.postings.get(term, []):
                doc = self.documents[index]
                if allowed is not None and doc.id not in allowed:
                    continue
                dl = sum(self.term_freqs[index].values())
                norm = frequency + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[index] += qtf * idf * frequency * (self.k1 + 1.0) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.documents[item[0]].id))[:k]
        return [SearchResult(self.documents[index].id, score, "bm25", rank + 1)
                for rank, (index, score) in enumerate(ranked)]
