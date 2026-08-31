"""BM25、Dense 与图结果所共用的 Reciprocal Rank Fusion。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from warp.models import Document, SearchResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .reranker import Reranker


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    k: int = 10,
    rrf_constant: int = 60,
    weights: Sequence[float] | None = None,
    source: str = "rrf",
) -> list[SearchResult]:
    """按排名而非异构原始分数融合结果，并按 doc_id 确定性打破平局。"""
    scores: dict[str, float] = defaultdict(float)
    regions: dict[str, str | None] = {}
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("RRF requires exactly one weight per result list")
    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results, 1):
            scores[result.doc_id] += weight / (rrf_constant + rank)
            regions.setdefault(result.doc_id, result.region_id)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [SearchResult(doc_id, score, source, rank + 1, regions[doc_id])
            for rank, (doc_id, score) in enumerate(ranked)]


class HybridRetriever:
    """所有方法都拥有的 BM25 + Dense 基础物理表示。"""
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_constant: int = 60) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_constant = rrf_constant

    def fit(self, documents: list[Document]) -> "HybridRetriever":
        """在同一 corpus 上同时拟合 lexical 与 dense 两套 Base 索引。"""
        self.bm25.fit(documents)
        self.dense.fit(documents)
        return self

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """各取更深候选后做 RRF，降低单路截断造成的候选损失。"""
        depth = max(k * 2, k)
        return reciprocal_rank_fusion(
            [self.bm25.search(query, depth, doc_ids), self.dense.search(query, depth, doc_ids)],
            k=k, rrf_constant=self.rrf_constant, source="hybrid",
        )


def fuse_and_rerank(
    query: str, result_lists: Sequence[Sequence[SearchResult]], reranker: Reranker, *,
    k: int, candidate_k: int, source: str, weights: Sequence[float] | None = None,
) -> list[SearchResult]:
    """供 Base、probe、selective 与 Full Graph 共用的唯一最终排序路径。"""
    if candidate_k < k:
        raise ValueError("candidate_k must be at least k")
    fused = reciprocal_rank_fusion(result_lists, candidate_k, weights=weights, source=source)
    return reranker.rerank(query, fused, k)
