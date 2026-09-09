"""由 train/design workload 构造便宜的文档共访问图。

这里的图仅用于 corpus partition，不是知识图谱。边权由 query top-k 共现次数和
小权重语义 kNN 两部分组成；dev/test query 永远不能传入该 builder。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from warp.models import Document, Query
from warp.retrieval.dense import DenseRetriever
from warp.retrieval.hybrid import HybridRetriever
from warp.retrieval.ann import semantic_knn


@dataclass
class CoaccessGraph:
    """保存总边、query 边、semantic 边及可审计的每题检索结果。"""
    nodes: list[str]
    edges: dict[tuple[str, str], float]
    query_edges: dict[tuple[str, str], float]
    semantic_edges: dict[tuple[str, str], float]
    query_results: dict[str, list[str]]

    def neighbors(self) -> dict[str, dict[str, float]]:
        """把无向 edge map 展开为 label propagation 使用的邻接表。"""
        output: dict[str, dict[str, float]] = {node: {} for node in self.nodes}
        for (left, right), weight in self.edges.items():
            if left not in output or right not in output:
                continue
            output[left][right] = weight
            output[right][left] = weight
        return output

    def subgraph(self, nodes: list[str]) -> "CoaccessGraph":
        """取出节点诱导子图，供过贵区域递归再切。"""
        allowed = set(nodes)
        ordered = [node for node in self.nodes if node in allowed]
        def _clip(edges: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
            return {edge: weight for edge, weight in edges.items() if edge[0] in allowed and edge[1] in allowed}
        query_results = {
            query_id: [doc_id for doc_id in doc_ids if doc_id in allowed]
            for query_id, doc_ids in self.query_results.items()
        }
        return CoaccessGraph(ordered, _clip(self.edges), _clip(self.query_edges),
                             _clip(self.semantic_edges), query_results)


class CoaccessGraphBuilder:
    """实现 w_ij = w_query + lambda * w_semantic；query 边默认 rank decay × 文档 IDF。"""
    def __init__(self, top_k: int = 20, semantic_k: int = 3, semantic_lambda: float = 0.05,
                 min_semantic_similarity: float = 0.0, weight_mode: str = "rank_idf") -> None:
        if weight_mode not in {"count", "rank_idf"}:
            raise ValueError("co-access weight_mode must be count or rank_idf")
        self.top_k = top_k
        self.semantic_k = semantic_k
        self.semantic_lambda = semantic_lambda
        self.min_semantic_similarity = min_semantic_similarity
        self.weight_mode = weight_mode

    def build(
        self,
        documents: list[Document],
        queries: list[Query],
        retriever: HybridRetriever,
        query_top_ids: dict[str, list[str]] | None = None,
    ) -> CoaccessGraph:
        """为每个设计 query 的 Base top-k 文档两两加边，再补语义邻居弱边。"""
        # query_edges 是真实 workload 共访问强度，也是社区划分的主要信号。
        query_edges: dict[tuple[str, str], float] = defaultdict(float)
        query_results: dict[str, list[str]] = {}
        ranked_lists: list[list[str]] = []
        for query in queries:
            if query_top_ids is not None:
                ids = list(dict.fromkeys(query_top_ids[query.id][:self.top_k]))
            else:
                ids = [result.doc_id for result in retriever.search(query.text, self.top_k)]
                ids = list(dict.fromkeys(ids))
            query_results[query.id] = ids
            ranked_lists.append(ids)
        df: dict[str, int] = defaultdict(int)
        for ids in ranked_lists:
            for doc_id in ids:
                df[doc_id] += 1
        n_queries = max(len(queries), 1)
        for ids in ranked_lists:
            ranks = {doc_id: rank for rank, doc_id in enumerate(ids, 1)}
            for left, right in combinations(sorted(set(ids)), 2):
                query_edges[(left, right)] += self._pair_weight(ranks[left], ranks[right], df[left], df[right], n_queries)

        # semantic_edges 只避免未被训练 query 覆盖的文档完全孤立。
        semantic_edges: dict[tuple[str, str], float] = defaultdict(float)
        if self.semantic_k > 0 and self.semantic_lambda > 0:
            dense: DenseRetriever = retriever.dense
            ids = [doc.id for doc in documents]
            neighbors = semantic_knn(ids, [dense.vector(doc_id) for doc_id in ids], self.semantic_k)
            for doc_id, values in neighbors.items():
                for other_id, similarity in values:
                    if similarity <= self.min_semantic_similarity:
                        continue
                    edge = tuple(sorted((doc_id, other_id)))
                    semantic_edges[edge] = max(semantic_edges[edge], similarity)

        edges: dict[tuple[str, str], float] = dict(query_edges)
        for edge, similarity in semantic_edges.items():
            edges[edge] = edges.get(edge, 0.0) + self.semantic_lambda * similarity
        return CoaccessGraph([doc.id for doc in documents], edges, dict(query_edges),
                             dict(semantic_edges), query_results)

    def _pair_weight(self, rank_left: int, rank_right: int, df_left: int, df_right: int,
                     n_queries: int) -> float:
        if self.weight_mode == "count":
            return 1.0
        rank_term = 1.0 / (math.log(1 + rank_left) * math.log(1 + rank_right))
        idf_left = math.log(n_queries / max(df_left, 1))
        idf_right = math.log(n_queries / max(df_right, 1))
        return rank_term * max(idf_left, 0.0) * max(idf_right, 0.0)
