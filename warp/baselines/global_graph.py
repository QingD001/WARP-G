"""KET-RAG、G2ConS 与 document-level matched-budget baselines。

KET-RAG 实现 KG skeleton + keyword bipartite retrieval；G2ConS 实现 concept graph +
core-KG dual-path retrieval。两者的昂贵 KG 都调用与 WARP 相同的 HippoRAG2 builder。
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import faiss
import numpy as np

from warp.eval.construction_cost import aggregate_costs
from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import ConstructionCost, Document, Region, SearchResult
from warp.retrieval.ann import semantic_knn
from warp.retrieval.hybrid import HybridRetriever, fuse_and_rerank
from warp.retrieval.reranker import Reranker
from warp.utils import cosine, tokenize


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
}
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _keywords(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


def _pagerank(nodes: list[str], edges: dict[tuple[str, str], float], damping: float = 0.85,
              iterations: int = 100, tolerance: float = 1e-10) -> dict[str, float]:
    if not nodes:
        raise ValueError("PageRank requires nodes")
    neighbors: dict[str, dict[str, float]] = {node: {} for node in nodes}
    for (left, right), weight in edges.items():
        neighbors[left][right] = neighbors[left].get(right, 0.0) + weight
        neighbors[right][left] = neighbors[right].get(left, 0.0) + weight
    rank = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(iterations):
        dangling = sum(rank[node] for node in nodes if not neighbors[node])
        updated = {node: (1.0 - damping) / len(nodes) + damping * dangling / len(nodes) for node in nodes}
        for node in nodes:
            total = sum(neighbors[node].values())
            if total:
                for other, weight in neighbors[node].items():
                    updated[other] += damping * rank[node] * weight / total
        delta = sum(abs(updated[node] - rank[node]) for node in nodes)
        rank = updated
        if delta <= tolerance:
            break
    return rank


def _select_by_cost(ranked: list[str], costs: dict[str, float], budget: float) -> list[str]:
    selected: list[str] = []
    spent = 0.0
    for doc_id in ranked:
        if spent + costs[doc_id] <= budget + 1e-9:
            selected.append(doc_id)
            spent += costs[doc_id]
    return selected


@dataclass
class LightweightGraphIndex:
    name: str
    concept_ids: list[str]
    concept_vectors: np.ndarray
    concept_to_docs: dict[str, list[str]]
    edges: dict[tuple[str, str], float]
    cost: ConstructionCost
    top_concepts: int = 25
    expansion_depth: int = 2

    def __post_init__(self) -> None:
        if not self.concept_ids or self.concept_vectors.shape[0] != len(self.concept_ids):
            raise ValueError("Lightweight graph requires aligned concepts and vectors")
        matrix = np.asarray(self.concept_vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self.concept_vectors = matrix
        self.index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128
        self.index.add(matrix)
        self.neighbors: dict[str, set[str]] = {concept: set() for concept in self.concept_ids}
        for left, right in self.edges:
            self.neighbors[left].add(right)
            self.neighbors[right].add(left)

    def search(self, query: str, dense: Any, k: int) -> list[SearchResult]:
        query_vector = np.asarray(dense.encode([query]), dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, positions = self.index.search(query_vector, min(self.top_concepts, len(self.concept_ids)))
        frontier: dict[str, tuple[int, float]] = {}
        for score, position in zip(scores[0], positions[0]):
            if position >= 0:
                frontier[self.concept_ids[int(position)]] = (0, float(score))
        active = set(frontier)
        for depth in range(1, self.expansion_depth + 1):
            expanded = {other for concept in active for other in self.neighbors[concept]} - set(frontier)
            for concept in expanded:
                frontier[concept] = (depth, 0.0)
            active = expanded
        doc_scores: dict[str, float] = defaultdict(float)
        for concept, (depth, similarity) in frontier.items():
            contribution = (similarity if depth == 0 else 1.0) / (depth + 1)
            for doc_id in self.concept_to_docs[concept]:
                doc_scores[doc_id] += contribution
        ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [SearchResult(doc_id, score, self.name, rank + 1) for rank, (doc_id, score) in enumerate(ranked)]


class _ConceptResources:
    def __init__(self, documents: list[Document], dense: Any) -> None:
        started = time.perf_counter()
        self.doc_keywords = {doc.id: _keywords(doc.content) for doc in documents}
        concept_sentences: dict[str, set[str]] = defaultdict(set)
        for doc in documents:
            sentences = [part.strip() for part in SENTENCE_RE.split(doc.content) if part.strip()]
            for sentence in sentences:
                for concept in _keywords(sentence):
                    concept_sentences[concept].add(sentence)
        sentence_list = sorted({sentence for values in concept_sentences.values() for sentence in values})
        sentence_vectors = dense.encoder.encode_documents(sentence_list)
        sentence_index = {sentence: index for index, sentence in enumerate(sentence_list)}
        self.concepts = sorted(concept_sentences)
        vectors: list[list[float]] = []
        for concept in self.concepts:
            rows = np.asarray([sentence_vectors[sentence_index[value]] for value in concept_sentences[concept]], dtype="float32")
            vector = rows.mean(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise RuntimeError(f"Concept {concept!r} has a zero embedding")
            vectors.append((vector / norm).tolist())
        self.vectors = np.asarray(vectors, dtype="float32")
        self.vector_map = {concept: self.vectors[index] for index, concept in enumerate(self.concepts)}
        postings: dict[str, list[str]] = defaultdict(list)
        for doc_id, concepts in self.doc_keywords.items():
            for concept in concepts:
                postings[concept].append(doc_id)
        self.concept_to_docs = {concept: sorted(postings[concept]) for concept in self.concepts}
        embedding_tokens = sum(len(tokenize(sentence)) for sentence in sentence_list)
        self.base_cost = ConstructionCost(
            embedding_tokens=embedding_tokens,
            wall_seconds=time.perf_counter() - started,
            nodes=len(self.concepts) + len(documents),
            storage_bytes=int(self.vectors.nbytes),
        )


class GlobalBaselineFactory:
    """一次构建 KET/G2ConS 轻量资源，并按预算物化各自 core graph。"""

    def __init__(self, documents: list[Document], base: HybridRetriever, graph_builder: GraphBuilder,
                 graph_retriever: GraphRetriever, reranker: Reranker, candidate_k: int,
                 ket_knn_k: int = 10, g2_semantic_threshold: float = 0.65,
                 g2_cooccurrence_threshold: int = 3) -> None:
        self.documents = documents
        self.doc_map = {doc.id: doc for doc in documents}
        self.base = base
        self.graph_builder = graph_builder
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.doc_costs = {doc.id: float(len(tokenize(doc.content))) for doc in documents}
        if any(cost <= 0 for cost in self.doc_costs.values()):
            raise ValueError("Global baselines require non-empty documents")
        self.resources = _ConceptResources(documents, base.dense)
        self.ket_index, self.ket_ranking = self._build_ket(ket_knn_k)
        self.g2_index, self.g2_ranking = self._build_g2(g2_semantic_threshold, g2_cooccurrence_threshold)

    def _build_ket(self, k: int) -> tuple[LightweightGraphIndex, list[str]]:
        if k <= 0 or k % 2:
            raise ValueError("KET K must be a positive even integer")
        started = time.perf_counter()
        ids = sorted(self.doc_map)
        lexical_counts: dict[tuple[str, str], float] = defaultdict(float)
        postings: dict[str, list[str]] = defaultdict(list)
        for doc_id, words in self.resources.doc_keywords.items():
            for word in words:
                postings[word].append(doc_id)
        for doc_ids in postings.values():
            for left, right in combinations(sorted(doc_ids), 2):
                lexical_counts[(left, right)] += 1.0
        lexical_neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (left, right), score in lexical_counts.items():
            lexical_neighbors[left].append((right, score))
            lexical_neighbors[right].append((left, score))
        semantic = semantic_knn(ids, [self.base.dense.vector(doc_id) for doc_id in ids], k // 2)
        edges: dict[tuple[str, str], float] = {}
        for doc_id in ids:
            lexical = sorted(lexical_neighbors[doc_id], key=lambda item: (-item[1], item[0]))[:k // 2]
            for other, score in lexical + semantic[doc_id]:
                edge = tuple(sorted((doc_id, other)))
                edges[edge] = 1.0
        rank = _pagerank(ids, edges)
        ranking = sorted(ids, key=lambda key: (-rank[key], key))
        cost = ConstructionCost(
            embedding_tokens=self.resources.base_cost.embedding_tokens,
            wall_seconds=self.resources.base_cost.wall_seconds + time.perf_counter() - started,
            nodes=self.resources.base_cost.nodes,
            edges=sum(len(values) for values in self.resources.concept_to_docs.values()),
            storage_bytes=self.resources.base_cost.storage_bytes,
        )
        return LightweightGraphIndex(
            "ket_keyword", self.resources.concepts, self.resources.vectors,
            self.resources.concept_to_docs, {}, cost, expansion_depth=0,
        ), ranking

    def _build_g2(self, threshold: float, co_threshold: int) -> tuple[LightweightGraphIndex, list[str]]:
        started = time.perf_counter()
        cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        for words in self.resources.doc_keywords.values():
            eligible = sorted(word for word in words if len(self.resources.concept_to_docs[word]) >= co_threshold)
            for left, right in combinations(eligible, 2):
                cooccurrence[(left, right)] += 1
        edges: dict[tuple[str, str], float] = {}
        for (left, right), count in cooccurrence.items():
            if count < co_threshold:
                continue
            if cosine(self.resources.vector_map[left], self.resources.vector_map[right]) < threshold:
                continue
            denominator = len(self.resources.concept_to_docs[left]) + len(self.resources.concept_to_docs[right])
            edges[(left, right)] = 2.0 * count / denominator
        concept_rank = _pagerank(self.resources.concepts, edges)
        chunk_score = {
            doc_id: sum(concept_rank[word] for word in words)
            for doc_id, words in self.resources.doc_keywords.items()
        }
        ranking = sorted(chunk_score, key=lambda key: (-chunk_score[key], key))
        payload_size = len(json.dumps({"edges": [(left, right, value) for (left, right), value in edges.items()]}).encode())
        cost = ConstructionCost(
            embedding_tokens=self.resources.base_cost.embedding_tokens,
            wall_seconds=self.resources.base_cost.wall_seconds + time.perf_counter() - started,
            nodes=self.resources.base_cost.nodes,
            edges=len(edges) + sum(len(values) for values in self.resources.concept_to_docs.values()),
            storage_bytes=self.resources.base_cost.storage_bytes + payload_size,
        )
        return LightweightGraphIndex(
            "g2cons_concept", self.resources.concepts, self.resources.vectors,
            self.resources.concept_to_docs, edges, cost,
        ), ranking

    def build(self, method: str, graph_budget: float) -> "GlobalGraphBaseline":
        method = method.lower()
        if graph_budget < 0:
            raise ValueError("Global baseline budget must be non-negative")
        if graph_budget == 0:
            return GlobalGraphBaseline(
                method, self.base, self.graph_retriever, self.reranker,
                self.candidate_k, None, None,
            )
        if method == "ket_rag":
            ranking, lightweight = self.ket_ranking, self.ket_index
        elif method == "g2cons":
            ranking, lightweight = self.g2_ranking, self.g2_index
        else:
            raise ValueError(f"Unknown global baseline: {method}")
        lightweight_cost = lightweight.cost.selection_cost if lightweight is not None else 0.0
        if lightweight_cost > graph_budget:
            return GlobalGraphBaseline(
                method, self.base, self.graph_retriever, self.reranker,
                self.candidate_k, None, None,
            )
        selected = _select_by_cost(ranking, self.doc_costs, graph_budget - lightweight_cost)
        graph = None
        if selected:
            graph = self.graph_builder.build(Region(f"__{method}__", selected), self.documents)
        return GlobalGraphBaseline(
            method, self.base, self.graph_retriever, self.reranker, self.candidate_k,
            graph, lightweight,
        )


class GlobalGraphBaseline:
    def __init__(self, name: str, base: HybridRetriever, graph_retriever: GraphRetriever,
                 reranker: Reranker, candidate_k: int, graph: RegionalGraph | None,
                 lightweight: LightweightGraphIndex | None) -> None:
        self.name = name
        self.base = base
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.graph = graph
        self.lightweight = lightweight

    @property
    def cost(self) -> ConstructionCost:
        costs = []
        if self.graph is not None:
            costs.append(self.graph.cost)
        if self.lightweight is not None:
            costs.append(self.lightweight.cost)
        return aggregate_costs(costs)

    def search(self, query: str, k: int) -> list[SearchResult]:
        result_lists = [self.base.search(query, self.candidate_k)]
        weights = [1.0]
        if self.graph is not None:
            result_lists.append(self.graph_retriever.search(query, self.graph, self.candidate_k))
            weights.append(0.6 if self.name == "g2cons" else 0.5 if self.name == "ket_rag" else 1.0)
        if self.lightweight is not None:
            result_lists.append(self.lightweight.search(query, self.base.dense, self.candidate_k))
            weights.append(0.4 if self.name == "g2cons" else 0.5 if self.name == "ket_rag" else 1.0)
        return fuse_and_rerank(
            query, result_lists, self.reranker, k=k,
            candidate_k=self.candidate_k, source=self.name, weights=weights,
        )
