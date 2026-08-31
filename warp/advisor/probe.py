"""分层选择 Region 构图，并用最终部署排序路径生成监督标签。"""

from __future__ import annotations

import random
from dataclasses import dataclass

from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import Document, Query, Region, RegionFeatures, SearchResult
from warp.retrieval.hybrid import fuse_and_rerank
from warp.retrieval.reranker import Reranker


def evidence_recall(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Probe queries require gold evidence")
    return len(gold & {result.doc_id for result in results}) / len(gold)


def complete_evidence(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Probe queries require gold evidence")
    return float(gold.issubset({result.doc_id for result in results}))


@dataclass
class ProbeOutcome:
    region_id: str
    gain: float
    recall_gain: float
    complete_gain: float
    base_utility: float
    graph_utility: float
    query_count: int
    graph: RegionalGraph


class RegionProber:
    """使用与上线检索完全一致的 RRF + CrossEncoder 计算 counterfactual gain。"""

    def __init__(
        self, graph_builder: GraphBuilder, graph_retriever: GraphRetriever, reranker: Reranker,
        probe_fraction: float, retrieval_k: int, candidate_k: int, objective: str, seed: int,
    ) -> None:
        if not 0.0 < probe_fraction <= 1.0:
            raise ValueError("probe_fraction must be in (0, 1]")
        if objective not in {"evidence_recall", "complete_evidence"}:
            raise ValueError("Probe objective must be evidence_recall or complete_evidence")
        self.graph_builder = graph_builder
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.probe_fraction = probe_fraction
        self.retrieval_k = retrieval_k
        self.candidate_k = candidate_k
        self.objective = objective
        self.seed = seed

    def select_probe_regions(self, regions: list[Region], features: dict[str, RegionFeatures],
                             costs: dict[str, float]) -> list[Region]:
        eligible = [region for region in regions if features[region.id].query_freq > 0]
        count = min(len(eligible), max(6, round(len(eligible) * self.probe_fraction)))
        if len(eligible) < 6:
            raise ValueError("Scientific benefit prediction requires at least six workload-covered regions")
        # 沿 workload density、failure 与估算 cost 三轴排序后轮转桶抽样。
        ordered = sorted(eligible, key=lambda region: (
            features[region.id].query_freq / features[region.id].num_tokens,
            features[region.id].failure_rate,
            costs[region.id],
            region.id,
        ))
        rng = random.Random(self.seed)
        buckets = [ordered[index::count] for index in range(count)]
        return sorted((rng.choice(bucket) for bucket in buckets), key=lambda region: region.id)

    def run(
        self, probe_regions: list[Region], documents: list[Document], queries: list[Query],
        region_queries: dict[str, list[str]], base_results: dict[str, list[SearchResult]],
    ) -> dict[str, ProbeOutcome]:
        query_map = {query.id: query for query in queries}
        outcomes: dict[str, ProbeOutcome] = {}
        for region in probe_regions:
            graph = self.graph_builder.build(region, documents)
            base_recall_values: list[float] = []
            graph_recall_values: list[float] = []
            base_complete_values: list[float] = []
            graph_complete_values: list[float] = []
            for query_id in region_queries.get(region.id, []):
                query = query_map[query_id]
                base_ranked = fuse_and_rerank(
                    query.text, [base_results[query_id]], self.reranker,
                    k=self.retrieval_k, candidate_k=self.candidate_k, source="probe_base",
                )
                graph_results = self.graph_retriever.search(query.text, graph, self.candidate_k)
                graph_ranked = fuse_and_rerank(
                    query.text, [base_results[query_id], graph_results], self.reranker,
                    k=self.retrieval_k, candidate_k=self.candidate_k, source="probe_graph",
                )
                base_recall_values.append(evidence_recall(base_ranked, query.gold_doc_ids))
                graph_recall_values.append(evidence_recall(graph_ranked, query.gold_doc_ids))
                base_complete_values.append(complete_evidence(base_ranked, query.gold_doc_ids))
                graph_complete_values.append(complete_evidence(graph_ranked, query.gold_doc_ids))
            if not base_recall_values:
                raise ValueError(f"Probe region {region.id} has no routed design queries")
            base_recall = sum(base_recall_values) / len(base_recall_values)
            graph_recall = sum(graph_recall_values) / len(graph_recall_values)
            base_complete = sum(base_complete_values) / len(base_complete_values)
            graph_complete = sum(graph_complete_values) / len(graph_complete_values)
            recall_gain = graph_recall - base_recall
            complete_gain = graph_complete - base_complete
            if self.objective == "evidence_recall":
                gain, base_utility, graph_utility = recall_gain, base_recall, graph_recall
            else:
                gain, base_utility, graph_utility = complete_gain, base_complete, graph_complete
            outcomes[region.id] = ProbeOutcome(
                region.id, gain, recall_gain, complete_gain, base_utility, graph_utility,
                len(base_recall_values), graph,
            )
        return outcomes
