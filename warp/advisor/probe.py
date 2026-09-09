"""分层选择 Region 构图，并用最终部署排序路径生成监督标签。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import Document, Query, Region, RegionFeatures, SearchResult
from warp.retrieval.hybrid import fuse_and_rerank
from warp.retrieval.reranker import Reranker
from warp.utils import write_json


PROBE_OUTCOME_NAME = "warp_probe_outcome.json"


def evidence_recall(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    """整题 Evidence Recall：gold 为该 query 的全部证据，不按 region 截断。"""
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Probe queries require gold evidence")
    return len(gold & {result.doc_id for result in results}) / len(gold)


def complete_evidence(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    """整题 CompleteEvidence：全部 gold 是否都进入结果，与评测指标 M 相同。"""
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
    """按最终部署路径测量 Graph_i 对整题 M 的边际增益。

    y_i 对路由到该区的 design query 平均 CompleteEvidence@10(Base+Graph_i) 与 Base 的差。
    gold 集合是该 query 的全部证据，不是 region ∩ gold。
    """

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

    def select_probe_regions(
        self, regions: list[Region], features: dict[str, RegionFeatures],
        costs: dict[str, float], count: int | None = None,
        predicted_gains: dict[str, float] | None = None,
        already_probed: set[str] | None = None,
        strategy: str = "active",
    ) -> list[Region]:
        eligible = [
            region for region in regions
            if features[region.id].query_freq > 0 and region.id not in (already_probed or set())
        ]
        target = count if count is not None else min(len(eligible), max(6, round(len(eligible) * self.probe_fraction)))
        if len(eligible) < 1:
            raise ValueError("Active probing requires workload-covered regions")
        target = min(target, len(eligible))
        if strategy == "stratified":
            return self._select_stratified(eligible, features, costs, target)
        if predicted_gains:
            return self._select_uncertain(eligible, features, predicted_gains, target)
        return self._select_demand_diverse(eligible, features, costs, target)

    def _select_stratified(
        self, eligible: list[Region], features: dict[str, RegionFeatures],
        costs: dict[str, float], count: int,
    ) -> list[Region]:
        ordered = sorted(eligible, key=lambda region: (
            features[region.id].query_freq / features[region.id].num_tokens,
            features[region.id].failure_rate,
            costs[region.id],
            region.id,
        ))
        rng = random.Random(self.seed)
        buckets = [ordered[index::count] for index in range(count)]
        return sorted((rng.choice(bucket) for bucket in buckets), key=lambda region: region.id)

    def _select_demand_diverse(
        self, eligible: list[Region], features: dict[str, RegionFeatures],
        costs: dict[str, float], count: int,
    ) -> list[Region]:
        """第一轮：沿 failure/demand/cost 分桶，桶内取 demand×failure 最大而不是随机。"""
        ordered = sorted(eligible, key=lambda region: (
            features[region.id].query_freq / max(features[region.id].num_tokens, 1.0),
            features[region.id].failure_rate,
            costs[region.id],
            region.id,
        ))
        buckets = [ordered[index::count] for index in range(count)]
        chosen = []
        for bucket in buckets:
            chosen.append(max(
                bucket,
                key=lambda region: (
                    features[region.id].query_freq * features[region.id].failure_rate,
                    features[region.id].query_freq,
                    region.id,
                ),
            ))
        return sorted(chosen, key=lambda region: region.id)

    def _select_uncertain(
        self, eligible: list[Region], features: dict[str, RegionFeatures],
        predicted_gains: dict[str, float], count: int,
    ) -> list[Region]:
        """后续轮：高需求且预测接近决策边界的区域。"""
        ranked = sorted(
            eligible,
            key=lambda region: (
                -features[region.id].query_freq / (abs(predicted_gains.get(region.id, 0.0)) + 1e-6),
                -features[region.id].failure_rate,
                region.id,
            ),
        )
        return sorted(ranked[:count], key=lambda region: region.id)

    def run(
        self, probe_regions: list[Region], documents: list[Document], queries: list[Query],
        region_queries: dict[str, list[str]], base_results: dict[str, list[SearchResult]],
    ) -> dict[str, ProbeOutcome]:
        query_map = {query.id: query for query in queries}
        outcomes: dict[str, ProbeOutcome] = {}
        for region in probe_regions:
            graph = self.graph_builder.build(region, documents)
            outcome_path = _probe_outcome_path(graph)
            if outcome_path.exists():
                outcomes[region.id] = _load_probe_outcome(outcome_path, graph)
                print(json.dumps({
                    "probe_checkpoint": "reuse",
                    "region_id": region.id,
                    "path": str(outcome_path),
                }, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({
                "probe_checkpoint": "label",
                "region_id": region.id,
                "path": str(outcome_path),
            }, ensure_ascii=False), flush=True)
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
            write_json(outcome_path, _probe_outcome_payload(outcomes[region.id]))
            print(json.dumps({
                "probe_checkpoint": "saved",
                "region_id": region.id,
                "path": str(outcome_path),
                "gain": outcomes[region.id].gain,
                "query_count": outcomes[region.id].query_count,
            }, ensure_ascii=False), flush=True)
        return outcomes


def _probe_outcome_path(graph: RegionalGraph) -> Path:
    artifact_dir = graph.metadata.get("artifact_dir")
    if not artifact_dir:
        raise RuntimeError("Probe graph is missing artifact_dir metadata")
    return Path(artifact_dir) / PROBE_OUTCOME_NAME


def _probe_outcome_payload(outcome: ProbeOutcome) -> dict[str, float | int | str]:
    return {
        "region_id": outcome.region_id,
        "gain": outcome.gain,
        "recall_gain": outcome.recall_gain,
        "complete_gain": outcome.complete_gain,
        "base_utility": outcome.base_utility,
        "graph_utility": outcome.graph_utility,
        "query_count": outcome.query_count,
    }


def _load_probe_outcome(path: Path, graph: RegionalGraph) -> ProbeOutcome:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "region_id", "gain", "recall_gain", "complete_gain",
        "base_utility", "graph_utility", "query_count",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing probe outcome fields: {sorted(missing)}")
    if str(payload["region_id"]) != graph.region_id:
        raise ValueError(f"{path} region_id does not match graph {graph.region_id}")
    return ProbeOutcome(
        str(payload["region_id"]),
        float(payload["gain"]),
        float(payload["recall_gain"]),
        float(payload["complete_gain"]),
        float(payload["base_utility"]),
        float(payload["graph_utility"]),
        int(payload["query_count"]),
        graph,
    )
