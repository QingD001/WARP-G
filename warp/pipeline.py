"""WARP-G 的训练/设计阶段与测试阶段编排器。

`fit` 是唯一允许读取 train/design query 的设计入口；评测函数只接收显式传入的
dev/test query。图 builder、图 retriever、base retriever 和 reranker 均可注入，
因此 advisor 算法不绑定具体 GraphRAG 实现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
import random
import statistics
from typing import Any

from warp.advisor.features import RegionFeatureExtractor
from warp.advisor.predictor import BenefitPredictor
from warp.advisor.probe import ProbeOutcome, RegionProber, complete_evidence, evidence_recall
from warp.advisor.selector import RegionSelector
from warp.eval.construction_cost import aggregate_costs
from warp.eval.cutoffs import RETRIEVAL_KS
from warp.eval.retrieval import evaluate_retrieval
from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import DatasetBundle, Region, RegionFeatures, SearchResult
from warp.partition.boundary import region_coaccess_neighbors, replicate_boundaries
from warp.partition.coaccess_graph import CoaccessGraph, CoaccessGraphBuilder
from warp.partition.leiden import RegionPartitioner
from warp.retrieval.hybrid import HybridRetriever, fuse_and_rerank
from warp.retrieval.reranker import Reranker
from warp.utils import cosine


@dataclass
class WARPConfig:
    """与具体 HippoRAG2 参数解耦的 WARP 物理设计超参数。"""
    retrieval_k: int = 10
    candidate_k: int = 50
    routing_k: int = 20
    coaccess_k: int = 20
    semantic_k: int = 3
    semantic_lambda: float = 0.05
    probe_fraction: float = 0.2
    benefit_objective: str = "complete_evidence"
    dispersion_pairs: int = 4096
    interaction_pairs: int = 30
    partition_resolution: float = 1.0
    min_region_size: int = 1
    partition_mode: str = "combined"
    seed: int = 42
    retrieval_ks: tuple[int, ...] = RETRIEVAL_KS
    region_router_k: int = 3
    bridge_k: int = 1
    boundary_replication: float = 0.05
    max_region_cost: float | None = None
    coaccess_weight: str = "rank_idf"
    probe_strategy: str = "active"
    multistep_max_steps: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.retrieval_ks, list):
            self.retrieval_ks = tuple(int(value) for value in self.retrieval_ks)
        if not self.retrieval_ks or any(k <= 0 for k in self.retrieval_ks):
            raise ValueError("retrieval_ks must be positive")
        if self.retrieval_k <= 0 or self.routing_k <= 0 or self.candidate_k < max(self.retrieval_k, self.routing_k):
            raise ValueError("candidate_k must cover positive retrieval_k and routing_k")
        if self.dispersion_pairs <= 0 or self.min_region_size <= 0 or self.interaction_pairs < 0:
            raise ValueError("dispersion_pairs and min_region_size must be positive; interaction_pairs must be >= 0")
        if self.partition_mode not in {"combined", "query", "semantic", "random"}:
            raise ValueError("partition_mode must be combined, query, semantic, or random")
        if self.coaccess_weight not in {"count", "rank_idf"}:
            raise ValueError("coaccess_weight must be count or rank_idf")
        if self.probe_strategy not in {"active", "stratified"}:
            raise ValueError("probe_strategy must be active or stratified")
        if self.region_router_k < 0 or self.bridge_k < 0:
            raise ValueError("region_router_k and bridge_k must be >= 0")
        if not 0.0 <= self.boundary_replication <= 1.0:
            raise ValueError("boundary_replication must be in [0, 1]")
        if self.multistep_max_steps < 0:
            raise ValueError("multistep_max_steps must be >= 0")


class WARPG:
    """End-to-end WARP-G physical-design and retrieval pipeline."""

    def __init__(self, config: WARPConfig, graph_builder: GraphBuilder,
                 graph_retriever: GraphRetriever, reranker: Reranker,
                 base_retriever: HybridRetriever) -> None:
        self.config = config
        self.base = base_retriever
        self.graph_builder = graph_builder
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.selector = RegionSelector(config.seed)
        self.bundle: DatasetBundle | None = None
        self.coaccess: CoaccessGraph | None = None
        self.regions: list[Region] = []
        self.region_map: dict[str, Region] = {}
        self.doc_region: dict[str, str] = {}
        self.features: dict[str, RegionFeatures] = {}
        self.region_queries: dict[str, list[str]] = {}
        self.probes: dict[str, ProbeOutcome] = {}
        self.predicted_gains: dict[str, float] = {}
        self.graphs: dict[str, RegionalGraph] = {}
        self.full_graph: RegionalGraph | None = None
        self.costs: dict[str, float] = {}
        self.predictor: BenefitPredictor | None = None
        self.design_timings: dict[str, float] = {}
        self.interaction_analysis: dict[str, Any] = {}
        self.probe_labeling_online_cost: dict[str, Any] = {}
        self.doc_to_regions: dict[str, list[str]] = {}
        self.region_vectors: dict[str, list[float]] = {}
        self.region_neighbors: dict[str, dict[str, float]] = {}

    def fit(self, bundle: DatasetBundle) -> "WARPG":
        """完成基础索引、分区、特征、probe 和 benefit predictor 训练。"""
        if not bundle.documents or not bundle.train:
            raise ValueError("WARP-G requires a non-empty corpus and training/design queries")
        doc_ids = [document.id for document in bundle.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Corpus document IDs must be unique")
        if any(not document.content.strip() for document in bundle.documents):
            raise ValueError("Corpus documents must contain text")
        contents = [document.content for document in bundle.documents]
        if len(contents) != len(set(contents)):
            raise ValueError("HippoRAG document contents must be unique")
        known = set(doc_ids)
        for split_name, queries in (("train", bundle.train), ("dev", bundle.dev), ("test", bundle.test)):
            query_ids = [query.id for query in queries]
            if len(query_ids) != len(set(query_ids)):
                raise ValueError(f"{split_name} query IDs must be unique")
            missing = sorted({doc_id for query in queries for doc_id in query.gold_doc_ids if doc_id not in known})
            if missing:
                raise ValueError(f"{split_name} gold evidence is absent from corpus: {missing[:10]}")
        self.bundle = bundle
        # design.md 顺序：Base 索引 → partition → features → probe → predictor → selection。
        # Base 索引覆盖 100% corpus，是所有方法共享且不计为选择性 Graph 成本的底座。
        started = time.perf_counter()
        self.base.fit(bundle.documents)
        self.design_timings["base_index_seconds"] = time.perf_counter() - started
        # 下面所有 workload 信号只来自 bundle.train，避免 test leakage。
        started = time.perf_counter()
        hybrid_by_query = {
            query.id: self.base.search(query.text, self.config.candidate_k) for query in bundle.train
        }
        base_by_query = {
            query.id: fuse_and_rerank(
                query.text, [hybrid_by_query[query.id]], self.reranker,
                k=self.config.candidate_k, candidate_k=self.config.candidate_k, source="hybrid_reranked",
            )
            for query in bundle.train
        }
        self.coaccess = CoaccessGraphBuilder(
            self.config.coaccess_k, self.config.semantic_k, self.config.semantic_lambda,
            weight_mode=self.config.coaccess_weight,
        ).build(
            bundle.documents, bundle.train, self.base,
            query_top_ids={
                query.id: [result.doc_id for result in base_by_query[query.id]]
                for query in bundle.train
            },
        )
        partitioner = RegionPartitioner(
            resolution=self.config.partition_resolution, seed=self.config.seed,
            min_region_size=self.config.min_region_size,
        )
        cost_fn = lambda doc_ids, documents=bundle.documents: self.graph_builder.estimate_cost(
            Region("__cost__", list(doc_ids)), documents,
        )
        partition_kwargs = {"cost_fn": cost_fn, "max_cost": self.config.max_region_cost}
        if self.config.partition_mode == "combined":
            self.regions = partitioner.partition(self.coaccess, **partition_kwargs)
        elif self.config.partition_mode == "query":
            query_graph = CoaccessGraph(
                self.coaccess.nodes, dict(self.coaccess.query_edges), dict(self.coaccess.query_edges), {},
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(query_graph, **partition_kwargs)
        elif self.config.partition_mode == "semantic":
            semantic_edges = {key: self.config.semantic_lambda * value
                              for key, value in self.coaccess.semantic_edges.items()}
            semantic_graph = CoaccessGraph(
                self.coaccess.nodes, semantic_edges, {}, dict(self.coaccess.semantic_edges),
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(semantic_graph, **partition_kwargs)
        else:
            reference = partitioner.partition(self.coaccess, **partition_kwargs)
            shuffled = list(self.coaccess.nodes)
            random.Random(self.config.seed).shuffle(shuffled)
            sizes = [len(region.metadata.get("core_doc_ids") or region.doc_ids) for region in reference]
            offset = 0
            self.regions = []
            for index, size in enumerate(sizes):
                ids = sorted(shuffled[offset:offset + size])
                self.regions.append(Region(
                    f"r{index:04d}", ids, metadata={"core_doc_ids": ids, "boundary_doc_ids": []},
                ))
                offset += size
        self.regions = replicate_boundaries(self.regions, self.coaccess, self.config.boundary_replication)
        self.design_timings["partition_seconds"] = time.perf_counter() - started
        self.region_map = {region.id: region for region in self.regions}
        self.doc_region = {
            doc_id: region.id
            for region in self.regions
            for doc_id in (region.metadata.get("core_doc_ids") or region.doc_ids)
        }
        self.doc_to_regions = {}
        for region in self.regions:
            for doc_id in region.doc_ids:
                self.doc_to_regions.setdefault(doc_id, []).append(region.id)
        self.region_neighbors = region_coaccess_neighbors(self.regions, self.coaccess)
        self.region_vectors = self._fit_region_router()
        started = time.perf_counter()
        extractor = RegionFeatureExtractor(
            self.config.routing_k, self.config.candidate_k, self.config.retrieval_k,
            self.config.dispersion_pairs, self.config.seed,
        )
        self.features, self.region_queries, _ = extractor.extract(
            self.regions, bundle.documents, bundle.train, self.base, self.coaccess,
            query_results=base_by_query,
        )
        self.design_timings["feature_seconds"] = time.perf_counter() - started
        # 选择前只能使用 token proxy；actual cost 必须等真实 build 后才可观测。
        self.costs = {region.id: self.graph_builder.estimate_cost(region, bundle.documents)
                      for region in self.regions}
        prober = RegionProber(
            self.graph_builder, self.graph_retriever, self.reranker,
            self.config.probe_fraction, self.config.retrieval_k, self.config.candidate_k,
            self.config.benefit_objective, self.config.seed,
        )
        eligible_count = sum(1 for region in self.regions if self.features[region.id].query_freq > 0)
        if eligible_count < 6:
            raise ValueError("Scientific benefit prediction requires at least six workload-covered regions")
        probe_total = min(eligible_count, max(6, round(eligible_count * self.config.probe_fraction)))
        labeling_before = self.graph_retriever.stats()
        if self.config.probe_strategy == "active" and probe_total >= 8:
            round1 = max(4, probe_total // 2)
            first = prober.select_probe_regions(
                self.regions, self.features, self.costs, count=round1, strategy="active",
            )
            self.probes = prober.run(first, bundle.documents, bundle.train,
                                     self.region_queries, hybrid_by_query)
            gains = {region_id: outcome.gain for region_id, outcome in self.probes.items()}
            interim = BenefitPredictor(self.config.seed).fit(self.features, gains)
            predicted = interim.predict(self.features)
            second = prober.select_probe_regions(
                self.regions, self.features, self.costs, count=probe_total - round1,
                predicted_gains=predicted, already_probed=set(self.probes), strategy="active",
            )
            more = prober.run(second, bundle.documents, bundle.train,
                              self.region_queries, hybrid_by_query)
            self.probes.update(more)
        else:
            probe_regions = prober.select_probe_regions(
                self.regions, self.features, self.costs, count=probe_total,
                strategy=self.config.probe_strategy,
            )
            self.probes = prober.run(probe_regions, bundle.documents, bundle.train,
                                     self.region_queries, hybrid_by_query)
        self.probe_labeling_online_cost = self.graph_retriever.delta(labeling_before)
        self.graphs.update({region_id: outcome.graph for region_id, outcome in self.probes.items()})
        gains = {region_id: outcome.gain for region_id, outcome in self.probes.items()}
        started = time.perf_counter()
        self.predictor = BenefitPredictor(self.config.seed).fit(self.features, gains)
        self.predicted_gains = self.predictor.predict(self.features)
        self.predicted_gains.update(gains)
        self.design_timings["predictor_seconds"] = time.perf_counter() - started
        if self.config.interaction_pairs == 0:
            self.interaction_analysis = {
                "pairs": [],
                "mean_absolute_interaction": 0.0,
                "skipped": True,
            }
        else:
            self.interaction_analysis = self._probe_interactions()
        return self

    def routing_diagnostics(self, queries: list[Any]) -> dict[str, float]:
        """测量 Base-hit 与独立 region router 能否命中 gold 所在 Region。"""
        eligible = [query for query in queries if query.gold_doc_ids]
        if not eligible:
            raise ValueError("Routing diagnostics require gold evidence")
        base_complete = base_any = combined_complete = combined_any = 0.0
        for query in eligible:
            gold_regions = {
                self.doc_region[doc_id] for doc_id in query.gold_doc_ids if doc_id in self.doc_region
            }
            ranked = self.rank_base(query.text, self.config.routing_k)
            base_hit = self._base_hit_regions(ranked, set(self.region_map))
            routed = self._route_regions(query.text, ranked, set(self.region_map))
            base_complete += float(gold_regions.issubset(set(base_hit)))
            base_any += float(bool(gold_regions & set(base_hit)))
            combined_complete += float(gold_regions.issubset(set(routed["routed_regions"])))
            combined_any += float(bool(gold_regions & set(routed["routed_regions"])))
        n = len(eligible)
        return {
            "queries": float(n),
            "any_gold_region_recall": base_any / n,
            "complete_gold_region_recall": base_complete / n,
            "router_any_gold_region_recall": combined_any / n,
            "router_complete_gold_region_recall": combined_complete / n,
        }

    def _probe_interactions(self) -> dict[str, Any]:
        """直接测量区域图二阶交互，检验独立 gain 假设。"""
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        region_ids = sorted(self.probes)
        pairs = [(left, right) for index, left in enumerate(region_ids) for right in region_ids[index + 1:]]
        if len(pairs) > self.config.interaction_pairs:
            pairs = random.Random(self.config.seed).sample(pairs, self.config.interaction_pairs)
        query_map = {query.id: query for query in self.bundle.train}
        before = self.graph_retriever.stats()
        rows: list[dict[str, float | str | int]] = []
        for left, right in pairs:
            query_ids = sorted(set(self.region_queries[left]) | set(self.region_queries[right]))
            base_values, left_values, right_values, joint_values = [], [], [], []
            for query_id in query_ids:
                query = query_map[query_id]
                metric = complete_evidence if self.config.benefit_objective == "complete_evidence" else evidence_recall
                base_values.append(metric(self.search(query.text, self.config.retrieval_k, set()), query.gold_doc_ids))
                left_values.append(metric(self.search(query.text, self.config.retrieval_k, {left}), query.gold_doc_ids))
                right_values.append(metric(self.search(query.text, self.config.retrieval_k, {right}), query.gold_doc_ids))
                joint_values.append(metric(self.search(query.text, self.config.retrieval_k, {left, right}), query.gold_doc_ids))
            base_mean = sum(base_values) / len(base_values)
            left_gain = sum(left_values) / len(left_values) - base_mean
            right_gain = sum(right_values) / len(right_values) - base_mean
            joint_gain = sum(joint_values) / len(joint_values) - base_mean
            rows.append({
                "left_region": left, "right_region": right, "queries": len(query_ids),
                "left_gain": left_gain, "right_gain": right_gain, "joint_gain": joint_gain,
                "interaction": joint_gain - left_gain - right_gain,
            })
        interactions = [abs(float(row["interaction"])) for row in rows]
        return {
            "pairs": rows,
            "mean_absolute_interaction": sum(interactions) / len(interactions) if interactions else 0.0,
            "online_analysis_cost": self.graph_retriever.delta(before),
        }

    @property
    def full_graph_cost(self) -> float:
        """返回所有区域 token proxy 之和，仅作成本对照分母，不再用于截断。"""
        return sum(self.costs.values())

    def select(self, method: str = "warp") -> list[str]:
        """WARP 按 score_i 选择；controls 只换排序公式，并取相同区域数。"""
        warp_selected = self.selector.select(
            "warp", self.features, self.costs, self.predicted_gains,
            region_queries=self.region_queries,
        )
        if method == "warp":
            return warp_selected
        return self.selector.select(
            method, self.features, self.costs, self.predicted_gains, limit=len(warp_selected),
            region_queries=self.region_queries,
        )

    def rank_base(self, query: str, k: int | None = None) -> list[SearchResult]:
        """统一 Base：BM25 + NV-Embed → RRF → pinned CrossEncoder → top-k。"""
        k = self.config.retrieval_k if k is None else k
        return fuse_and_rerank(
            query, [self.base.search(query, self.config.candidate_k)], self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="hybrid_reranked",
        )

    def search_base(self, query: str, k: int | None = None, method: str = "hybrid") -> list[SearchResult]:
        """所有 Base baseline 也走与图方法相同的 candidate depth 和 CrossEncoder。"""
        k = self.config.retrieval_k if k is None else k
        if method == "hybrid":
            return self.rank_base(query, k)
        retriever = {"bm25": self.base.bm25, "dense": self.base.dense}.get(method)
        if retriever is None:
            raise ValueError(f"Unknown base method: {method}")
        candidates = retriever.search(query, self.config.candidate_k)
        return fuse_and_rerank(
            query, [candidates], self.reranker, k=k,
            candidate_k=self.config.candidate_k, source=f"{method}_reranked",
        )

    def materialize(self, region_ids: list[str]) -> None:
        """按需构建尚未缓存的区域图；probe 图会被安全复用。"""
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        pending = [region_id for region_id in region_ids if region_id not in self.graphs]
        print(json.dumps({
            "materialize": "start",
            "requested": len(region_ids),
            "cached": len(region_ids) - len(pending),
            "pending": pending,
        }, ensure_ascii=False), flush=True)
        for index, region_id in enumerate(pending, 1):
            print(json.dumps({
                "materialize": "build",
                "region_id": region_id,
                "index": index,
                "total": len(pending),
                "num_docs": len(self.region_map[region_id].doc_ids),
            }, ensure_ascii=False), flush=True)
            self.graphs[region_id] = self.graph_builder.build(self.region_map[region_id], self.bundle.documents)
            print(json.dumps({
                "materialize": "saved",
                "region_id": region_id,
                "index": index,
                "total": len(pending),
            }, ensure_ascii=False), flush=True)

    def search(self, query: str, k: int | None = None, selected_regions: set[str] | None = None) -> list[SearchResult]:
        """Base + region router + cheap bridge，只查询已物化区域图。"""
        return self.search_with_trace(query, k, selected_regions)[0]

    def search_with_trace(
        self, query: str, k: int | None = None, selected_regions: set[str] | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        """返回检索结果与路由轨迹，供多步 log 使用。"""
        k = self.config.retrieval_k if k is None else k
        base_candidates = self.base.search(query, self.config.candidate_k)
        base_ranked = fuse_and_rerank(
            query, [base_candidates], self.reranker,
            k=max(self.config.routing_k, k), candidate_k=self.config.candidate_k,
            source="hybrid_reranked",
        )
        if selected_regions is None:
            selected_regions = set(self.graphs)
        trace = self._route_regions(query, base_ranked, selected_regions)
        graph_results = []
        for region_id in trace["routed_regions"]:
            if region_id not in self.graphs:
                continue
            hits = self.graph_retriever.search(query, self.graphs[region_id], self.config.candidate_k)
            for item in hits:
                item.region_id = region_id
            graph_results.append(hits)
        fused = fuse_and_rerank(
            query, [base_candidates] + graph_results, self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="warp",
        )
        return fused, trace

    def _fit_region_router(self) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        for region in self.regions:
            core = region.metadata.get("core_doc_ids") or region.doc_ids
            if not core:
                continue
            rows = [self.base.dense.vector(doc_id) for doc_id in core]
            dim = len(rows[0])
            vectors[region.id] = [sum(row[index] for row in rows) / len(rows) for index in range(dim)]
        return vectors

    def _base_hit_regions(self, ranked: list[SearchResult], selected_regions: set[str]) -> list[str]:
        routed: list[str] = []
        for result in ranked[:self.config.routing_k]:
            for region_id in self.doc_to_regions.get(result.doc_id, []):
                if region_id in selected_regions and region_id not in routed:
                    routed.append(region_id)
        return routed

    def _route_regions(
        self, query: str, ranked: list[SearchResult], selected_regions: set[str],
    ) -> dict[str, list[str]]:
        base_hit = self._base_hit_regions(ranked, selected_regions)
        router_regions: list[str] = []
        if self.config.region_router_k > 0 and self.region_vectors:
            query_vector = self.base.dense.encode([query])[0]
            scored = sorted(
                (
                    (region_id, cosine(query_vector, vector))
                    for region_id, vector in self.region_vectors.items()
                    if region_id in selected_regions
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for region_id, _score in scored[:self.config.region_router_k]:
                if region_id not in router_regions:
                    router_regions.append(region_id)
        seeded = list(dict.fromkeys(base_hit + router_regions))
        bridge_regions: list[str] = []
        if self.config.bridge_k > 0:
            weights: dict[str, float] = {}
            for region_id in seeded:
                for other, weight in self.region_neighbors.get(region_id, {}).items():
                    if other in selected_regions and other not in seeded:
                        weights[other] = weights.get(other, 0.0) + weight
            for region_id, _weight in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:self.config.bridge_k]:
                bridge_regions.append(region_id)
        routed = list(dict.fromkeys(seeded + bridge_regions))
        return {
            "base_hit_regions": base_hit,
            "router_regions": router_regions,
            "bridge_regions": bridge_regions,
            "routed_regions": routed,
        }

    def generate_next_query(self, prompt: str) -> str:
        """多步检索的 LLM 扩展；无共享 LLM 时返回空字符串以停止。"""
        llm = getattr(self.graph_builder, "_shared_llm", None)
        if llm is None or not hasattr(llm, "infer"):
            return "END"
        result = llm.infer(prompt)
        if not isinstance(result, tuple) or not result:
            return "END"
        text = result[0]
        return str(text or "").strip()

    def materialize_full_graph(self) -> RegionalGraph:
        """Build one corpus-wide graph for the true Full Graph baseline.

        This is intentionally not represented as a union of independent regional
        graphs: the baseline must retain cross-region facts and synonymy edges.
        """
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        if self.full_graph is None:
            region = Region("__full_corpus__", [doc.id for doc in self.bundle.documents])
            self.full_graph = self.graph_builder.build_full_graph(region, self.bundle.documents)
        return self.full_graph

    def search_full_graph(self, query: str, k: int | None = None) -> list[SearchResult]:
        """融合 Base 与真正 corpus-wide 单图结果，并使用相同 reranker。"""
        k = self.config.retrieval_k if k is None else k
        graph = self.materialize_full_graph()
        base_results = self.base.search(query, self.config.candidate_k)
        graph_results = self.graph_retriever.search(query, graph, self.config.candidate_k)
        return fuse_and_rerank(
            query, [base_results, graph_results], self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="full_graph",
        )

    def evaluate_full_graph(self, queries: list[Any], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测 Base + Full Graph fusion。"""
        self.materialize_full_graph()
        return evaluate_retrieval(
            queries, self.search_full_graph, ks or self.config.retrieval_ks,
            bootstrap_seed=self.config.seed,
        )

    def evaluate_full_graph_only(self, queries: list[Any], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测官方图检索通路；仍使用共享 CrossEncoder。"""
        graph = self.materialize_full_graph()
        return evaluate_retrieval(
            queries, lambda query, k: fuse_and_rerank(
                query, [self.graph_retriever.search(query, graph, self.config.candidate_k)],
                self.reranker, k=k, candidate_k=self.config.candidate_k, source="hipporag2_reranked",
            ), ks or self.config.retrieval_ks, bootstrap_seed=self.config.seed,
        )

    def evaluate(self, queries: list[Any], selected_regions: list[str], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """物化给定区域并评测 Selective Regional Graph 系统。"""
        self.materialize(selected_regions)
        selected = set(selected_regions)
        return evaluate_retrieval(
            queries, lambda query, k: self.search(query, k, selected),
            ks or self.config.retrieval_ks, bootstrap_seed=self.config.seed,
        )

    def evaluate_base(self, queries: list[Any], method: str, ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测 BM25、Dense 或 Hybrid 基础检索。"""
        normalized = "hybrid" if method.lower() == "base" else method.lower()
        return evaluate_retrieval(
            queries, lambda query, k: self.search_base(query, k, normalized),
            ks or self.config.retrieval_ks, bootstrap_seed=self.config.seed,
        )

    def report(self) -> dict[str, Any]:
        """导出物理设计、probe 成本、特征和预测结果供 artifact 审计。"""
        probe_cost = aggregate_costs([outcome.graph.cost for outcome in self.probes.values()])
        observed = sorted(outcome.gain for outcome in self.probes.values())
        nonnegative = sorted(max(value, 0.0) for value in observed)
        total = sum(nonnegative)
        gini = (sum((2 * index - len(nonnegative) - 1) * value
                    for index, value in enumerate(nonnegative, 1))
                / (len(nonnegative) * total)) if total > 0 else 0.0
        return {
            "config": asdict(self.config),
            "num_documents": len(self.bundle.documents) if self.bundle else 0,
            "num_train_queries": len(self.bundle.train) if self.bundle else 0,
            "num_regions": len(self.regions),
            "probe_regions": sorted(self.probes),
            "probe_cost": probe_cost.to_dict(),
            "observed_gain_distribution": {
                "minimum": observed[0],
                "median": statistics.median(observed),
                "maximum": observed[-1],
                "mean": sum(observed) / len(observed),
                "positive_fraction": sum(value > 0 for value in observed) / len(observed),
                "nonnegative_gain_gini": gini,
            },
            "non_graph_design_wall_seconds": self.design_timings,
            "full_graph_estimated_cost": self.full_graph_cost,
            "predictor": type(self.predictor).__name__,
            "graph_implementation": type(self.graph_builder).__name__,
            "graph_retriever": type(self.graph_retriever).__name__,
            "feature_importance": self.predictor.feature_importance(),
            "predictor_validation": BenefitPredictor.cross_validate(self.features, {
                region_id: outcome.gain for region_id, outcome in self.probes.items()
            }, self.config.seed),
            "feature_ablation": BenefitPredictor.feature_ablation(self.features, {
                region_id: outcome.gain for region_id, outcome in self.probes.items()
            }, self.config.seed),
            "probe_learning_curve": BenefitPredictor.learning_curve(self.features, {
                region_id: outcome.gain for region_id, outcome in self.probes.items()
            }, self.config.seed),
            "probe_interactions": self.interaction_analysis,
            "regions": [{
                **self.features[region.id].to_dict(),
                "doc_ids": region.doc_ids,
                "estimated_cost": self.costs[region.id],
                "predicted_gain": self.predicted_gains.get(region.id),
                "probe_gain": self.probes[region.id].gain if region.id in self.probes else None,
                "probe_recall_gain": self.probes[region.id].recall_gain if region.id in self.probes else None,
                "probe_complete_gain": self.probes[region.id].complete_gain if region.id in self.probes else None,
            } for region in self.regions],
        }
