"""WARP-G 的训练/设计阶段与测试阶段编排器。

`fit` 是唯一允许读取 train/design query 的设计入口；评测函数只接收显式传入的
dev/test query。图 builder、图 retriever、base retriever 和 reranker 均可注入，
因此 advisor 算法不绑定具体 GraphRAG 实现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import random
import statistics
from typing import Any

from warp.advisor.features import RegionFeatureExtractor
from warp.advisor.predictor import BenefitPredictor
from warp.advisor.probe import ProbeOutcome, RegionProber, complete_evidence, evidence_recall
from warp.advisor.selector import BudgetSelector
from warp.eval.construction_cost import aggregate_costs
from warp.eval.retrieval import evaluate_retrieval
from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import DatasetBundle, Region, RegionFeatures, SearchResult
from warp.partition.coaccess_graph import CoaccessGraph, CoaccessGraphBuilder
from warp.partition.leiden import RegionPartitioner
from warp.retrieval.hybrid import HybridRetriever, fuse_and_rerank
from warp.retrieval.reranker import Reranker


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

    def __post_init__(self) -> None:
        if self.retrieval_k <= 0 or self.routing_k <= 0 or self.candidate_k < max(self.retrieval_k, self.routing_k):
            raise ValueError("candidate_k must cover positive retrieval_k and routing_k")
        if self.dispersion_pairs <= 0 or self.min_region_size <= 0 or self.interaction_pairs <= 0:
            raise ValueError("dispersion_pairs, interaction_pairs and min_region_size must be positive")
        if self.partition_mode not in {"combined", "query", "semantic", "random"}:
            raise ValueError("partition_mode must be combined, query, semantic, or random")


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
        self.selector = BudgetSelector(config.seed)
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
        # Base 索引覆盖 100% corpus，是所有方法共享且不计为选择性 Graph 成本的底座。
        started = time.perf_counter()
        self.base.fit(bundle.documents)
        self.design_timings["base_index_seconds"] = time.perf_counter() - started
        # 下面所有 workload 信号只来自 bundle.train，避免 test leakage。
        started = time.perf_counter()
        self.coaccess = CoaccessGraphBuilder(
            self.config.coaccess_k, self.config.semantic_k, self.config.semantic_lambda,
        ).build(bundle.documents, bundle.train, self.base)
        partitioner = RegionPartitioner(
            resolution=self.config.partition_resolution, seed=self.config.seed,
            min_region_size=self.config.min_region_size,
        )
        if self.config.partition_mode == "combined":
            self.regions = partitioner.partition(self.coaccess)
        elif self.config.partition_mode == "query":
            query_graph = CoaccessGraph(
                self.coaccess.nodes, dict(self.coaccess.query_edges), dict(self.coaccess.query_edges), {},
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(query_graph)
        elif self.config.partition_mode == "semantic":
            semantic_edges = {key: self.config.semantic_lambda * value
                              for key, value in self.coaccess.semantic_edges.items()}
            semantic_graph = CoaccessGraph(
                self.coaccess.nodes, semantic_edges, {}, dict(self.coaccess.semantic_edges),
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(semantic_graph)
        else:
            reference = partitioner.partition(self.coaccess)
            shuffled = list(self.coaccess.nodes)
            random.Random(self.config.seed).shuffle(shuffled)
            sizes = [len(region.doc_ids) for region in reference]
            offset = 0
            self.regions = []
            for index, size in enumerate(sizes):
                self.regions.append(Region(f"r{index:04d}", sorted(shuffled[offset:offset + size])))
                offset += size
        self.design_timings["partition_seconds"] = time.perf_counter() - started
        self.region_map = {region.id: region for region in self.regions}
        self.doc_region = {doc_id: region.id for region in self.regions for doc_id in region.doc_ids}
        started = time.perf_counter()
        extractor = RegionFeatureExtractor(
            self.config.routing_k, self.config.candidate_k, self.config.retrieval_k,
            self.config.dispersion_pairs, self.config.seed,
        )
        self.features, self.region_queries, base_results = extractor.extract(
            self.regions, bundle.documents, bundle.train, self.base, self.coaccess,
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
        probe_regions = prober.select_probe_regions(self.regions, self.features, self.costs)
        labeling_before = self.graph_retriever.stats()
        self.probes = prober.run(probe_regions, bundle.documents, bundle.train,
                                 self.region_queries, base_results)
        # probe 打标阶段的在线图检索 token 单独入账，供 design-search 口径完整报告。
        self.probe_labeling_online_cost = self.graph_retriever.delta(labeling_before)
        self.graphs.update({region_id: outcome.graph for region_id, outcome in self.probes.items()})
        gains = {region_id: outcome.gain for region_id, outcome in self.probes.items()}
        started = time.perf_counter()
        self.predictor = BenefitPredictor(self.config.seed).fit(self.features, gains)
        self.predicted_gains = self.predictor.predict(self.features)
        # A measured probe label is more reliable than its fitted value.
        self.predicted_gains.update(gains)
        self.design_timings["predictor_seconds"] = time.perf_counter() - started
        self.interaction_analysis = self._probe_interactions()
        return self

    def routing_diagnostics(self, queries: list[Any]) -> dict[str, float]:
        """测量 Base router 能否命中 gold evidence 所在 Region。"""
        eligible = [query for query in queries if query.gold_doc_ids]
        if not eligible:
            raise ValueError("Routing diagnostics require gold evidence")
        complete = any_hit = 0.0
        for query in eligible:
            gold_regions = {self.doc_region[doc_id] for doc_id in query.gold_doc_ids}
            routed = {self.doc_region[result.doc_id]
                      for result in self.base.search(query.text, self.config.routing_k)}
            complete += float(gold_regions.issubset(routed))
            any_hit += float(bool(gold_regions & routed))
        return {
            "queries": float(len(eligible)),
            "any_gold_region_recall": any_hit / len(eligible),
            "complete_gold_region_recall": complete / len(eligible),
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
        """返回所有区域 token proxy 之和，作为选择时预算分母。"""
        return sum(self.costs.values())

    def select(self, budget_fraction: float, method: str = "warp") -> list[str]:
        """把相对预算转换为绝对 proxy cost，并调用统一 BudgetSelector。"""
        if not 0.0 <= budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be between 0 and 1")
        return self.selector.select(
            method, budget_fraction * self.full_graph_cost, self.features, self.costs, self.predicted_gains,
        )

    def search_base(self, query: str, k: int | None = None, method: str = "hybrid") -> list[SearchResult]:
        """所有 Base baseline 也走与图方法相同的 candidate depth 和 CrossEncoder。"""
        k = self.config.retrieval_k if k is None else k
        retriever = {"bm25": self.base.bm25, "dense": self.base.dense, "hybrid": self.base}.get(method)
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
        for region_id in region_ids:
            if region_id not in self.graphs:
                self.graphs[region_id] = self.graph_builder.build(self.region_map[region_id], self.bundle.documents)

    def search(self, query: str, k: int | None = None, selected_regions: set[str] | None = None) -> list[SearchResult]:
        """Base 路由到已物化区域，融合对应图结果后统一 rerank。"""
        k = self.config.retrieval_k if k is None else k
        base_results = self.base.search(query, self.config.candidate_k)
        if selected_regions is None:
            selected_regions = set(self.graphs)
        # 路由是确定性的文档归属查表，不使用 LLM/agent 做检索决策。
        routed: list[str] = []
        for result in base_results[:self.config.routing_k]:
            region_id = self.doc_region[result.doc_id]
            if region_id in selected_regions and region_id not in routed:
                routed.append(region_id)
        graph_results = [self.graph_retriever.search(query, self.graphs[region_id], self.config.candidate_k)
                         for region_id in routed if region_id in self.graphs]
        return fuse_and_rerank(
            query, [base_results] + graph_results, self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="warp",
        )

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

    def evaluate_full_graph(self, queries: list[Any], ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
        """评测 Base + Full Graph fusion。"""
        self.materialize_full_graph()
        return evaluate_retrieval(queries, self.search_full_graph, ks, bootstrap_seed=self.config.seed)

    def evaluate_full_graph_only(self, queries: list[Any], ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
        """评测官方图检索通路；仍使用共享 CrossEncoder。"""
        graph = self.materialize_full_graph()
        return evaluate_retrieval(
            queries, lambda query, k: fuse_and_rerank(
                query, [self.graph_retriever.search(query, graph, self.config.candidate_k)], self.reranker,
                k=k, candidate_k=self.config.candidate_k, source="hipporag2_reranked",
            ), ks, bootstrap_seed=self.config.seed,
        )

    def evaluate(self, queries: list[Any], selected_regions: list[str], ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
        """物化给定区域并评测 Selective Regional Graph 系统。"""
        self.materialize(selected_regions)
        selected = set(selected_regions)
        return evaluate_retrieval(
            queries, lambda query, k: self.search(query, k, selected), ks,
            bootstrap_seed=self.config.seed,
        )

    def evaluate_base(self, queries: list[Any], method: str, ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
        """评测 BM25、Dense 或 Hybrid 基础检索。"""
        normalized = "hybrid" if method.lower() == "base" else method.lower()
        return evaluate_retrieval(
            queries, lambda query, k: self.search_base(query, k, normalized), ks,
            bootstrap_seed=self.config.seed,
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
