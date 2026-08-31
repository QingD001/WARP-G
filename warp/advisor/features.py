"""在构图前计算 region 的九维 workload/corpus 特征。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import combinations

from warp.models import Document, Query, Region, RegionFeatures, SearchResult
from warp.partition.coaccess_graph import CoaccessGraph
from warp.retrieval.hybrid import HybridRetriever
from warp.utils import cosine, tokenize


class RegionFeatureExtractor:
    """只依赖 corpus、train query、基础检索和便宜共访问图。"""
    def __init__(self, routing_k: int = 20, candidate_k: int = 50, retrieval_k: int = 10,
                 dispersion_pairs: int = 4096, seed: int = 42) -> None:
        if candidate_k < routing_k or candidate_k < retrieval_k:
            raise ValueError("candidate_k must cover routing_k and retrieval_k")
        self.routing_k = routing_k
        self.candidate_k = candidate_k
        self.retrieval_k = retrieval_k
        self.dispersion_pairs = dispersion_pairs
        self.seed = seed

    def extract(
        self,
        regions: list[Region],
        documents: list[Document],
        queries: list[Query],
        retriever: HybridRetriever,
        coaccess: CoaccessGraph,
    ) -> tuple[dict[str, RegionFeatures], dict[str, list[str]], dict[str, list[SearchResult]]]:
        """返回特征、region-query 路由关系和可复用的 base results。"""
        doc_map = {doc.id: doc for doc in documents}
        doc_region = {doc_id: region.id for region in regions for doc_id in region.doc_ids}
        query_results: dict[str, list[SearchResult]] = {
            query.id: retriever.search(query.text, self.candidate_k) for query in queries
        }
        # 一个 query 可访问多个 region，但在同一区域只计一次 query_freq。
        region_queries: dict[str, list[str]] = defaultdict(list)
        for query in queries:
            seen: set[str] = set()
            for result in query_results[query.id][:self.routing_k]:
                region_id = doc_region[result.doc_id]
                if region_id not in seen:
                    region_queries[region_id].append(query.id)
                    seen.add(region_id)

        query_map = {query.id: query for query in queries}
        features: dict[str, RegionFeatures] = {}
        for region in regions:
            qids = region_queries.get(region.id, [])
            recalls: list[float] = []
            failures: list[float] = []
            entropies: list[float] = []
            multi_docs: list[float] = []
            region_docs = set(region.doc_ids)
            for qid in qids:
                query = query_map[qid]
                # recall/failure 对齐 probe 标签与最终指标的深度 (retrieval_k)。
                ranked = query_results[qid][:self.retrieval_k]
                retrieved = {result.doc_id for result in ranked}
                gold = set(query.gold_doc_ids)
                region_gold = gold & region_docs
                if gold:
                    recalls.append(len(gold & retrieved) / len(gold))
                    failures.append(float(not gold.issubset(retrieved)))
                scores = [max(result.score, 0.0) for result in ranked if result.doc_id in region_docs]
                total = sum(scores)
                if total > 0 and len(scores) > 1:
                    probabilities = [score / total for score in scores]
                    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities) / math.log(len(scores))
                    entropies.append(entropy)
                if region_gold:
                    multi_docs.append(float(len(region_gold) > 1 or len(gold) > 1))

            # dispersion 与 density 分别刻画语义异质性和真实 workload 内聚性。
            pair_count = len(region.doc_ids) * (len(region.doc_ids) - 1) // 2
            if pair_count <= self.dispersion_pairs:
                all_pairs = list(combinations(region.doc_ids, 2))
            else:
                rng = random.Random(f"{self.seed}:{region.id}")
                sampled_indices: set[tuple[int, int]] = set()
                while len(sampled_indices) < self.dispersion_pairs:
                    left, right = rng.sample(range(len(region.doc_ids)), 2)
                    sampled_indices.add((min(left, right), max(left, right)))
                all_pairs = [(region.doc_ids[left], region.doc_ids[right])
                             for left, right in sorted(sampled_indices)]
            distances = [1.0 - cosine(retriever.dense.vector(left), retriever.dense.vector(right))
                         for left, right in all_pairs]
            possible = len(region.doc_ids) * (len(region.doc_ids) - 1) / 2
            internal_weight = sum(weight for (left, right), weight in coaccess.query_edges.items()
                                  if left in region_docs and right in region_docs)
            query_normalizer = max(len(queries), 1)
            density = internal_weight / max(possible * query_normalizer, 1.0)
            features[region.id] = RegionFeatures(
                region_id=region.id,
                num_docs=float(len(region.doc_ids)),
                num_tokens=float(sum(len(tokenize(doc_map[doc_id].content)) for doc_id in region.doc_ids)),
                query_freq=float(len(qids)),
                base_recall=sum(recalls) / len(recalls) if recalls else 0.0,
                failure_rate=sum(failures) / len(failures) if failures else 0.0,
                avg_retrieval_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
                multi_doc_rate=sum(multi_docs) / len(multi_docs) if multi_docs else 0.0,
                embedding_dispersion=sum(distances) / len(distances) if distances else 0.0,
                coaccess_density=density,
            )
        return features, dict(region_queries), query_results
