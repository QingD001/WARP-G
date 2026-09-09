"""文档级 Evidence Recall、Complete Evidence 与 query-level 统计。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from warp.models import Query, SearchResult
from .cutoffs import RETRIEVAL_KS
from .statistics import paired_bootstrap_interval


def serialize_ranked(results: list[SearchResult]) -> list[dict[str, Any]]:
    """把检索结果写成 JSON 可落盘、可供阅读器复用的字典。"""
    return [
        {
            "doc_id": result.doc_id,
            "score": result.score,
            "source": result.source,
            "rank": result.rank,
            "region_id": result.region_id,
        }
        for result in results
    ]


def ranked_from_payload(rows: list[dict[str, Any]] | None) -> list[SearchResult]:
    """从评测缓存恢复 SearchResult，阅读器不得再调 search。"""
    return [
        SearchResult(
            str(row["doc_id"]), float(row["score"]), str(row.get("source") or "cached"),
            int(row.get("rank") or 0), row.get("region_id"),
        )
        for row in rows or []
    ]


def query_metrics(results: list[SearchResult], query: Query, ks: tuple[int, ...]) -> dict[str, float]:
    gold = set(query.gold_doc_ids)
    if not gold:
        raise ValueError(f"Query {query.id!r} has no gold evidence")
    values: dict[str, float] = {}
    for k in ks:
        found = {result.doc_id for result in results[:k]}
        values[f"evidence_recall@{k}"] = len(gold & found) / len(gold)
        values[f"complete_evidence@{k}"] = float(gold.issubset(found))
    return values


def evaluate_retrieval(
    queries: list[Query], search: Callable[[str, int], list[SearchResult]],
    ks: tuple[int, ...] = RETRIEVAL_KS,
    *, bootstrap_samples: int = 10_000, bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """一次检索同时计算多档指标，并把 ranked_results 交给阅读器复用。"""
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Retrieval cutoffs must be positive")
    eligible = [query for query in queries if query.gold_doc_ids]
    if not eligible:
        raise ValueError("Retrieval evaluation requires queries with gold evidence")
    depth = max(ks)
    per_query: dict[str, dict[str, float]] = {}
    ranked_results: dict[str, list[dict[str, Any]]] = {}
    for query in eligible:
        results = search(query.text, depth)
        per_query[query.id] = query_metrics(results, query, ks)
        ranked_results[query.id] = serialize_ranked(results[:depth])
    metric_names = list(next(iter(per_query.values())))
    output: dict[str, Any] = {
        "num_queries": len(eligible),
        "per_query": per_query,
        "ranked_results": ranked_results,
        "retrieval_ks": list(ks),
    }
    for metric in metric_names:
        values = [per_query[query.id][metric] for query in eligible]
        lower, upper = paired_bootstrap_interval(values, samples=bootstrap_samples, seed=bootstrap_seed)
        output[metric] = sum(values) / len(values)
        output[f"{metric}_ci95"] = [lower, upper]
    return output
