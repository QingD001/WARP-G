"""文档级 Evidence Recall、Complete Evidence 与 query-level 统计。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from warp.models import Query, SearchResult
from .statistics import paired_bootstrap_interval


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
    queries: list[Query], search: Callable[[str, int], list[SearchResult]], ks: tuple[int, ...] = (5, 10),
    *, bootstrap_samples: int = 10_000, bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Retrieval cutoffs must be positive")
    eligible = [query for query in queries if query.gold_doc_ids]
    if not eligible:
        raise ValueError("Retrieval evaluation requires queries with gold evidence")
    per_query: dict[str, dict[str, float]] = {}
    for query in eligible:
        per_query[query.id] = query_metrics(search(query.text, max(ks)), query, ks)
    metric_names = list(next(iter(per_query.values())))
    output: dict[str, Any] = {"num_queries": len(eligible), "per_query": per_query}
    for metric in metric_names:
        values = [per_query[query.id][metric] for query in eligible]
        lower, upper = paired_bootstrap_interval(values, samples=bootstrap_samples, seed=bootstrap_seed)
        output[metric] = sum(values) / len(values)
        output[f"{metric}_ci95"] = [lower, upper]
    return output
