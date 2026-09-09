"""IRCoT 风格多步检索：每步检索 → 推理扩展 → 再检索，并落盘全量 step log。"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from warp.eval.cutoffs import RETRIEVAL_KS
from warp.eval.retrieval import query_metrics, serialize_ranked
from warp.models import Query, SearchResult
from warp.utils import write_json


IRCOT_PROMPT = """You are performing multi-hop retrieval.
Question: {question}
Documents retrieved so far:
{documents}

If the documents are sufficient to answer the question, reply with exactly END.
Otherwise reply with one short search query that would retrieve the missing evidence.
Do not answer the question. Reply with END or a search query only.
"""


def merge_ranked(existing: list[SearchResult], incoming: list[SearchResult]) -> list[SearchResult]:
    """按 doc_id 去重，保留更高分，重写 rank。"""
    best: dict[str, SearchResult] = {}
    for result in existing + incoming:
        previous = best.get(result.doc_id)
        if previous is None or result.score > previous.score:
            best[result.doc_id] = result
    ordered = sorted(best.values(), key=lambda item: (-item.score, item.doc_id))
    return [
        SearchResult(item.doc_id, item.score, item.source, rank + 1, item.region_id)
        for rank, item in enumerate(ordered)
    ]


def run_multistep_retrieval(
    queries: list[Query],
    search: Callable[[str, int], tuple[list[SearchResult], dict[str, Any]]],
    generate: Callable[[str], str],
    *,
    max_steps: int,
    retrieval_k: int,
    ks: tuple[int, ...] = RETRIEVAL_KS,
    log_path: Path | None = None,
    method: str = "unknown",
) -> dict[str, Any]:
    """对每条 query 跑固定步数的多步检索，并可选写入 JSONL 轨迹。"""
    if max_steps < 1:
        raise ValueError("multistep max_steps must be >= 1")
    eligible = [query for query in queries if query.gold_doc_ids]
    if not eligible:
        raise ValueError("Multistep retrieval requires gold evidence")
    handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
    per_query: dict[str, dict[str, float]] = {}
    ranked: dict[str, list[dict[str, Any]]] = {}
    try:
        for query in eligible:
            accumulated: list[SearchResult] = []
            current_query = query.text
            step_rows: list[dict[str, Any]] = []
            for step in range(1, max_steps + 1):
                results, trace = search(current_query, retrieval_k)
                accumulated = merge_ranked(accumulated, results)
                step_metrics = query_metrics(accumulated, query, ks)
                gold = set(query.gold_doc_ids)
                found = {item.doc_id for item in accumulated[: max(ks)]}
                generation = ""
                next_query = current_query
                stop_reason = "continue"
                if step_metrics.get(f"complete_evidence@{max(ks)}", 0.0) >= 1.0:
                    stop_reason = "complete_evidence"
                elif step == max_steps:
                    stop_reason = "max_steps"
                else:
                    documents = "\n".join(
                        f"- {item.doc_id} ({item.source})" for item in results[:retrieval_k]
                    )
                    generation = generate(IRCOT_PROMPT.format(
                        question=query.text, documents=documents or "(none)",
                    )).strip()
                    if generation.upper().startswith("END"):
                        stop_reason = "model_end"
                    elif generation:
                        next_query = generation.splitlines()[0].strip()
                    else:
                        stop_reason = "empty_generation"
                row = {
                    "query_id": query.id,
                    "method": method,
                    "step": step,
                    "stop_reason": stop_reason,
                    "original_query": query.text,
                    "step_query": current_query,
                    "next_query": next_query,
                    "generation": generation,
                    "base_hit_regions": trace.get("base_hit_regions", []),
                    "router_regions": trace.get("router_regions", []),
                    "bridge_regions": trace.get("bridge_regions", []),
                    "routed_regions": trace.get("routed_regions", []),
                    "step_results": serialize_ranked(results),
                    "accumulated_results": serialize_ranked(accumulated[: max(ks)]),
                    "gold_hit": sorted(gold & found),
                    "metrics": step_metrics,
                    "tokens": trace.get("tokens"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                step_rows.append(row)
                if handle is not None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if stop_reason != "continue":
                    break
                current_query = next_query
            final_metrics = query_metrics(accumulated, query, ks)
            per_query[query.id] = final_metrics
            ranked[query.id] = serialize_ranked(accumulated[: max(ks)])
            if handle is not None:
                handle.write(json.dumps({
                    "query_id": query.id,
                    "method": method,
                    "event": "query_complete",
                    "steps": len(step_rows),
                    "stop_reason": step_rows[-1]["stop_reason"],
                    "metrics": final_metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
    finally:
        if handle is not None:
            handle.close()
    summary: dict[str, Any] = {
        "num_queries": len(eligible),
        "max_steps": max_steps,
        "log_path": str(log_path) if log_path is not None else None,
        "per_query": per_query,
        "ranked_results": ranked,
    }
    for metric in next(iter(per_query.values())):
        values = [per_query[query.id][metric] for query in eligible]
        summary[metric] = sum(values) / len(values)
    if log_path is not None:
        write_json(log_path.with_suffix(".summary.json"), {
            "method": method,
            "num_queries": len(eligible),
            "max_steps": max_steps,
            **{key: summary[key] for key in summary if key not in {"per_query", "ranked_results"}},
        })
    return summary
