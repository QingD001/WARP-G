"""在已缓存的检索结果上运行固定的官方 HippoRAG2 QA reader。"""

from __future__ import annotations

from typing import Any

from warp.models import Document, Query, SearchResult
from .qa import answer_em, answer_f1
from .retrieval import ranked_from_payload


def evaluate_hipporag2_reader(
    queries: list[Query],
    ranked_by_query: dict[str, list[SearchResult] | list[dict[str, Any]]],
    documents: list[Document],
    hipporag_graph: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Use HippoRAG 2's frozen QA prompt on cached retrieval output. Do not search again."""
    try:
        from hipporag.utils.misc_utils import QuerySolution
    except ImportError as exc:
        raise ImportError("Install the project with `pip install -e .` to run HippoRAG reader evaluation") from exc
    rag = hipporag_graph.backend
    if rag is None:
        raise TypeError("Reader evaluation requires an official HippoRAG2 full graph")
    doc_map = {doc.id: doc for doc in documents}
    eligible = [query for query in queries if query.answer is not None]
    if not eligible:
        raise ValueError("Reader evaluation requires gold answers")
    solutions = []
    for query in eligible:
        payload = ranked_by_query.get(query.id)
        if payload is None:
            raise KeyError(
                f"Reader is missing cached retrieval for {query.id!r}; "
                "the retrieval stage must run first and pass ranked_results"
            )
        results = ranked_from_payload(payload) if not payload or not isinstance(payload[0], SearchResult) else payload
        solutions.append(QuerySolution(
            question=query.text,
            docs=[doc_map[result.doc_id].content for result in results[:top_k] if result.doc_id in doc_map],
            doc_scores=None,
            doc_metadata=[{"warp_doc_id": result.doc_id} for result in results[:top_k]],
        ))
    tracker = rag._warp_usage_tracker
    before = tracker.get("reader")
    tracker.phase = "reader"
    try:
        answered, _, _ = rag.qa(solutions)
    finally:
        tracker.phase = "idle"
    em_total = f1_total = 0.0
    predictions: list[dict[str, Any]] = []
    for query, solution in zip(eligible, answered):
        golds = query.answer if isinstance(query.answer, list) else [query.answer]
        em = max(answer_em(solution.answer, gold) for gold in golds)
        f1 = max(answer_f1(solution.answer, gold) for gold in golds)
        em_total += em
        f1_total += f1
        predictions.append({"query_id": query.id, "prediction": solution.answer, "gold_answers": golds})
    after = tracker.get("reader")
    usage = {key: after.get(key, 0) - before.get(key, 0) for key in set(after) | set(before)}
    return {
        "answer_em": em_total / len(eligible), "answer_f1": f1_total / len(eligible),
        "num_queries": len(eligible), "reader_usage": usage, "predictions": predictions,
        "reader_reused_retrieval": True,
    }
