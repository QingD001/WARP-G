"""在任意检索结果上运行固定的官方 HippoRAG2 QA reader。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from warp.models import Document, Query, SearchResult
from .qa import answer_em, answer_f1


def evaluate_hipporag2_reader(
    queries: list[Query],
    search: Callable[[str, int], list[SearchResult]],
    documents: list[Document],
    hipporag_graph: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Use HippoRAG 2's frozen QA prompt and QA LLM on arbitrary retrieval output."""
    try:
        from hipporag.utils.misc_utils import QuerySolution
    except ImportError as exc:
        raise ImportError("Install the project with `pip install -e .` to run HippoRAG reader evaluation") from exc
    rag = hipporag_graph.backend
    if rag is None:
        raise TypeError("Reader evaluation requires an official HippoRAG2 full graph")
    # Reader 始终使用同一个 full-graph HippoRAG 实例中的 prompt manager/QA LLM，
    # 但 docs 由待比较的检索方法提供，因此只改变 evidence，不改变生成器。
    doc_map = {doc.id: doc for doc in documents}
    eligible = [query for query in queries if query.answer is not None]
    if not eligible:
        raise ValueError("Reader evaluation requires gold answers")
    solutions = []
    for query in eligible:
        results = search(query.text, top_k)
        solutions.append(QuerySolution(
            question=query.text,
            docs=[doc_map[result.doc_id].content for result in results[:top_k]],
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
    # 多答案问题取所有规范答案中的最佳 EM/F1，这是 QA benchmark 的常规口径。
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
    }
