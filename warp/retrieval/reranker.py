"""融合候选的正式 CrossEncoder 重排器。"""

from __future__ import annotations

from typing import Any, Protocol

from warp.models import Document, SearchResult


class Reranker(Protocol):
    """Pipeline 依赖的候选重排接口。"""

    def rerank(self, query: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
        ...


class CrossEncoderReranker:
    """使用锁定 revision 的 sentence-transformers CrossEncoder 重排候选。"""

    def __init__(self, documents: list[Document], model_name: str = "BAAI/bge-reranker-v2-m3",
                 batch_size: int = 32, max_length: int = 512, device: str | None = None,
                 revision: str | None = None) -> None:
        if not revision:
            raise ValueError("CrossEncoderReranker requires a pinned model revision")
        from sentence_transformers import CrossEncoder

        self.documents = {doc.id: doc for doc in documents}
        self.batch_size = batch_size
        self.model_name = model_name
        self.model: Any = CrossEncoder(
            model_name, max_length=max_length, device=device,
            trust_remote_code=True, revision=revision,
        )

    def rerank(self, query: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
        """对 query-passage 对直接打分，并保留候选原来的 region 归属。"""
        if not candidates:
            return []
        missing = [result.doc_id for result in candidates if result.doc_id not in self.documents]
        if missing:
            raise KeyError(f"Reranker received unknown document IDs: {missing[:5]}")
        pairs = [(query, self.documents[result.doc_id].content) for result in candidates]
        values = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = values.tolist() if hasattr(values, "tolist") else list(values)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"CrossEncoder returned {len(scores)} scores for {len(candidates)} candidates"
            )
        ranked = sorted(zip(candidates, scores), key=lambda item: (-float(item[1]), item[0].doc_id))[:k]
        return [
            SearchResult(result.doc_id, float(score), "cross_encoder", rank + 1, result.region_id)
            for rank, (result, score) in enumerate(ranked)
        ]
