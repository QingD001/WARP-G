"""使用正式外部 encoder 的精确 Dense retrieval。"""

from __future__ import annotations

from typing import Any

from warp.models import Document, SearchResult
from warp.utils import cosine


class DenseRetriever:
    """精确向量检索器；文档与查询编码由同一正式模型提供。"""

    def __init__(self, encoder: Any, model_name: str) -> None:
        if encoder is None:
            raise TypeError("DenseRetriever requires an explicit production encoder")
        self.model_name = model_name
        self.encoder = encoder
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []
        self._index: dict[str, int] = {}

    def fit(self, documents: list[Document]) -> "DenseRetriever":
        """一次性编码 corpus，并校验 encoder 输出与文档一一对应。"""
        self.documents = list(documents)
        self.vectors = self.encoder.encode_documents([doc.content for doc in documents])
        if len(self.vectors) != len(self.documents):
            raise RuntimeError(
                f"Dense encoder returned {len(self.vectors)} vectors for {len(self.documents)} documents"
            )
        self._index = {doc.id: index for index, doc in enumerate(documents)}
        return self

    def encode(self, texts: list[str]) -> list[list[float]]:
        """使用正式 query instruction 编码查询。"""
        values = self.encoder.encode_queries(texts)
        if len(values) != len(texts):
            raise RuntimeError(f"Dense encoder returned {len(values)} vectors for {len(texts)} queries")
        return values

    def vector(self, doc_id: str) -> list[float]:
        """返回已缓存文档向量，供语义边和 dispersion 特征复用。"""
        return self.vectors[self._index[doc_id]]

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """执行精确余弦检索。"""
        if k <= 0:
            return []
        query_vector = self.encode([query])[0]
        allowed = None if doc_ids is None else set(doc_ids)
        scores = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
            if allowed is None or self.documents[index].id in allowed
        ]
        scores.sort(key=lambda item: (-item[1], self.documents[item[0]].id))
        return [
            SearchResult(self.documents[index].id, score, "dense", rank + 1)
            for rank, (index, score) in enumerate(scores[:k])
        ]
