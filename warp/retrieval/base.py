"""所有基础检索器遵循的最小结构化接口。"""

from __future__ import annotations

from typing import Protocol

from warp.models import Document, SearchResult


class Retriever(Protocol):
    """支持全语料检索和 `doc_ids` 限定区域检索的协议。"""
    def fit(self, documents: list[Document]) -> "Retriever":
        """在共享 corpus 上构建索引。"""
        ...

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """检索全 corpus 或 doc_ids 指定的子集。"""
        ...
