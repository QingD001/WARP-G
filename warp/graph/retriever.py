"""正式区域图检索协议。"""

from typing import Protocol

from warp.models import SearchResult
from .builder import RegionalGraph


class GraphRetriever(Protocol):
    """Advisor 与 pipeline 依赖的图检索协议。"""

    def search(self, query: str, graph: RegionalGraph, k: int = 10) -> list[SearchResult]:
        """从一个已物化区域图检索 passage。"""
        ...

    def stats(self) -> dict[str, float | int]:
        ...

    def delta(self, before: dict[str, float | int]) -> dict[str, float | int]:
        ...
