"""区域图句柄与正式图构建协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from warp.models import ConstructionCost, Document, Region


@dataclass
class RegionalGraph:
    """正式区域图句柄；`backend` 保存官方 HippoRAG2 实例。"""

    region_id: str
    doc_ids: list[str]
    cost: ConstructionCost
    metadata: dict[str, object] = field(default_factory=dict)
    backend: Any = field(default=None, repr=False)


class GraphBuilder(Protocol):
    """Advisor 依赖的图构建协议。"""

    def estimate_cost(self, region: Region, documents: list[Document]) -> float:
        """在真实构图前返回预算选择所需的成本估计。"""
        ...

    def build(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """真实构建一个区域图并测量实际成本。"""
        ...

    def build_full_graph(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """构建隔离的 corpus-wide Full Graph 基线。"""
        ...
