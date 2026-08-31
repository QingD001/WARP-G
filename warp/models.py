"""WARP-G 跨模块共享的数据模型。

这些 dataclass 是各层之间的稳定边界：数据加载器只负责生成 Document/Query，
检索器只交换 SearchResult，物理设计层通过 Region/RegionFeatures/ConstructionCost
通信。保持这些对象简单，可以替换图后端而不改 advisor 和 evaluator。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """一个可检索 passage；`id` 是评测和后端映射使用的永久标识。"""
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """返回统一送入 lexical、dense 和 graph 后端的标题加正文。"""
        return f"{self.title}\n{self.text}".strip()


@dataclass(slots=True)
class Query:
    """一条设计或评测问题，gold_doc_ids 保存完整证据 passage ID。"""
    id: str
    text: str
    gold_doc_ids: list[str] = field(default_factory=list)
    answer: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """统一的检索结果；source/region_id 用于追踪结果来自哪个物理结构。"""
    doc_id: str
    score: float
    source: str
    rank: int = 0
    region_id: str | None = None


@dataclass(slots=True)
class Region:
    """由共访问图社区发现得到的 passage ID 集合。"""
    id: str
    doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetBundle:
    """共享 corpus 及严格分离的 train/design、dev、test query。"""
    documents: list[Document]
    train: list[Query]
    dev: list[Query] = field(default_factory=list)
    test: list[Query] = field(default_factory=list)


@dataclass(slots=True)
class ConstructionCost:
    """一次图构建的多维实测成本，不把时间等强行折成单一货币值。"""
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    wall_seconds: float = 0.0
    nodes: int = 0
    edges: int = 0
    storage_bytes: int = 0
    estimated_usd: float = 0.0

    @property
    def selection_cost(self) -> float:
        """Stable pre-construction cost proxy used by the budget selector."""
        return float(self.input_tokens + self.output_tokens + self.embedding_tokens)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class RegionFeatures:
    """构图前即可获得的九维便宜特征，是 benefit predictor 的唯一输入。"""
    region_id: str
    num_docs: float
    num_tokens: float
    query_freq: float
    base_recall: float
    failure_rate: float
    avg_retrieval_entropy: float
    multi_doc_rate: float
    embedding_dispersion: float
    coaccess_density: float

    @classmethod
    def names(cls) -> list[str]:
        """返回 predictor 训练时固定的特征列顺序。"""
        return [
            "num_docs", "num_tokens", "query_freq", "base_recall",
            "failure_rate", "avg_retrieval_entropy", "multi_doc_rate",
            "embedding_dispersion", "coaccess_density",
        ]

    def vector(self) -> list[float]:
        """按 `names()` 顺序生成模型输入向量。"""
        return [float(getattr(self, name)) for name in self.names()]

    def to_dict(self) -> dict[str, Any]:
        """输出带 region_id 的可审计特征字典。"""
        return asdict(self)
