"""区域图成本的逐维聚合，避免把异质单位隐藏在单个分数中。"""

from __future__ import annotations

from warp.models import ConstructionCost


def aggregate_costs(costs: list[ConstructionCost]) -> ConstructionCost:
    """对 token、时间、图规模和存储分别求和。"""
    return ConstructionCost(
        input_tokens=sum(cost.input_tokens for cost in costs),
        output_tokens=sum(cost.output_tokens for cost in costs),
        embedding_tokens=sum(cost.embedding_tokens for cost in costs),
        wall_seconds=sum(cost.wall_seconds for cost in costs),
        nodes=sum(cost.nodes for cost in costs),
        edges=sum(cost.edges for cost in costs),
        storage_bytes=sum(cost.storage_bytes for cost in costs),
        estimated_usd=sum(cost.estimated_usd for cost in costs),
    )
