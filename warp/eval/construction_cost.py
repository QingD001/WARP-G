"""区域图成本的逐维聚合，避免把异质单位隐藏在单个分数中。"""

from __future__ import annotations

from typing import Any

from warp.models import ConstructionCost


def usage_to_cost(usage: dict[str, Any] | None) -> ConstructionCost:
    """把图检索器 delta/stats 折成 ConstructionCost，便于并入设计成本。"""
    payload = usage or {}
    return ConstructionCost(
        input_tokens=int(payload.get("logical_input_tokens") or 0),
        output_tokens=int(payload.get("logical_output_tokens") or 0),
        wall_seconds=float(payload.get("wall_seconds") or 0.0),
    )


def consumed_tokens(payload: dict[str, Any] | None) -> int:
    """把 deployment / first-run / online 字典折成文档中的总消耗 tokens。"""
    if not payload:
        return 0
    if "input_tokens" in payload or "embedding_tokens" in payload:
        return (
            int(payload.get("input_tokens") or 0)
            + int(payload.get("output_tokens") or 0)
            + int(payload.get("embedding_tokens") or 0)
        )
    return int(payload.get("logical_input_tokens") or 0) + int(payload.get("logical_output_tokens") or 0)


def token_efficiency(quality: float, tokens: int) -> float | None:
    """检索效果 / 总消耗 tokens；分母为 0 时无法定义。"""
    if tokens <= 0:
        return None
    return float(quality) / float(tokens)


def attach_token_efficiency(row: dict[str, Any], extra_online: int = 0) -> None:
    """按 design.md 写入含/不含设计成本的 Token Efficiency，暂不作为正式检验指标。"""
    deployed = consumed_tokens(row.get("deployment_cost") or row.get("actual_construction_cost"))
    first_payload = row.get("first_run_cost_including_probe")
    first_run = consumed_tokens(first_payload) if first_payload else deployed
    online = consumed_tokens(row.get("online_retrieval_cost")) + extra_online
    excluding = deployed + online
    including = first_run + online
    row["total_tokens_excluding_design"] = excluding
    row["total_tokens_including_design"] = including
    evidence = row.get("complete_evidence@10")
    if evidence is not None:
        row["token_efficiency_excluding_design"] = token_efficiency(float(evidence), excluding)
        row["token_efficiency_including_design"] = token_efficiency(float(evidence), including)
    answer_em = row.get("answer_em")
    if answer_em is not None:
        row["token_efficiency_em_excluding_design"] = token_efficiency(float(answer_em), excluding)
        row["token_efficiency_em_including_design"] = token_efficiency(float(answer_em), including)


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
