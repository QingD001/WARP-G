"""硬核心区域上的少量跨区 boundary 复制。"""

from __future__ import annotations

from collections import defaultdict

from warp.models import Region
from .coaccess_graph import CoaccessGraph


def replicate_boundaries(regions: list[Region], graph: CoaccessGraph, rho: float) -> list[Region]:
    """把跨区 co-access 最强的 top-ρ passage 复制进邻居区域，核心仍互斥。"""
    if not regions:
        return regions
    if rho < 0:
        raise ValueError("boundary_replication must be >= 0")
    for region in regions:
        core = list(region.metadata.get("core_doc_ids") or region.doc_ids)
        region.metadata["core_doc_ids"] = core
        region.metadata["boundary_doc_ids"] = list(region.metadata.get("boundary_doc_ids") or [])
        region.doc_ids = list(core)
    if rho == 0.0:
        return regions
    core_of = {doc_id: region.id for region in regions for doc_id in region.metadata["core_doc_ids"]}
    region_map = {region.id: region for region in regions}
    cross: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (left, right), weight in graph.query_edges.items():
        home_left = core_of.get(left)
        home_right = core_of.get(right)
        if home_left is None or home_right is None or home_left == home_right:
            continue
        cross[left][home_right] += weight
        cross[right][home_left] += weight
    scored = sorted(
        ((doc_id, sum(weights.values()), dict(weights)) for doc_id, weights in cross.items()),
        key=lambda item: (-item[1], item[0]),
    )
    budget = max(1, round(rho * len(core_of))) if scored else 0
    for doc_id, _total, neighbors in scored[:budget]:
        for region_id in neighbors:
            region = region_map[region_id]
            if doc_id in region.doc_ids:
                continue
            region.doc_ids.append(doc_id)
            region.metadata["boundary_doc_ids"].append(doc_id)
    for region in regions:
        region.doc_ids = sorted(dict.fromkeys(region.doc_ids))
        region.metadata["boundary_doc_ids"] = sorted(dict.fromkeys(region.metadata["boundary_doc_ids"]))
    return regions


def region_coaccess_neighbors(regions: list[Region], graph: CoaccessGraph) -> dict[str, dict[str, float]]:
    """区域之间由核心文档 query 边聚合的廉价 bridge 权重。"""
    core_of = {doc_id: region.id for region in regions for doc_id in region.metadata.get("core_doc_ids") or region.doc_ids}
    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (left, right), weight in graph.query_edges.items():
        home_left = core_of.get(left)
        home_right = core_of.get(right)
        if home_left is None or home_right is None or home_left == home_right:
            continue
        weights[home_left][home_right] += weight
        weights[home_right][home_left] += weight
    return {left: dict(right) for left, right in weights.items()}
