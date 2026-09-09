"""把共访问图划分为稳定 Region 的社区发现实现。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from warp.models import Region
from .coaccess_graph import CoaccessGraph


class RegionPartitioner:
    """使用 Leiden 将共访问图划分为稳定 Region。"""

    def __init__(self, resolution: float = 1.0, seed: int = 42, min_region_size: int = 1) -> None:
        self.resolution = resolution
        self.seed = seed
        self.min_region_size = min_region_size

    def partition(
        self, graph: CoaccessGraph,
        cost_fn: Callable[[list[str]], float] | None = None,
        max_cost: float | None = None,
    ) -> list[Region]:
        """优先运行 Leiden，合并过小社区，过贵则递归再切，再生成稳定 rXXXX ID。"""
        membership = self._leiden(graph)
        groups: dict[int, list[str]] = defaultdict(list)
        for node, community in zip(graph.nodes, membership):
            groups[int(community)].append(node)
        groups = self._merge_small(groups, graph)
        groups = self._split_expensive(groups, graph, cost_fn, max_cost)
        ordered = sorted((sorted(ids) for ids in groups.values()), key=lambda ids: (ids[0], len(ids)))
        regions = []
        for index, ids in enumerate(ordered):
            region = Region(f"r{index:04d}", ids, metadata={"core_doc_ids": list(ids), "boundary_doc_ids": []})
            regions.append(region)
        return regions

    def _leiden(self, graph: CoaccessGraph) -> list[int]:
        import igraph as ig
        import leidenalg as la
        index = {node: i for i, node in enumerate(graph.nodes)}
        edges = [(index[a], index[b]) for a, b in graph.edges]
        weights = list(graph.edges.values())
        value = ig.Graph(n=len(graph.nodes), edges=edges, directed=False)
        partition = la.find_partition(
            value, la.RBConfigurationVertexPartition, weights=weights,
            resolution_parameter=self.resolution, seed=self.seed,
        )
        return list(partition.membership)

    def _merge_small(self, groups: dict[int, list[str]], graph: CoaccessGraph) -> dict[int, list[str]]:
        """把小社区并入连接权重最大的合格社区，避免大量碎片图。"""
        if self.min_region_size <= 1 or len(groups) <= 1:
            return groups
        node_group = {node: group for group, nodes in groups.items() for node in nodes}
        neighbors = graph.neighbors()
        for group in list(sorted(groups)):
            nodes = groups.get(group, [])
            if not nodes or len(nodes) >= self.min_region_size:
                continue
            scores: dict[int, float] = defaultdict(float)
            for node in nodes:
                for other, weight in neighbors[node].items():
                    target = node_group[other]
                    if target != group and len(groups.get(target, [])) >= self.min_region_size:
                        scores[target] += weight
            candidates = [key for key, value in groups.items() if key != group and value]
            if not candidates:
                continue
            target = min(scores, key=lambda key: (-scores[key], key)) if scores else min(candidates)
            groups[target].extend(nodes)
            groups[group] = []
            for node in nodes:
                node_group[node] = target
        return {key: value for key, value in groups.items() if value}

    def _split_expensive(
        self,
        groups: dict[int, list[str]],
        graph: CoaccessGraph,
        cost_fn: Callable[[list[str]], float] | None,
        max_cost: float | None,
    ) -> dict[int, list[str]]:
        """把估计成本超过上限的社区递归 Leiden，直到成为可物化 physical unit。"""
        if cost_fn is None or max_cost is None or max_cost <= 0:
            return groups
        pending = [list(nodes) for nodes in groups.values() if nodes]
        output: list[list[str]] = []
        while pending:
            nodes = pending.pop()
            if len(nodes) <= max(self.min_region_size, 1) or cost_fn(nodes) <= max_cost:
                output.append(nodes)
                continue
            subgraph = graph.subgraph(nodes)
            if len(subgraph.nodes) <= 1 or not subgraph.edges:
                output.append(nodes)
                continue
            membership = self._leiden(subgraph)
            children: dict[int, list[str]] = defaultdict(list)
            for node, community in zip(subgraph.nodes, membership):
                children[int(community)].append(node)
            children = self._merge_small(children, subgraph)
            parts = [value for value in children.values() if value]
            if len(parts) <= 1:
                output.append(nodes)
                continue
            pending.extend(parts)
        return {index: nodes for index, nodes in enumerate(output)}
