"""WARP-G 的无预算区域选择器。"""

from __future__ import annotations

import random
from collections import defaultdict

from warp.models import RegionFeatures


class RegionSelector:
    """只替换区域排序公式；WARP 按边际 score>0 自然结束，controls 取同样多的区域。

    四个 control 复用同一折 regions / 成本估计 / 预测，不重新构图，也不再用
    token 预算截断。WARP 默认用 coverage 折减的 greedy 边际效用。
    """

    METHODS = {"warp", "random_region", "frequency_only", "gain_only", "cost_only"}

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def score(self, region_id: str, features: dict[str, RegionFeatures],
              costs: dict[str, float], predicted_gains: dict[str, float]) -> float:
        return features[region_id].query_freq * max(predicted_gains[region_id], 0.0) / costs[region_id]

    def rank(
        self, method: str, features: dict[str, RegionFeatures], costs: dict[str, float],
        predicted_gains: dict[str, float],
    ) -> list[str]:
        if set(features) != set(costs) or any(cost <= 0 for cost in costs.values()):
            raise ValueError("Selection requires aligned regions and positive costs")
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Unknown selection method: {method}")
        ids = sorted(features)
        if set(predicted_gains) != set(features):
            raise ValueError("Selection requires one predicted gain per region")
        if method == "random_region":
            ranked = ids.copy()
            random.Random(self.seed).shuffle(ranked)
            return ranked
        if method == "frequency_only":
            return sorted(ids, key=lambda key: (-features[key].query_freq, key))
        if method == "gain_only":
            return sorted(ids, key=lambda key: (-predicted_gains[key], key))
        if method == "cost_only":
            return sorted(ids, key=lambda key: (costs[key], key))
        return sorted(
            ids,
            key=lambda key: (-self.score(key, features, costs, predicted_gains), costs[key], key),
        )

    def select(
        self, method: str, features: dict[str, RegionFeatures], costs: dict[str, float],
        predicted_gains: dict[str, float], limit: int | None = None,
        region_queries: dict[str, list[str]] | None = None,
    ) -> list[str]:
        method = method.lower()
        if method == "warp":
            return self._select_warp_marginal(features, costs, predicted_gains, region_queries or {})
        ranked = self.rank(method, features, costs, predicted_gains)
        if limit is None:
            raise ValueError("Control selectors must reuse WARP's selected region count")
        if limit < 0:
            raise ValueError("Control selector limit must be non-negative")
        return ranked[:limit]

    def _select_warp_marginal(
        self,
        features: dict[str, RegionFeatures],
        costs: dict[str, float],
        predicted_gains: dict[str, float],
        region_queries: dict[str, list[str]],
    ) -> list[str]:
        """greedy：Δ(r|S) ≈ 未覆盖需求折减 × max(gain,0) / cost。"""
        if set(features) != set(costs) or any(cost <= 0 for cost in costs.values()):
            raise ValueError("Selection requires aligned regions and positive costs")
        if set(predicted_gains) != set(features):
            raise ValueError("Selection requires one predicted gain per region")
        remaining = set(features)
        selected: list[str] = []
        coverage: dict[str, int] = defaultdict(int)
        while remaining:
            best_id: str | None = None
            best_score = 0.0
            for region_id in remaining:
                gain = max(predicted_gains[region_id], 0.0)
                if gain <= 0.0:
                    continue
                queries = region_queries.get(region_id) or []
                demand = (
                    sum(1.0 / (1.0 + coverage[query_id]) for query_id in queries)
                    if queries else float(features[region_id].query_freq)
                )
                score = demand * gain / costs[region_id]
                if score <= 0.0:
                    continue
                better = best_id is None or score > best_score
                tie = best_id is not None and score == best_score and (
                    costs[region_id], region_id
                ) < (costs[best_id], best_id)
                if better or tie:
                    best_score = score
                    best_id = region_id
            if best_id is None or best_score <= 0.0:
                break
            selected.append(best_id)
            remaining.remove(best_id)
            for query_id in region_queries.get(best_id) or []:
                coverage[query_id] += 1
        return selected


BudgetSelector = RegionSelector
