"""WARP-G 的无预算区域选择器。"""

from __future__ import annotations

import random

from warp.models import RegionFeatures


class RegionSelector:
    """只替换区域排序公式；WARP 按 score_i>0 自然结束，controls 取同样多的区域。

    四个 control 复用同一折 regions / 成本估计 / 预测，不重新构图，也不再用
    token 预算截断。
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
    ) -> list[str]:
        ranked = self.rank(method, features, costs, predicted_gains)
        method = method.lower()
        if method == "warp":
            return [
                region_id for region_id in ranked
                if self.score(region_id, features, costs, predicted_gains) > 0
            ]
        if limit is None:
            raise ValueError("Control selectors must reuse WARP's selected region count")
        if limit < 0:
            raise ValueError("Control selector limit must be non-negative")
        return ranked[:limit]


BudgetSelector = RegionSelector
