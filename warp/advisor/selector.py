"""WARP-G 的预算约束区域选择器。"""

from __future__ import annotations

import random

from warp.models import RegionFeatures


class BudgetSelector:
    """在同一组不可拆分区域上比较预算选择规则。"""

    METHODS = {"warp", "random_region", "frequency_only", "gain_only", "cost_only"}

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def select(
        self, method: str, budget: float, features: dict[str, RegionFeatures], costs: dict[str, float],
        predicted_gains: dict[str, float],
    ) -> list[str]:
        if budget < 0 or set(features) != set(costs) or any(cost <= 0 for cost in costs.values()):
            raise ValueError("Selection requires aligned regions, positive costs, and a non-negative budget")
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Unknown selection method: {method}")
        ids = sorted(features)
        if set(predicted_gains) != set(features):
            raise ValueError("Selection requires one predicted gain per region")

        if method == "random_region":
            ranked = ids.copy()
            random.Random(self.seed).shuffle(ranked)
        elif method == "frequency_only":
            # 破平局只用稳定 id，避免 cost 信号泄漏回单信号消融。
            ranked = sorted(ids, key=lambda key: (-features[key].query_freq, key))
        elif method == "gain_only":
            ranked = sorted(ids, key=lambda key: (-predicted_gains[key], key))
        elif method == "cost_only":
            ranked = sorted(ids, key=lambda key: (costs[key], key))
        else:
            score = {
                key: features[key].query_freq * max(predicted_gains[key], 0.0) / costs[key]
                for key in ids
            }
            ranked = sorted(ids, key=lambda key: (-score[key], costs[key], key))

        selected: list[str] = []
        spent = 0.0
        for region_id in ranked:
            if method in {"warp", "gain_only"} and predicted_gains[region_id] <= 0:
                continue
            if spent + costs[region_id] <= budget + 1e-9:
                selected.append(region_id)
                spent += costs[region_id]
        return selected
