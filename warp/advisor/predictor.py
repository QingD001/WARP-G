"""使用 probe 标签训练正式的 region-level Graph Benefit Predictor。"""

from __future__ import annotations

from typing import Any
import random

from warp.models import RegionFeatures
from warp.eval.statistics import regression_metrics


class BenefitPredictor:
    """使用 LightGBM 预测未 probe Region 的 Graph 增益。"""

    def __init__(self, seed: int = 42, **params: Any) -> None:
        self.seed = seed
        self.params = params
        self.model: Any = None

    def fit(self, features: dict[str, RegionFeatures], gains: dict[str, float]) -> "BenefitPredictor":
        """仅用 probe region 的真实 gain 拟合。"""
        ids = sorted(gains)
        if len(ids) < 4:
            raise ValueError(
                "LightGBM benefit prediction requires at least four probe regions; "
                "increase corpus region count or probe_fraction"
            )
        missing = [region_id for region_id in ids if region_id not in features]
        if missing:
            raise KeyError(f"Probe outcomes refer to unknown regions: {missing}")
        from lightgbm import LGBMRegressor

        defaults = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "min_child_samples": 2,
            "random_state": self.seed,
            "verbosity": -1,
        }
        defaults.update(self.params)
        x = [features[region_id].vector() for region_id in ids]
        y = [gains[region_id] for region_id in ids]
        self.model = LGBMRegressor(**defaults).fit(x, y)
        return self

    def predict(self, features: dict[str, RegionFeatures]) -> dict[str, float]:
        """按稳定 region ID 顺序预测所有区域的 graph gain。"""
        if self.model is None:
            raise RuntimeError("BenefitPredictor.fit must be called first")
        ids = sorted(features)
        values = self.model.predict([features[region_id].vector() for region_id in ids])
        return {region_id: float(value) for region_id, value in zip(ids, values)}

    def feature_importance(self) -> dict[str, float]:
        """返回与九维特征名对齐的 LightGBM feature importance。"""
        if self.model is None:
            raise RuntimeError("BenefitPredictor.fit must be called first")
        return dict(zip(RegionFeatures.names(), map(float, self.model.feature_importances_)))

    @classmethod
    def cross_validate(
        cls, features: dict[str, RegionFeatures], gains: dict[str, float], seed: int = 42,
    ) -> dict[str, object]:
        """对 probe regions 做 leave-one-region-out，直接检验收益预测能力。"""
        ids = sorted(gains)
        if len(ids) < 6:
            raise ValueError("Predictor validation requires at least six probe regions")
        expected: list[float] = []
        predicted: list[float] = []
        rows: list[dict[str, float | str]] = []
        for held_out in ids:
            training = {key: gains[key] for key in ids if key != held_out}
            model = cls(seed).fit(features, training)
            value = model.predict({held_out: features[held_out]})[held_out]
            expected.append(gains[held_out])
            predicted.append(value)
            rows.append({"region_id": held_out, "observed_gain": gains[held_out], "predicted_gain": value})
        return {**regression_metrics(expected, predicted), "predictions": rows}

    @classmethod
    def feature_ablation(
        cls, features: dict[str, RegionFeatures], gains: dict[str, float], seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """逐项置零并执行 region-held-out 验证，输出每个 cheap feature 的贡献。"""
        from dataclasses import replace

        output: dict[str, dict[str, float]] = {}
        for name in RegionFeatures.names():
            ablated = {key: replace(value, **{name: 0.0}) for key, value in features.items()}
            result = cls.cross_validate(ablated, gains, seed)
            output[name] = {metric: float(result[metric]) for metric in ("mae", "rmse", "spearman")}
        return output

    @classmethod
    def learning_curve(
        cls, features: dict[str, RegionFeatures], gains: dict[str, float], seed: int = 42,
        repetitions: int = 20,
    ) -> list[dict[str, float | int]]:
        """在已获得的 probe labels 内测量训练区域数量与泛化误差的关系。"""
        ids = sorted(gains)
        if len(ids) < 6 or repetitions <= 0:
            raise ValueError("Learning curve requires at least six probes and positive repetitions")
        rng = random.Random(seed)
        output: list[dict[str, float | int]] = []
        for train_size in sorted(set([4, max(4, len(ids) // 2), len(ids) - 1])):
            metrics: list[dict[str, float]] = []
            for _ in range(repetitions):
                training_ids = set(rng.sample(ids, train_size))
                held_out = [key for key in ids if key not in training_ids]
                model = cls(seed).fit(features, {key: gains[key] for key in training_ids})
                values = model.predict({key: features[key] for key in held_out})
                metrics.append(regression_metrics(
                    [gains[key] for key in held_out], [values[key] for key in held_out],
                ))
            output.append({
                "train_regions": train_size,
                "held_out_regions": len(ids) - train_size,
                **{name: sum(row[name] for row in metrics) / len(metrics)
                   for name in ("mae", "rmse", "spearman")},
            })
        return output
