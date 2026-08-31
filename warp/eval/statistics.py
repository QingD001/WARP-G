"""论文结果所需的确定性 paired bootstrap 与配对随机化检验。"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    return sum(values) / len(values)


def paired_bootstrap_interval(
    values: Sequence[float], *, samples: int = 10_000, confidence: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    if not values or samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap sample or configuration")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples))
    tail = (1.0 - confidence) / 2.0
    return estimates[min(int(tail * samples), samples - 1)], estimates[min(int((1.0 - tail) * samples), samples - 1)]


def paired_randomization_pvalue(
    candidate: Sequence[float], reference: Sequence[float], *, samples: int = 10_000, seed: int = 42,
) -> float:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Paired randomization requires non-empty aligned samples")
    differences = [left - right for left, right in zip(candidate, reference)]
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(value if rng.random() < 0.5 else -value for value in differences) / len(differences))
        >= observed - 1e-15
        for _ in range(samples)
    )
    return (extreme + 1) / (samples + 1)


def regression_metrics(expected: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Regression metrics require non-empty aligned samples")
    errors = [left - right for left, right in zip(expected, predicted)]

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        output = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for position in range(start, end):
                output[order[position]] = rank
            start = end
        return output

    left_ranks, right_ranks = ranks(expected), ranks(predicted)
    left_mean, right_mean = mean(left_ranks), mean(right_ranks)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks) * sum((y - right_mean) ** 2 for y in right_ranks))
    return {
        "mae": mean([abs(value) for value in errors]),
        "rmse": math.sqrt(mean([value * value for value in errors])),
        "spearman": numerator / denominator if denominator else 0.0,
    }
