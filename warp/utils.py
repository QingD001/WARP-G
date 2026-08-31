"""无第三方依赖的文本、向量、JSON 与批处理辅助函数。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """提供 BM25 与构图前成本估算共用的确定性分词。"""
    return TOKEN_RE.findall(text.lower())


def stable_hash(value: str, modulo: int) -> int:
    """使用稳定哈希，避免 Python 进程级 hash seed 破坏实验复现。"""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def cosine(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度；零向量返回 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def minmax(values: list[float]) -> list[float]:
    """把数值缩放到 [0,1]，常量零列保持为零。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 if hi else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL、JSON list 或常见顶层 data/documents/queries 容器。"""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if isinstance(value, list):
        return value
    for key in ("data", "documents", "queries", "items"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"Cannot find a record list in {path}")


def write_json(path: str | Path, value: Any) -> None:
    """创建父目录并以 UTF-8、可读缩进形式原子化语义地写结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    """按固定大小产生连续 batch，最后一批允许不足。"""
    for start in range(0, len(values), size):
        yield values[start:start + size]
