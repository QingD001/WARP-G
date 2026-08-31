"""基于 FAISS HNSW 的正式语义近邻索引。"""

from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np


def semantic_knn(
    ids: Sequence[str], vectors: Sequence[Sequence[float]], k: int, *, hnsw_m: int = 32,
    ef_construction: int = 200, ef_search: int = 128,
) -> dict[str, list[tuple[str, float]]]:
    """返回每个 ID 的 cosine top-k 邻居，不构造二次规模相似度矩阵。"""
    if len(ids) != len(vectors) or not ids:
        raise ValueError("ANN index requires aligned non-empty IDs and vectors")
    if k < 0 or hnsw_m <= 0 or ef_construction <= 0 or ef_search <= 0:
        raise ValueError("Invalid HNSW configuration")
    if k == 0:
        return {doc_id: [] for doc_id in ids}
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("ANN vectors must form a non-empty 2D matrix")
    faiss.normalize_L2(matrix)
    index = faiss.IndexHNSWFlat(matrix.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(matrix)
    scores, neighbors = index.search(matrix, min(k + 1, len(ids)))
    output: dict[str, list[tuple[str, float]]] = {}
    for row, doc_id in enumerate(ids):
        values = [
            (ids[int(position)], float(score))
            for position, score in zip(neighbors[row], scores[row])
            if position >= 0 and ids[int(position)] != doc_id
        ]
        output[doc_id] = values[:k]
    return output
