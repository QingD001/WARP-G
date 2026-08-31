"""规范化 JSON/JSONL 数据加载器。

本层只按固定规则生成交叉拟合 folds，不从 held-out test 反推任何设计信息；
外部字段统一映射到共享 Document/Query 模型。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from warp.models import DatasetBundle, Document, Query
from warp.utils import read_json_records


def _document(row: dict[str, Any], index: int) -> Document:
    """兼容常见 passage 字段名，同时把未知字段保留在 metadata。"""
    doc_id = row.get("id", row.get("doc_id", row.get("_id", index)))
    title = row.get("title", "") or ""
    text = row.get("text", row.get("passage", row.get("contents", row.get("content", ""))))
    if isinstance(text, list):
        text = " ".join(map(str, text))
    known = {"id", "doc_id", "_id", "title", "text", "passage", "contents", "content"}
    return Document(str(doc_id), str(text), str(title), {k: v for k, v in row.items() if k not in known})


def _query(row: dict[str, Any], index: int) -> Query:
    """规范化问题、证据 ID 和单/多答案格式。"""
    qid = row.get("id", row.get("query_id", row.get("question_id", row.get("_id", index))))
    text = row.get("query", row.get("question", row.get("text", "")))
    gold = row.get("gold_doc_ids", row.get("supporting_doc_ids", row.get("gold_docs", [])))
    if isinstance(gold, str):
        gold = [gold]
    answer = row.get("answer")
    known = {"id", "query_id", "question_id", "_id", "query", "question", "text", "gold_doc_ids", "supporting_doc_ids", "gold_docs", "answer"}
    normalized_answer = ([str(value) for value in answer] if isinstance(answer, list)
                         else None if answer is None else str(answer))
    return Query(str(qid), str(text), [str(x) for x in gold], normalized_answer,
                 {k: v for k, v in row.items() if k not in known})


def load_documents(path: str | Path) -> list[Document]:
    """从 JSON 或 JSONL 加载共享 corpus。"""
    return [_document(row, i) for i, row in enumerate(read_json_records(path))]


def load_queries(path: str | Path) -> list[Query]:
    """加载一个已经确定好的 query split。"""
    return [_query(row, i) for i, row in enumerate(read_json_records(path))]


def load_bundle(config: dict[str, Any]) -> DatasetBundle:
    """Load normalized JSON/JSONL files or one of the benchmark adapters."""
    name = str(config.get("name", "generic")).lower()
    if name != "generic":
        from .benchmarks import LOADERS
        if name not in LOADERS:
            raise ValueError(f"Unknown dataset {name!r}; choose generic or {sorted(LOADERS)}")
        return LOADERS[name](config)
    required = ("corpus", "train", "dev", "test")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing dataset paths: {', '.join(missing)}")
    return DatasetBundle(
        documents=load_documents(config["corpus"]),
        train=load_queries(config["train"]),
        dev=load_queries(config["dev"]),
        test=load_queries(config["test"]),
    )


def load_crossfit_bundles(config: dict[str, Any], folds: int, seed: int) -> list[DatasetBundle]:
    """Create deterministic held-out folds from one complete query set."""
    if not config.get("corpus") or not config.get("queries"):
        raise ValueError("Cross-fitting requires dataset.corpus and dataset.queries")
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    documents = load_documents(config["corpus"])
    queries = load_queries(config["queries"])
    ordered = sorted(queries, key=lambda query: (
        hashlib.sha256(f"{seed}:{query.id}".encode()).hexdigest(), query.id,
    ))
    output: list[DatasetBundle] = []
    for fold in range(folds):
        test = [query for index, query in enumerate(ordered) if index % folds == fold]
        train = [query for index, query in enumerate(ordered) if index % folds != fold]
        output.append(DatasetBundle(documents=list(documents), train=train, dev=[], test=test))
    return output
