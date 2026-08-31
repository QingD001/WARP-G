"""HotpotQA、2Wiki、MuSiQue 与 PopQA 的薄适配层。

适配器只处理原始 schema 差异，并强制所有 split 使用显式共享 corpus。
"""

from __future__ import annotations

from typing import Any

from warp.models import DatasetBundle, Document, Query
from warp.utils import read_json_records


def _support_titles(row: dict[str, Any]) -> list[str]:
    """从多个 benchmark schema 提取文档级 supporting evidence ID。"""
    explicit = row.get("gold_doc_ids", row.get("supporting_doc_ids", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    if explicit:
        # Normalized WARP files already contain stable corpus IDs.  Prefer them
        # over benchmark-specific title extraction so duplicate Wikipedia titles
        # do not collapse distinct passages.
        return list(dict.fromkeys(str(value) for value in explicit))
    supports = row.get("supporting_facts", row.get("supporting_docs", row.get("supporting_paragraphs", [])))
    found: list[str] = []
    for support in supports:
        if isinstance(support, (list, tuple)):
            value = support[0]
        elif isinstance(support, dict):
            value = support.get("title", support.get("doc_id", support.get("paragraph_id")))
        else:
            value = support
        if value is not None and str(value) not in found:
            found.append(str(value))
    if not found:
        for paragraph in row.get("paragraphs", row.get("contexts", [])):
            if isinstance(paragraph, dict) and paragraph.get("is_supporting", False):
                value = paragraph.get("title", paragraph.get("doc_id", paragraph.get("idx")))
                if value is not None and str(value) not in found:
                    found.append(str(value))
    return found


def _qa_rows(path: str, prefix: str) -> tuple[list[dict[str, Any]], list[Query]]:
    rows = read_json_records(path)
    queries = [Query(
        str(row.get("_id", row.get("id", f"{prefix}-{i}"))),
        str(row.get("question", row.get("query", ""))),
        _support_titles(row),
        ([str(value) for value in row["answer"]] if isinstance(row.get("answer"), list)
         else None if row.get("answer") is None else str(row["answer"])),
        {"type": row.get("type")},
    ) for i, row in enumerate(rows)]
    return rows, queries


def _multihop(config: dict[str, Any]) -> DatasetBundle:
    """加载三个多跳 benchmark，并确保所有 split 共享同一 corpus。"""
    required = ("corpus", "train", "dev", "test")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Multi-hop benchmarks require explicit paths for: {', '.join(missing)}")
    splits: dict[str, list[Query]] = {}
    for split in ("train", "dev", "test"):
        _, splits[split] = _qa_rows(config[split], split)
    corpus_rows = read_json_records(config["corpus"])
    documents = [Document(
        str(row.get("id", row.get("doc_id", row.get("title", row.get("idx", i))))),
        str(row.get("text", row.get("passage", row.get("content", "")))),
        str(row.get("title", "")),
        {"source_idx": row.get("idx")},
    ) for i, row in enumerate(corpus_rows)]
    return DatasetBundle(documents, splits["train"], splits["dev"], splits["test"])


def _popqa(config: dict[str, Any]) -> DatasetBundle:
    from .base import load_documents, load_queries
    required = ("corpus", "train", "dev", "test")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"PopQA requires explicit paths for: {', '.join(missing)}")
    return DatasetBundle(
        load_documents(config["corpus"]),
        load_queries(config["train"]),
        load_queries(config["dev"]),
        load_queries(config["test"]),
    )


LOADERS = {
    "hotpotqa": _multihop,
    "hotpot": _multihop,
    "musique": _multihop,
    "2wiki": _multihop,
    "twowiki": _multihop,
    "popqa": _popqa,
}
