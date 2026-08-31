#!/usr/bin/env python3
"""把原始共享 corpus/QA 文件规范化为 WARP-G 可审计数据集。

脚本只接受 canonical train/dev/test split，校验证据 ID 全部存在，并写 split_manifest.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    """读取原始 benchmark JSON list。"""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def supporting_titles(row: dict[str, Any]) -> list[str]:
    """统一抽取显式 supporting facts、support paragraphs 与 gold IDs。"""
    found: list[str] = []
    for support in row.get("supporting_facts", row.get("supporting_docs", [])):
        value = support[0] if isinstance(support, list) else support.get("title", support.get("doc_id"))
        if value is not None and str(value) not in found:
            found.append(str(value))
    for paragraph in row.get("paragraphs", row.get("contexts", [])):
        if isinstance(paragraph, dict) and paragraph.get("is_supporting", False):
            value = paragraph.get("title", paragraph.get("doc_id", paragraph.get("idx")))
            if value is not None and str(value) not in found:
                found.append(str(value))
    explicit = row.get("gold_doc_ids", row.get("supporting_doc_ids", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    for value in explicit:
        if str(value) not in found:
            found.append(str(value))
    return found


def normalize_question(row: dict[str, Any], index: int) -> dict[str, Any]:
    """映射为共享 query schema，并保存分层所需 type/hops。"""
    return {
        "id": str(row.get("_id", row.get("id", f"q-{index}"))),
        "query": str(row.get("question", row.get("query", ""))),
        "gold_doc_ids": supporting_titles(row),
        "answer": row.get("answer"),
        "type": row.get("type"),
        "hops": len(row.get("question_decomposition", [])) or len(supporting_titles(row)),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以 UTF-8 JSONL 写出规范记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """解析 CLI、规范化记录、验证 evidence identity 并固化 split。"""
    parser = argparse.ArgumentParser(description="Normalize a shared-corpus QA benchmark for WARP-G")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--train-questions", required=True, type=Path)
    parser.add_argument("--dev-questions", required=True, type=Path)
    parser.add_argument("--test-questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    corpus_rows = read_records(args.corpus)
    corpus = [{
        "id": str(row.get("id", row.get("doc_id", row.get("title", row.get("idx", index))))),
        "title": str(row.get("title", "")),
        "text": str(row.get("text", row.get("passage", row.get("content", "")))),
    } for index, row in enumerate(corpus_rows)]
    ids = [row["id"] for row in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus document IDs/titles are not unique")
    splits = {
        split: [normalize_question(row, index) for index, row in enumerate(read_records(path))]
        for split, path in {
            "train": args.train_questions, "dev": args.dev_questions, "test": args.test_questions,
        }.items()
    }
    source_questions: Any = {key: str(value) for key, value in {
        "train": args.train_questions, "dev": args.dev_questions, "test": args.test_questions,
    }.items()}
    missing_gold = sorted({
        doc_id for rows in splits.values() for row in rows for doc_id in row["gold_doc_ids"]
        if doc_id not in set(ids)
    })
    if missing_gold:
        raise ValueError(f"Gold documents missing from shared corpus: {missing_gold[:10]}")
    write_jsonl(args.output_dir / "corpus.jsonl", corpus)
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    manifest = {
        "source_corpus": str(args.corpus), "source_questions": source_questions,
        "source_sha256": {
            "corpus": sha256(args.corpus),
            "train": sha256(args.train_questions),
            "dev": sha256(args.dev_questions),
            "test": sha256(args.test_questions),
        },
        "split_mode": "canonical",
        "num_documents": len(corpus), **{f"num_{key}": len(value) for key, value in splits.items()},
    }
    manifest["output_sha256"] = {
        "corpus": sha256(args.output_dir / "corpus.jsonl"),
        **{split: sha256(args.output_dir / f"{split}.jsonl") for split in splits},
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
