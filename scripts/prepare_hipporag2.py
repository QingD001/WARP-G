#!/usr/bin/env python3
"""Convert HippoRAG2's released corpus/query pairs into WARP-G splits.

The upstream release contains one 1,000-query evaluation file per dataset. This
script preserves the complete query set. The formal runner assigns deterministic
held-out folds so every released query is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATASETS = {
    "hotpotqa": ("hotpotqa", "hotpotqa"),
    "2wiki": ("2wikimultihopqa", "2wiki"),
    "musique": ("musique", "musique"),
    "popqa": ("popqa", "popqa"),
}
SPACE_RE = re.compile(r"\s+")


def _read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return value


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(map(str, value))
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_id(title: str, text: str) -> str:
    digest = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:20]
    return f"doc-{digest}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        value = decoded
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item is not None and str(item).strip()]


def _answers(row: dict[str, Any]) -> str | list[str] | None:
    if row.get("answer") is not None:
        value = row["answer"]
        return [str(item) for item in value] if isinstance(value, list) else str(value)
    values: list[str] = []
    for key in ("obj", "o_wiki_title", "possible_answers", "o_aliases", "answer_aliases"):
        for value in _parse_list(row.get(key)):
            if value not in values:
                values.append(value)
    return values or None


def _local_passages(row: dict[str, Any]) -> list[tuple[str, str, bool]]:
    supporting_titles = {
        str(item[0] if isinstance(item, (list, tuple)) else item.get("title", item.get("doc_id")))
        for item in row.get("supporting_facts", row.get("supporting_docs", []))
        if item is not None
    }
    passages: list[tuple[str, str, bool]] = []
    for item in row.get("context", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, text = str(item[0]), _text(item[1])
            passages.append((title, text, title in supporting_titles))
    for item in row.get("paragraphs", row.get("contexts", [])):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", item.get("doc_id", item.get("idx", ""))))
        text = _text(item.get("text", item.get("paragraph_text", item.get("content", ""))))
        passages.append((title, text, bool(item.get("is_supporting", False))))
    return passages


def _normalize_dataset(
    source_name: str, corpus_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]], seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    exact: dict[tuple[str, str], str] = {}
    by_title: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for row in corpus_rows:
        title = _text(row.get("title", ""))
        text = _text(row.get("text", row.get("passage", row.get("content", ""))))
        if not text:
            raise ValueError(f"{source_name}: corpus contains an empty passage")
        doc_id = _document_id(title, text)
        content = f"{title}\n{text}".strip()
        if content in seen_content:
            # The released MuSiQue/PopQA corpora contain repeated copies of a
            # handful of identical passages.  WARP/HippoRAG require unique
            # content, and all such copies have the same evidence identity.
            continue
        if doc_id in seen_ids:
            raise RuntimeError(f"{source_name}: document hash collision for {title!r}")
        seen_ids.add(doc_id)
        seen_content.add(content)
        exact[(title, text)] = doc_id
        by_title.setdefault(title, []).append(doc_id)
        documents.append({"id": doc_id, "title": title, "text": text})

    queries: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for index, row in enumerate(query_rows):
        query_id = str(row.get("id", row.get("_id", f"{source_name}-{index}")))
        if query_id in seen_queries:
            raise ValueError(f"{source_name}: duplicate query ID {query_id!r}")
        seen_queries.add(query_id)
        gold: list[str] = []
        for title, text, is_supporting in _local_passages(row):
            if not is_supporting:
                continue
            doc_id = exact.get((title, text))
            if doc_id is None:
                candidates = by_title.get(title, [])
                if len(candidates) == 1:
                    doc_id = candidates[0]
                else:
                    raise ValueError(
                        f"{source_name}: cannot uniquely map supporting passage {title!r} "
                        f"for query {query_id!r}"
                    )
            if doc_id not in gold:
                gold.append(doc_id)
        if not gold:
            raise ValueError(f"{source_name}: query {query_id!r} has no supporting passages")
        question = _text(row.get("question", row.get("query", "")))
        if not question:
            raise ValueError(f"{source_name}: query {query_id!r} has empty text")
        queries.append({
            "id": query_id,
            "query": question,
            "gold_doc_ids": gold,
            "answer": _answers(row),
            "source_dataset": source_name,
        })

    # This is the exact stable ordering used by the formal fold assignment.
    queries.sort(key=lambda row: (
        hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"],
    ))
    return documents, queries


def _convert_one(
    public_name: str, source_stem: str, input_dir: Path, output_root: Path,
    seed: int, folds: int,
) -> dict[str, Any]:
    corpus_path = input_dir / f"{source_stem}_corpus.json"
    query_path = input_dir / f"{source_stem}.json"
    documents, queries = _normalize_dataset(
        public_name, _read_list(corpus_path), _read_list(query_path), seed,
    )
    output_dir = output_root / public_name
    _write_jsonl(output_dir / "corpus.jsonl", documents)
    _write_jsonl(output_dir / "queries.jsonl", queries)
    fold_ids = [
        [row["id"] for index, row in enumerate(queries) if index % folds == fold]
        for fold in range(folds)
    ]
    manifest = {
        "source": "osunlp/HippoRAG_2",
        "source_files": {"corpus": str(corpus_path), "queries": str(query_path)},
        "source_sha256": {"corpus": _sha256(corpus_path), "queries": _sha256(query_path)},
        "evaluation_protocol": "deterministic_cross_fitting",
        "seed": seed,
        "folds": folds,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "fold_test_query_ids": fold_ids,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "source": manifest["source"],
        "evaluation_protocol": manifest["evaluation_protocol"],
        "seed": seed,
        "folds": folds,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "fold_test_sizes": [len(query_ids) for query_ids in fold_ids],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official HippoRAG2 release for WARP-G")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/hipporag2"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    reports = []
    for public_name in args.datasets:
        source_stem, output_name = DATASETS[public_name]
        reports.append(_convert_one(
            output_name, source_stem, args.input_dir, args.output_root,
            args.seed, args.folds,
        ))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
