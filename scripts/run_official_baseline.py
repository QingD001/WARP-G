#!/usr/bin/env python3
"""在完整 1,000-query 数据上运行锁定的作者官方端到端系统。"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import string
import subprocess
import sys
import time
from typing import Any, Iterator

import yaml

from warp.data.base import load_documents, load_queries
from warp.utils import write_json


@contextmanager
def official_import(repo: Path) -> Iterator[None]:
    """临时把作者仓库置于导入路径首位，避免复制或重写其算法。"""
    sys.path.insert(0, str(repo.resolve()))
    try:
        yield
    finally:
        sys.path.pop(0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gold_answers(answer: str | list[str] | None) -> list[str]:
    if answer is None:
        raise ValueError("Official reader evaluation requires an answer for every query")
    return answer if isinstance(answer, list) else [answer]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_predictions(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        gold = gold_answers(row["gold_answer"])
        row["answer_em"] = max(answer_em(row["prediction"], value) for value in gold)
        row["answer_f1"] = max(answer_f1(row["prediction"], value) for value in gold)
    return {
        "method": method,
        "num_queries": len(rows),
        "answer_em": sum(row["answer_em"] for row in rows) / len(rows),
        "answer_f1": sum(row["answer_f1"] for row in rows) / len(rows),
    }


def run_linearrag(
    repo: Path, settings: dict[str, Any], documents: list[Any], queries: list[Any], index_dir: Path,
) -> tuple[list[dict[str, Any]], float, float]:
    with official_import(repo):
        from sentence_transformers import SentenceTransformer
        from src.config import LinearRAGConfig
        from src.LinearRAG import LinearRAG
        from src.utils import LLM_Model

        embedding = SentenceTransformer(settings["embedding_model"], device="cuda")
        config = LinearRAGConfig(
            dataset_name=index_dir.name,
            embedding_model=embedding,
            spacy_model=settings["spacy_model"],
            working_dir=str(index_dir.parent),
            llm_model=LLM_Model(settings["llm_model"]),
            max_workers=int(settings["max_workers"]),
            retrieval_top_k=int(settings["retrieval_top_k"]),
        )
        rag = LinearRAG(global_config=config)
        passages = [f"{index}:{document.content}" for index, document in enumerate(documents)]
        started = time.perf_counter()
        rag.index(passages)
        build_seconds = time.perf_counter() - started
        official_queries = [
            {"id": query.id, "question": query.text, "answer": gold_answers(query.answer)[0]}
            for query in queries
        ]
        started = time.perf_counter()
        answers = rag.qa(official_queries)
        query_seconds = time.perf_counter() - started
    rows = [
        {
            "query_id": query.id,
            "query": query.text,
            "prediction": result["pred_answer"],
            "gold_answer": query.answer,
        }
        for query, result in zip(queries, answers)
    ]
    return rows, build_seconds, query_seconds


async def run_lightrag(
    repo: Path, settings: dict[str, Any], documents: list[Any], queries: list[Any], index_dir: Path,
) -> tuple[list[dict[str, Any]], float, float]:
    with official_import(repo):
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

        rag = LightRAG(
            working_dir=str(index_dir), embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        await rag.initialize_storages()
        try:
            started = time.perf_counter()
            await rag.ainsert(
                [document.content for document in documents],
                ids=[document.id for document in documents],
                file_paths=[document.id for document in documents],
            )
            build_seconds = time.perf_counter() - started
            param = QueryParam(
                mode=settings["mode"], top_k=int(settings["top_k"]),
                chunk_top_k=int(settings["chunk_top_k"]), enable_rerank=False,
            )
            started = time.perf_counter()
            predictions = [str(await rag.aquery(query.text, param=param)) for query in queries]
            query_seconds = time.perf_counter() - started
        finally:
            await rag.finalize_storages()
    rows = [
        {
            "query_id": query.id,
            "query": query.text,
            "prediction": prediction,
            "gold_answer": query.answer,
        }
        for query, prediction in zip(queries, predictions)
    ]
    return rows, build_seconds, query_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an official end-to-end GraphRAG baseline")
    parser.add_argument("--method", required=True, choices=["linearrag", "lightrag"])
    parser.add_argument("--config", type=Path, required=True, help="WARP-G dataset YAML")
    parser.add_argument("--manifest", type=Path, default=Path("configs/official_baselines.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("external/official"))
    parser.add_argument("--index-root", type=Path, default=Path("outputs/official_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    corpus_path, query_path = Path(dataset["corpus"]), Path(dataset["queries"])
    documents, queries = load_documents(corpus_path), load_queries(query_path)
    repo = args.repo_root / args.method
    expected_commit = manifest["repositories"][args.method]["commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"Expected {args.method} commit {expected_commit}, got {actual_commit}")
    index_dir = args.index_root / args.method / dataset["name"]
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "linearrag":
        rows, build_seconds, query_seconds = run_linearrag(
            repo, manifest["linearrag"], documents, queries, index_dir,
        )
    else:
        rows, build_seconds, query_seconds = asyncio.run(run_lightrag(
            repo, manifest["lightrag"], documents, queries, index_dir,
        ))
    result = {
        "run_metadata": {
            "method": args.method,
            "official_repository": manifest["repositories"][args.method]["url"],
            "official_commit": actual_commit,
            "dataset": dataset["name"],
            "corpus_sha256": file_sha256(corpus_path),
            "queries_sha256": file_sha256(query_path),
            "num_documents": len(documents),
            "num_queries": len(queries),
            "build_wall_seconds": build_seconds,
            "query_wall_seconds": query_seconds,
            "settings": manifest[args.method],
        },
        "summary": score_predictions(args.method, rows),
        "predictions": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
