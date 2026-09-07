"""挡住 HippoRAG OpenIE 对脏 LLM 输出的崩溃，并按 chunk 落盘以便断点续抽。

官方 OpenIE 用 eval() 解析 JSON，DeepSeek 一旦写出 `...` 就会变成 Ellipsis：
NER 结果随后在 try 块外 json.dumps，直接杀死整次构图。这里不改官方图算法，
只在适配器里把实体/三元组清洗成可序列化字符串，并把已完成 chunk 立即写入
HippoRAG 的 openie_results JSON。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def sanitize_entities(values: Any) -> list[str]:
    """只保留可写成 JSON 字符串的实体；丢掉 Ellipsis/None/嵌套对象。"""
    if values is None or values is Ellipsis:
        return []
    if isinstance(values, dict) and "named_entities" in values:
        values = values["named_entities"]
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None or item is Ellipsis or isinstance(item, (list, dict, tuple, set)):
            continue
        if not isinstance(item, (str, int, float, bool)):
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def sanitize_triples(values: Any) -> list[list[str]]:
    """只保留三个原子字段的三元组，避免 len(Ellipsis) 在构图期炸掉。"""
    if values is None or values is Ellipsis:
        return []
    if isinstance(values, dict):
        for key in ("triples", "extracted_triples", "fact"):
            if key in values:
                values = values[key]
                break
    if not isinstance(values, list):
        return []
    output: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in values:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        parts: list[str] = []
        valid = True
        for item in triple:
            if item is None or item is Ellipsis or isinstance(item, (list, dict, tuple, set)):
                valid = False
                break
            if not isinstance(item, (str, int, float, bool)):
                valid = False
                break
            text = str(item).strip()
            if not text:
                valid = False
                break
            parts.append(text)
        if not valid:
            continue
        key = (parts[0], parts[1], parts[2])
        if key in seen:
            continue
        seen.add(key)
        output.append(parts)
    return output


def parse_jsonish_key(text: str, key: str) -> Any:
    """解析 LLM JSON，绝不使用 eval()。失败时返回空列表。"""
    if not text or not isinstance(text, str):
        return []
    candidates = [text]
    match = _JSON_OBJECT.search(text)
    if match is not None:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        if isinstance(payload, list):
            return payload
    return []


def install_openie_patches() -> None:
    """进程内只打一次补丁：清洗 eval 产物，并让非法三元组被跳过而不是抛错。"""
    global _PATCHED
    if _PATCHED:
        return
    from hipporag.information_extraction import openie_openai
    from hipporag.utils import llm_utils, misc_utils

    original_ner_extract = openie_openai._extract_ner_from_response
    original_filter = llm_utils.filter_invalid_triples
    original_entities = misc_utils.extract_entity_nodes
    original_flatten = misc_utils.flatten_facts

    def extract_ner(response: str) -> list[str]:
        try:
            raw = original_ner_extract(response)
        except Exception:
            raw = parse_jsonish_key(response, "named_entities")
        entities = sanitize_entities(raw)
        return entities or sanitize_entities(parse_jsonish_key(response, "named_entities"))

    def filter_triples(triples: Any) -> list[list[str]]:
        return original_filter(sanitize_triples(triples))

    def extract_nodes(chunk_triples: Any) -> Any:
        cleaned = [sanitize_triples(triples) for triples in chunk_triples]
        return original_entities(cleaned)

    def flatten(chunk_triples: Any) -> Any:
        cleaned = [sanitize_triples(triples) for triples in chunk_triples]
        return original_flatten(cleaned)

    openie_openai._extract_ner_from_response = extract_ner
    llm_utils.filter_invalid_triples = filter_triples
    misc_utils.filter_invalid_triples = filter_triples
    misc_utils.extract_entity_nodes = extract_nodes
    misc_utils.flatten_facts = flatten
    _PATCHED = True


def harden_hipporag(rag: Any) -> None:
    """给单个 HippoRAG 实例装上清洗、按 chunk 保存和检索 fallback。"""
    if getattr(rag, "_warp_openie_hardened", False):
        return
    install_openie_patches()
    _wrap_openie_instance(rag)
    _wrap_retrieval_fallback(rag)
    rag._warp_openie_hardened = True


def _wrap_openie_instance(rag: Any) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput
    from tqdm import tqdm

    openie = rag.openie
    original_ner = openie.ner
    original_triple = openie.triple_extraction
    lock = threading.Lock()

    def safe_ner(chunk_key: str, passage: str) -> Any:
        try:
            result = original_ner(chunk_key, passage)
        except Exception as exc:
            logger.warning("OpenIE NER failed for chunk %s: %s", chunk_key, exc)
            return NerRawOutput(chunk_id=chunk_key, response="", unique_entities=[], metadata={"error": str(exc)})
        result.unique_entities = sanitize_entities(getattr(result, "unique_entities", []))
        return result

    def safe_triple(chunk_key: str, passage: str, named_entities: Any) -> Any:
        entities = sanitize_entities(named_entities)
        try:
            result = original_triple(chunk_key, passage, entities)
        except Exception as exc:
            logger.warning("OpenIE triples failed for chunk %s: %s", chunk_key, exc)
            return TripleRawOutput(chunk_id=chunk_key, response="", metadata={"error": str(exc)}, triples=[])
        result.triples = sanitize_triples(getattr(result, "triples", []))
        return result

    def persist_chunk(chunk_key: str, row: dict[str, Any], ner: Any, triples: Any) -> None:
        with lock:
            all_info, _ = rag.load_existing_openie([])
            if any(str(item.get("idx")) == str(chunk_key) for item in all_info):
                return
            rag.merge_openie_results(
                all_info, {chunk_key: row}, {chunk_key: ner}, {chunk_key: triples},
            )
            rag.save_openie_results(all_info)

    def safe_batch(chunks: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not chunks:
            return {}, {}
        ner_results: dict[str, Any] = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(safe_ner, chunk_key, row["content"]): chunk_key
                for chunk_key, row in chunks.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="NER"):
                result = future.result()
                ner_results[result.chunk_id] = result

        triple_results: dict[str, Any] = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    safe_triple, ner.chunk_id, chunks[ner.chunk_id]["content"], ner.unique_entities,
                ): ner.chunk_id
                for ner in ner_results.values()
            }
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting triples"):
                result = future.result()
                triple_results[result.chunk_id] = result
                persist_chunk(
                    result.chunk_id, chunks[result.chunk_id],
                    ner_results[result.chunk_id], result,
                )
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(futures):
                    print(json.dumps({
                        "openie_checkpoint": "chunk",
                        "saved": completed,
                        "total": len(futures),
                        "path": rag.openie_results_path,
                    }, ensure_ascii=False), flush=True)
        missing = set(chunks) - set(ner_results)
        for chunk_key in missing:
            ner_results[chunk_key] = NerRawOutput(
                chunk_id=chunk_key, response="", unique_entities=[], metadata={"error": "missing"},
            )
        missing = set(chunks) - set(triple_results)
        for chunk_key in missing:
            triple_results[chunk_key] = TripleRawOutput(
                chunk_id=chunk_key, response="", metadata={"error": "missing"}, triples=[],
            )
        return ner_results, triple_results

    openie.ner = safe_ner
    openie.triple_extraction = safe_triple
    openie.batch_openie = safe_batch


def _wrap_retrieval_fallback(rag: Any) -> None:
    original = rag.graph_search_with_fact_entities

    def wrapped(query: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(query, *args, **kwargs)
        except (AssertionError, KeyError) as exc:
            logger.warning("HippoRAG PPR failed for a query (%s); falling back to dense retrieval", exc)
            return rag.dense_passage_retrieval(query)

    rag.graph_search_with_fact_entities = wrapped
