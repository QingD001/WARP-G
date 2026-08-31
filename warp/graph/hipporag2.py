"""官方 HippoRAG2 的区域物化、检索和成本审计适配器。

本模块不重写任何 HippoRAG2 图算法；它只负责把 WARP Region 映射为互相隔离的
官方 index，并把官方检索结果映射回稳定的 WARP document ID。Selective Graph、
Graph baseline 和 Full Graph 因而共享完全相同的 OpenIE/embedding/PPR 后端。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from warp.models import ConstructionCost, Document, Region, SearchResult
from warp.utils import tokenize, write_json
from .builder import RegionalGraph


UPSTREAM_REPOSITORY = "https://github.com/OSU-NLP-Group/HippoRAG"
SUPPORTED_API_VERSION = "2.0.0a4"
UPSTREAM_COMMIT = "c617143f01477243992a63b2e2151cc003dd3b21"


def _installed_upstream_commit() -> str | None:
    """从 pip direct_url.json 读取安装来源并解析 git commit；无法确定时返回 None。

    支持两种安装方式：(1) `pip install git+https://...@<commit>` 的 VCS 安装，
    direct_url.json 直接记录 vcs_info.commit_id；(2) 本地 `git clone` 后
    `pip install -e ./HippoRAG` 的目录安装，direct_url.json 记录 dir_info.url，
    再从该目录的 .git 读取 HEAD。
    """
    import subprocess
    try:
        payload = importlib.metadata.distribution("hipporag").read_text("direct_url.json")
        info = json.loads(payload or "{}")
    except Exception:
        return None
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return str(commit)
    local_dir = (info.get("dir_info") or {}).get("url")
    if local_dir:
        parsed = urllib.parse.urlparse(str(local_dir))
        local_path = urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else str(local_dir)
        try:
            result = subprocess.run(
                ["git", "-C", local_path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            return None
    return None


def _validate_commit(version: str) -> None:
    """版本与锁定 commit 双重校验；任何不一致直接失败。"""
    if version != SUPPORTED_API_VERSION:
        raise RuntimeError(f"Expected hipporag=={SUPPORTED_API_VERSION}, found {version}")
    installed = _installed_upstream_commit()
    if installed is None:
        raise RuntimeError(
            "Cannot verify the installed HippoRAG git commit (pip direct_url.json unavailable). "
            "Reinstall through the locked git dependency, e.g. `pip install -e .`"
        )
    if installed != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected HippoRAG commit {UPSTREAM_COMMIT}, found {installed}. "
            "Reinstall the project dependency locked to the pinned upstream commit."
        )


@dataclass(slots=True)
class HippoRAG2Config:
    """Paper backend settings mirroring HippoRAG 2's official experiment runner."""

    artifact_root: str = "outputs/hipporag2"
    dataset: str | None = None
    llm_name: str = "gpt-4o-mini"
    llm_base_url: str | None = "https://api.openai.com/v1"
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    embedding_base_url: str | None = None
    azure_endpoint: str | None = None
    azure_embedding_endpoint: str | None = None
    embedding_batch_size: int = 8
    retrieval_top_k: int = 200
    linking_top_k: int = 5
    qa_top_k: int = 5
    damping: float = 0.5
    passage_node_weight: float = 0.05
    synonymy_edge_topk: int = 2047
    synonymy_edge_sim_threshold: float = 0.8
    synonymy_edge_query_batch_size: int = 1000
    synonymy_edge_key_batch_size: int = 10000
    graph_type: str = "facts_and_sim_passage_node_unidirectional"
    vector_store_type: str = "parquet"
    force_index_from_scratch: bool = False
    force_openie_from_scratch: bool = False
    reuse_artifacts: bool = True
    input_usd_per_million_tokens: float = 0.15
    output_usd_per_million_tokens: float = 0.60
    seed: int = 42


class _UsageTracker:
    """Counts physical and logical tokens returned by HippoRAG's LLM wrapper."""

    def __init__(self) -> None:
        self.phase = "idle"
        self.values: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add(self, metadata: dict[str, Any], cache_hit: bool) -> None:
        """同时累计逻辑 cold cost 与真正发生 API 请求的 physical cost。"""
        missing = {"prompt_tokens", "completion_tokens"} - set(metadata)
        if missing:
            raise RuntimeError(f"HippoRAG LLM usage metadata is missing fields: {sorted(missing)}")
        prompt = int(metadata["prompt_tokens"])
        completion = int(metadata["completion_tokens"])
        with self._lock:
            row = self.values.setdefault(self.phase, {
                "logical_input_tokens": 0, "logical_output_tokens": 0,
                "physical_input_tokens": 0, "physical_output_tokens": 0,
                "calls": 0, "cache_hits": 0,
            })
            row["logical_input_tokens"] += prompt
            row["logical_output_tokens"] += completion
            row["calls"] += 1
            row["cache_hits"] += int(cache_hit)
            if not cache_hit:
                row["physical_input_tokens"] += prompt
                row["physical_output_tokens"] += completion

    def get(self, phase: str) -> dict[str, int]:
        """返回指定阶段计数快照，调用方用前后差得到单次成本。"""
        return dict(self.values.get(phase, {}))


class HippoRAG2GraphBuilder:
    """Regional adapter around the official HippoRAG 2.0 indexing implementation.

    No graph logic is reimplemented here. Every region owns an isolated official
    HippoRAG index, while WARP remains responsible only for deciding where to build.
    """

    def __init__(self, config: HippoRAG2Config) -> None:
        self.config = config
        self._shared_llm: Any = None
        self._shared_embedding_model: Any = None
        self._usage_tracker: _UsageTracker | None = None

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, str]:
        try:
            from hipporag import Chunk, HippoRAG
            from hipporag.utils.config_utils import BaseConfig
        except ImportError as exc:
            raise ImportError(
                "HippoRAG 2 is required. Install the project with `pip install -e .`."
            ) from exc
        version = importlib.metadata.version("hipporag")
        return HippoRAG, Chunk, BaseConfig, version

    @staticmethod
    def _validate_upstream_api(HippoRAG: Any, Chunk: Any) -> None:
        hippo_parameters = inspect.signature(HippoRAG.__init__).parameters
        required_constructor = {"extraction_llm", "qa_llm", "embedding_model", "text_preprocessor"}
        missing_constructor = required_constructor - set(hippo_parameters)
        chunk_fields = set(getattr(Chunk, "__dataclass_fields__", {}))
        missing_chunk = {"content", "source_id", "metadata"} - chunk_fields
        if missing_constructor or missing_chunk:
            raise RuntimeError(
                "Installed HippoRAG does not expose the pinned regional adapter API. "
                f"Missing constructor parameters={sorted(missing_constructor)}, "
                f"Chunk fields={sorted(missing_chunk)}. Reinstall the project dependency locked "
                f"to upstream commit {UPSTREAM_COMMIT}."
            )

    def estimate_cost(self, region: Region, documents: list[Document]) -> float:
        """正式构图前同样使用区域文本 token proxy 做 matched budget。"""
        selected = set(region.doc_ids)
        return float(sum(len(tokenize(doc.content)) for doc in documents if doc.id in selected))

    def _fingerprint(self, region: Region, selected: list[Document], version: str) -> str:
        payload = {
            "region": region.id,
            "documents": [(doc.id, hashlib.sha256(doc.content.encode()).hexdigest()) for doc in selected],
            "config": asdict(self.config),
            "hipporag_version": version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _artifact_dir(self, region: Region, selected: list[Document], version: str) -> Path:
        safe_region = "".join(char if char.isalnum() or char in "-_" else "_" for char in region.id)
        return Path(self.config.artifact_root) / "regions" / f"{safe_region}-{self._fingerprint(region, selected, version)}"

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _token_count(texts: list[str]) -> int:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(text)) for text in texts)

    @staticmethod
    def _wrap_llm(llm: Any) -> _UsageTracker:
        tracker = _UsageTracker()
        original = llm.infer

        def counted_infer(*args: Any, **kwargs: Any) -> Any:
            """透明转发 infer，并从官方 metadata/cache flag 读取 token。"""
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) < 3 or not isinstance(result[1], dict):
                raise RuntimeError(
                    "HippoRAG LLM infer must return (response, usage_metadata, cache_hit)"
                )
            tracker.add(result[1], bool(result[2]))
            return result

        llm.infer = counted_infer
        return tracker

    def _ensure_shared_resources(self, BaseConfig: Any) -> None:
        """延迟创建并跨区域共享昂贵的 LLM 与 embedding 模型权重。"""
        if self._shared_llm is not None and self._shared_embedding_model is not None and self._usage_tracker is not None:
            return
        from hipporag.llm import _get_llm_class
        from hipporag.embedding_model import _get_embedding_model_class

        shared_dir = Path(self.config.artifact_root) / "_shared_models"
        shared_dir.mkdir(parents=True, exist_ok=True)
        shared_config = self._base_config(BaseConfig, shared_dir)
        if self._shared_llm is None:
            self._shared_llm = _get_llm_class(shared_config)
        if self._shared_embedding_model is None:
            embedding_class = _get_embedding_model_class(self.config.embedding_model_name)
            self._shared_embedding_model = embedding_class(
                global_config=shared_config, embedding_model_name=self.config.embedding_model_name,
            )
        if self._usage_tracker is None:
            self._usage_tracker = self._wrap_llm(self._shared_llm)

    def passage_encoder(self) -> Any:
        """Return the exact shared HippoRAG embedding model for the base Dense index."""
        HippoRAG, Chunk, BaseConfig, version = self._imports()
        self._validate_upstream_api(HippoRAG, Chunk)
        _validate_commit(version)
        self._ensure_shared_resources(BaseConfig)
        return _HippoRAGPassageEncoder(self._shared_embedding_model)

    def build_full_graph(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """Measure Full Graph with an isolated LLM cache but shared GPU encoder weights."""
        child_config = replace(
            self.config,
            artifact_root=str(Path(self.config.artifact_root) / "full_graph_baseline"),
        )
        child = type(self)(child_config)
        child._shared_embedding_model = self._shared_embedding_model
        return child.build(region, documents)

    @staticmethod
    def _usage_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
        return {key: after.get(key, 0) - before.get(key, 0) for key in set(after) | set(before)}

    def _base_config(self, BaseConfig: Any, artifact_dir: Path) -> Any:
        """把 WARP 配置逐项映射到锁定版本的官方 BaseConfig。"""
        dataset = self.config.dataset
        if dataset == "2wiki":
            dataset = "2wikimultihopqa"
        supported_datasets = {"hotpotqa", "hotpotqa_train", "musique", "2wikimultihopqa"}
        return BaseConfig(
            save_dir=str(artifact_dir), dataset=dataset if dataset in supported_datasets else None,
            llm_name=self.config.llm_name, llm_base_url=self.config.llm_base_url,
            embedding_model_name=self.config.embedding_model_name,
            embedding_base_url=self.config.embedding_base_url,
            azure_endpoint=self.config.azure_endpoint,
            azure_embedding_endpoint=self.config.azure_embedding_endpoint,
            embedding_batch_size=self.config.embedding_batch_size,
            retrieval_top_k=self.config.retrieval_top_k,
            linking_top_k=self.config.linking_top_k, qa_top_k=self.config.qa_top_k,
            damping=self.config.damping, passage_node_weight=self.config.passage_node_weight,
            synonymy_edge_topk=self.config.synonymy_edge_topk,
            synonymy_edge_sim_threshold=self.config.synonymy_edge_sim_threshold,
            synonymy_edge_query_batch_size=self.config.synonymy_edge_query_batch_size,
            synonymy_edge_key_batch_size=self.config.synonymy_edge_key_batch_size,
            graph_type=self.config.graph_type, vector_store_type=self.config.vector_store_type,
            openie_mode="online",
            force_index_from_scratch=self.config.force_index_from_scratch,
            force_openie_from_scratch=self.config.force_openie_from_scratch,
            seed=self.config.seed,
        )

    def build(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """在独立 artifact 目录中调用官方 `HippoRAG.index(List[Chunk])`。"""
        HippoRAG, Chunk, BaseConfig, version = self._imports()
        self._validate_upstream_api(HippoRAG, Chunk)
        _validate_commit(version)
        by_id = {doc.id: doc for doc in documents}
        missing = [doc_id for doc_id in region.doc_ids if doc_id not in by_id]
        if missing:
            raise KeyError(f"Region {region.id} refers to unknown documents: {missing[:5]}")
        selected = [by_id[doc_id] for doc_id in region.doc_ids]
        # 官方 chunk store 以内容哈希为主键；重复文本会破坏 doc-level evidence ID。
        contents: dict[str, str] = {}
        for doc in selected:
            previous = contents.get(doc.content)
            if previous is not None and previous != doc.id:
                raise ValueError(
                    f"HippoRAG hashes chunks by content, so duplicate content for {previous!r} and "
                    f"{doc.id!r} cannot preserve document-level evaluation identity. Deduplicate the corpus first."
                )
            contents[doc.content] = doc.id
        artifact_dir = self._artifact_dir(region, selected, version)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "warp_hipporag2_manifest.json"

        config = self._base_config(BaseConfig, artifact_dir)
        self._ensure_shared_resources(BaseConfig)
        tracker = self._usage_tracker
        # 图和 vector stores 每个 region 独立，模型权重可以安全共享。
        rag = HippoRAG(
            global_config=config, extraction_llm=self._shared_llm, qa_llm=self._shared_llm,
            embedding_model=self._shared_embedding_model,
        )
        rag.openie.llm_model = self._shared_llm
        rag.rerank_filter.llm_infer_fn = self._shared_llm.infer
        rag._warp_usage_tracker = tracker
        started = time.perf_counter()
        before_usage = tracker.get("construction")
        tracker.phase = "construction"
        chunks = [Chunk(content=doc.content, source_id=doc.id,
                        metadata={"warp_doc_id": doc.id, "title": doc.title}) for doc in selected]
        try:
            rag.index(chunks)
        finally:
            wall_seconds = time.perf_counter() - started
            tracker.phase = "idle"

        graph_info = rag.get_graph_info()
        fact_count = int(graph_info.get("num_extracted_triples", 0))
        if selected and fact_count == 0:
            raise RuntimeError(
                f"HippoRAG 2 extracted zero facts for region {region.id}. "
                "Check the LLM endpoint and OpenIE output."
            )

        usage = self._usage_delta(tracker.get("construction"), before_usage)
        # embedding API 未统一返回 token usage，因此对实际写入三套 store 的文本计数。
        embedding_texts: list[str] = []
        for store in (rag.chunk_embedding_store, rag.entity_embedding_store, rag.fact_embedding_store):
            embedding_texts.extend(list(store.get_all_texts()))
        cost = ConstructionCost(
            # Logical usage gives a stable cold-construction cost even when the
            # shared upstream response cache avoids duplicate paid calls.
            input_tokens=usage.get("logical_input_tokens", 0),
            output_tokens=usage.get("logical_output_tokens", 0),
            embedding_tokens=self._token_count(embedding_texts),
            wall_seconds=wall_seconds,
            nodes=int(rag.graph.vcount()), edges=int(rag.graph.ecount()),
            storage_bytes=self._directory_size(artifact_dir),
            estimated_usd=(
                usage.get("logical_input_tokens", 0) * self.config.input_usd_per_million_tokens
                + usage.get("logical_output_tokens", 0) * self.config.output_usd_per_million_tokens
            ) / 1_000_000,
        )
        # Reused official caches represent the same original physical construction cost.
        if manifest_path.exists() and self.config.reuse_artifacts and not self.config.force_index_from_scratch:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))["construction_cost"]
            cost = ConstructionCost(**stored)
        metadata = {
            "builder": type(self).__name__, "backend": "official-hipporag2",
            "hipporag_version": version, "supported_api_version": SUPPORTED_API_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_repository": UPSTREAM_REPOSITORY, "artifact_dir": str(artifact_dir),
            "graph_info": graph_info, "construction_llm_usage": usage,
            "embedding_tokens_are_tokenizer_estimate": True,
        }
        write_json(manifest_path, {
            "region_id": region.id, "doc_ids": region.doc_ids,
            "configuration": asdict(self.config), "construction_cost": cost.to_dict(),
            "metadata": metadata,
        })
        return RegionalGraph(
            region_id=region.id,
            doc_ids=list(region.doc_ids),
            cost=cost,
            metadata=metadata,
            backend=rag,
        )


class HippoRAG2GraphRetriever:
    """Retrieval adapter calling the official fact filtering + PPR pipeline."""

    def __init__(self) -> None:
        self.logical_input_tokens = 0
        self.logical_output_tokens = 0
        self.physical_input_tokens = 0
        self.physical_output_tokens = 0
        self.wall_seconds = 0.0
        self.calls = 0

    def search(self, query: str, graph: RegionalGraph, k: int = 10) -> list[SearchResult]:
        """直接运行官方 query-to-fact、fact filter、personalization 与 PPR。"""
        rag = graph.backend
        if rag is None:
            raise TypeError("HippoRAG2GraphRetriever requires a graph built by HippoRAG2GraphBuilder")
        tracker = getattr(rag, "_warp_usage_tracker", None)
        before = tracker.get("retrieval") if tracker is not None else {}
        if tracker is not None:
            tracker.phase = "retrieval"
        started = time.perf_counter()
        try:
            solutions = rag.retrieve(queries=[query], num_to_retrieve=k)
        finally:
            elapsed = time.perf_counter() - started
            if tracker is not None:
                tracker.phase = "idle"
                after = tracker.get("retrieval")
                self.logical_input_tokens += after.get("logical_input_tokens", 0) - before.get("logical_input_tokens", 0)
                self.logical_output_tokens += after.get("logical_output_tokens", 0) - before.get("logical_output_tokens", 0)
                self.physical_input_tokens += after.get("physical_input_tokens", 0) - before.get("physical_input_tokens", 0)
                self.physical_output_tokens += after.get("physical_output_tokens", 0) - before.get("physical_output_tokens", 0)
        self.wall_seconds += elapsed
        self.calls += 1
        if isinstance(solutions, tuple):
            solutions = solutions[0]
        solution = solutions[0]
        metadata = solution.doc_metadata
        if metadata is None or len(metadata) != len(solution.docs):
            raise RuntimeError("HippoRAG retrieval must return metadata for every document")
        scores = solution.doc_scores.tolist() if hasattr(solution.doc_scores, "tolist") else list(solution.doc_scores)
        if len(scores) != len(solution.docs):
            raise RuntimeError("HippoRAG retrieval must return one score per document")
        results: list[SearchResult] = []
        allowed = set(graph.doc_ids)
        for rank, (score, row) in enumerate(zip(scores, metadata), 1):
            doc_id = row.get("source_id")
            if doc_id is None:
                raise RuntimeError("HippoRAG result metadata is missing the required source_id")
            if str(doc_id) not in allowed:
                raise RuntimeError(
                    f"HippoRAG region {graph.region_id} returned out-of-region document {doc_id!r}"
                )
            results.append(SearchResult(str(doc_id), float(score), "hipporag2", rank, graph.region_id))
        return results

    def stats(self) -> dict[str, float | int]:
        """汇总整个实验期间官方图在线检索的 token 与 wall time。"""
        return {
            "calls": self.calls, "wall_seconds": self.wall_seconds,
            "logical_input_tokens": self.logical_input_tokens,
            "logical_output_tokens": self.logical_output_tokens,
            "physical_input_tokens": self.physical_input_tokens,
            "physical_output_tokens": self.physical_output_tokens,
        }

    def delta(self, before: dict[str, float | int]) -> dict[str, float | int]:
        """返回一次方法评测独立产生的在线图检索成本。"""
        after = self.stats()
        return {key: after[key] - before[key] for key in after}


class _HippoRAGPassageEncoder:
    """让全局 Dense baseline 复用 HippoRAG2 的同一个 encoder 和 query instruction。"""
    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    def _rows(values: Any) -> list[list[float]]:
        return values.tolist() if hasattr(values, "tolist") else [list(row) for row in values]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """兼容通用 encoder 接口，默认按 passage 编码。"""
        return self.encode_documents(texts)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """使用官方 embedding 模型无 query instruction 编码 passage。"""
        return self._rows(self.model.batch_encode(texts, instruction="", norm=True))

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """使用官方 query_to_passage instruction 编码 query。"""
        from hipporag.prompts.linking import get_query_instruction
        return self._rows(self.model.batch_encode(
            texts, instruction=get_query_instruction("query_to_passage"), norm=True,
        ))
