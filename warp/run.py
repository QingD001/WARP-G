"""从 YAML 运行完整论文实验并输出自描述 JSON artifact。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import torch

from warp.baselines import GlobalBaselineFactory
from warp.data import load_crossfit_bundles
from warp.eval.construction_cost import aggregate_costs, attach_token_efficiency, consumed_tokens, usage_to_cost
from warp.eval.reader import evaluate_hipporag2_reader
from warp.eval.retrieval import evaluate_retrieval
from warp.eval.statistics import paired_bootstrap_interval, paired_randomization_pvalue
from warp.graph import HippoRAG2Config, HippoRAG2GraphBuilder, HippoRAG2GraphRetriever
from warp.graph.hipporag2 import SUPPORTED_API_VERSION, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from warp.pipeline import WARPConfig, WARPG
from warp.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever, fuse_and_rerank
from warp.utils import write_json


METRICS = ("evidence_recall@5", "evidence_recall@10", "complete_evidence@5", "complete_evidence@10")
TOKEN_EFFICIENCY_KEYS = (
    "total_tokens_excluding_design", "total_tokens_including_design",
    "token_efficiency_excluding_design", "token_efficiency_including_design",
)
REGIONAL_METHODS = {"warp", "random_region", "frequency_only", "gain_only", "cost_only"}
GLOBAL_METHODS = {"ket_rag", "g2cons"}
BASE_RETRIEVAL_METHODS = {"bm25", "dense", "hybrid"}
GRAPH_BASELINE_METHODS = {"hipporag2", "full_graph"}
_METHOD_ORDER = (
    "bm25", "dense", "hybrid",
    "warp", "random_region", "frequency_only", "gain_only", "cost_only",
    "ket_rag", "g2cons", "hipporag2", "full_graph",
)


def _ordered_methods(methods: list[str]) -> list[str]:
    """按 design.md 的比较顺序执行：先 WARP selection，再只换公式的 controls，最后独立 pipeline。"""
    remaining = list(methods)
    ordered: list[str] = []
    for name in _METHOD_ORDER:
        if name in remaining:
            ordered.append(name)
            remaining.remove(name)
    return ordered + remaining


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Experiment config must be a non-empty mapping: {path}")
    return value


def reproducibility_metadata(config: dict[str, Any]) -> dict[str, Any]:
    packages = [
        "warp-g", "hipporag", "numpy", "torch", "sentence-transformers", "lightgbm",
        "igraph", "leidenalg", "faiss-cpu", "datasets", "tiktoken", "PyYAML",
    ]
    if not torch.cuda.is_available():
        raise RuntimeError("The formal experiment requires a CUDA device")
    dataset_paths = {key: Path(value) for key, value in config["dataset"].items()
                     if key in {"corpus", "queries"}}
    data_hashes = {}
    for key, path in dataset_paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        data_hashes[key] = digest.hexdigest()
    device = torch.cuda.current_device()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "configuration": config,
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "data_sha256": data_hashes,
        "cuda": {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "hipporag2_upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "validated_api_version": SUPPORTED_API_VERSION,
        },
    }


def _build_model(
    config: dict[str, Any], bundle: Any, design_seed: int, partition_mode: str | None = None,
) -> tuple[WARPG, HippoRAG2GraphRetriever]:
    graph_config = dict(config["graph"])
    graph_config["dataset"] = config["dataset"]["name"]
    graph_config["seed"] = design_seed
    builder = HippoRAG2GraphBuilder(HippoRAG2Config(**graph_config))
    graph_retriever = HippoRAG2GraphRetriever()
    dense = DenseRetriever(
        encoder=builder.passage_encoder(), model_name=f"hipporag:{builder.config.embedding_model_name}",
    )
    base = HybridRetriever(BM25Retriever(), dense)
    reranker = CrossEncoderReranker(bundle.documents, **config["reranker"])
    warp_config = replace(WARPConfig(**config["warp"]), seed=design_seed)
    if partition_mode is not None:
        warp_config = replace(warp_config, partition_mode=partition_mode)
    return WARPG(warp_config, builder, graph_retriever, reranker, base).fit(bundle), graph_retriever


def _summarize(
    rows: list[dict[str, Any]], keys: tuple[str, ...], metric_names: tuple[str, ...] = METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["trials"] = len(values)
        for metric in metric_names:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        if "actual_cost_fraction" in values[0]:
            samples = [float(row["actual_cost_fraction"]) for row in values]
            item["actual_cost_fraction_mean"] = statistics.fmean(samples)
            item["actual_cost_fraction_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        for key in TOKEN_EFFICIENCY_KEYS:
            if key not in values[0] or values[0][key] is None:
                continue
            samples = [row[key] for row in values if row.get(key) is not None]
            if not samples:
                continue
            item[f"{key}_mean"] = statistics.fmean(samples)
            item[f"{key}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def _crossfit_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...], seed: int,
    metric_names: tuple[str, ...] = METRICS,
) -> list[dict[str, Any]]:
    """Merge disjoint held-out folds and compute final query-level statistics."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        per_query = {
            query_id: metrics
            for row in values for query_id, metrics in row["per_query"].items()
        }
        if len(per_query) != sum(len(row["per_query"]) for row in values):
            raise ValueError("A query appeared in more than one held-out fold")
        item: dict[str, Any] = {
            **{key: value for key, value in zip(keys, group_key)},
            "folds": len(values), "num_queries": len(per_query),
        }
        for metric in metric_names:
            samples = [float(metrics[metric]) for metrics in per_query.values()]
            lower, upper = paired_bootstrap_interval(samples, seed=seed)
            item[metric] = statistics.fmean(samples)
            item[f"{metric}_ci95"] = [lower, upper]
        if "actual_cost_fraction" in values[0]:
            costs = [float(row["actual_cost_fraction"]) for row in values]
            item["actual_cost_fraction_mean"] = statistics.fmean(costs)
            item["actual_cost_fraction_std"] = statistics.stdev(costs) if len(costs) > 1 else 0.0
        for key in TOKEN_EFFICIENCY_KEYS:
            samples = [row[key] for row in values if row.get(key) is not None]
            if not samples:
                continue
            item[key] = statistics.fmean(samples)
            item[f"{key}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def _significance(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    references = sorted({str(row["method"]) for row in rows if row["method"] != "warp"})
    candidate_rows = [row for row in rows if row["method"] == "warp"]
    candidate_per_query = {
        query_id: values
        for row in candidate_rows for query_id, values in row["per_query"].items()
    }
    for reference_name in references:
        reference_rows = [row for row in rows if row["method"] == reference_name]
        reference_per_query = {
            query_id: values
            for row in reference_rows for query_id, values in row["per_query"].items()
        }
        if not candidate_per_query or not reference_per_query:
            continue
        query_ids = sorted(candidate_per_query)
        if set(query_ids) != set(reference_per_query):
            raise ValueError("Cross-fit significance requires identical held-out query coverage")
        for metric in METRICS:
            candidate = [candidate_per_query[query_id][metric] for query_id in query_ids]
            baseline = [reference_per_query[query_id][metric] for query_id in query_ids]
            output.append({
                "design_seed": seed,
                "cross_fitting_folds": len(candidate_rows),
                "num_queries": len(query_ids),
                "candidate": "warp",
                "reference": reference_name,
                "metric": metric,
                "mean_difference": statistics.fmean(left - right for left, right in zip(candidate, baseline)),
                "paired_randomization_pvalue": paired_randomization_pvalue(
                    candidate, baseline, samples=samples, seed=seed,
                ),
            })
    for metric in METRICS:
        group = [row for row in output if row["metric"] == metric]
        ordered = sorted(group, key=lambda row: float(row["paired_randomization_pvalue"]))
        running = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * float(row["paired_randomization_pvalue"]))
            running = max(running, adjusted)
            row["holm_adjusted_pvalue"] = running
    return output


def _quality_cost_auc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Integrate each method's mean quality over its measured deployment-cost curve."""
    eligible = [row for row in rows if "actual_cost_fraction" in row]
    if not eligible:
        return []
    methods = sorted({str(row["method"]) for row in eligible})
    output: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in eligible if row["method"] == method]
        budget_points: list[tuple[float, dict[str, float]]] = []
        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in method_rows:
            groups.setdefault(row.get("budget_fraction"), []).append(row)
        for trials in groups.values():
            cost = statistics.fmean(float(row["actual_cost_fraction"]) for row in trials)
            metrics = {metric: statistics.fmean(float(row[metric]) for row in trials) for metric in METRICS}
            efficiency = [
                row["token_efficiency_excluding_design"]
                for row in trials if row.get("token_efficiency_excluding_design") is not None
            ]
            if efficiency:
                metrics["token_efficiency_excluding_design"] = statistics.fmean(efficiency)
            budget_points.append((cost, metrics))
        by_cost: dict[float, list[dict[str, float]]] = {}
        for cost, metrics in budget_points:
            by_cost.setdefault(cost, []).append(metrics)
        points = [
            (cost, {metric: statistics.fmean(item[metric] for item in values if metric in item)
                    for metric in METRICS if any(metric in item for item in values)})
            for cost, values in sorted(by_cost.items())
        ]
        item: dict[str, Any] = {
            "method": method,
            "cost_axis": "actual_cost_fraction",
            "minimum_cost_fraction": points[0][0],
            "maximum_cost_fraction": points[-1][0],
            "points": [{"actual_cost_fraction": cost, **metrics} for cost, metrics in points],
        }
        for metric in METRICS:
            if len(points) < 2:
                item[f"{metric}_auc"] = None
                continue
            item[f"{metric}_auc"] = sum(
                (right_cost - left_cost) * (left_metrics[metric] + right_metrics[metric]) / 2.0
                for (left_cost, left_metrics), (right_cost, right_metrics) in zip(points, points[1:])
            )
        output.append(item)
    return output


def _run_fold_experiment(
    config: dict[str, Any], bundle: Any, fold: int, checkpoint_output: Path | None = None,
    data_sha256: dict[str, str] | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    required = ("dataset", "warp", "graph", "reranker", "experiment", "reader")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing experiment configuration sections: {', '.join(missing)}")
    experiment = config["experiment"]
    for required_key in (
        "methods", "seed", "randomization_samples",
        "partition_ablations",
    ):
        if required_key not in experiment:
            raise ValueError(f"experiment.{required_key} is required")
    if "budgets" in experiment:
        print(json.dumps({
            "warning": "experiment.budgets is ignored; each method runs its full native pipeline",
        }, ensure_ascii=False), flush=True)
    methods = [str(value).lower() for value in experiment["methods"]]
    unknown = set(methods) - REGIONAL_METHODS - GLOBAL_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    design_seed = int(experiment["seed"])

    if not bundle.test:
        raise ValueError("A paper experiment requires a non-empty test query split")
    progress = _load_fold_progress(checkpoint_output, fold, data_sha256)
    rows: list[dict[str, Any]] = list(progress["quality_cost_curve"])
    baseline_trials: list[dict[str, Any]] = [
        row for row in progress["baseline_trials"] if str(row.get("method")).lower() in BASE_RETRIEVAL_METHODS
    ]
    design_reports: list[dict[str, Any]] = []
    reader_trials: list[dict[str, Any]] = list(progress["reader_evaluation_trials"])
    partition_ablation_rows: list[dict[str, Any]] = []
    full_graph_trials = [
        row for row in progress["baseline_trials"] if str(row.get("method")).lower() in GRAPH_BASELINE_METHODS
    ]

    def persist() -> dict[str, Any]:
        fold_result = {
            "design_trials": design_reports,
            "baseline_trials": baseline_trials + full_graph_trials,
            "quality_cost_curve": rows,
            "partition_ablations": partition_ablation_rows,
            "reader_evaluation_trials": reader_trials,
        }
        if checkpoint_output is not None:
            _write_fold_progress(checkpoint_output, fold, data_sha256 or {}, fold_result)
        if on_progress is not None:
            try:
                on_progress(fold_result)
            except Exception as exc:
                print(json.dumps({
                    "fold_checkpoint": "partial_write_failed",
                    "fold": fold,
                    "error": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False), flush=True)
        return fold_result

    for design_seed in [design_seed]:
        model, graph_retriever = _build_model(config, bundle, design_seed)
        factory = GlobalBaselineFactory(
            bundle.documents, model.base, model.graph_builder, model.graph_retriever,
            model.reranker, model.config.candidate_k,
            ket_core_fraction=float(experiment.get("ket_core_fraction", 0.8)),
            g2_core_fraction=float(experiment.get("g2_core_fraction", 0.8)),
        ) if set(methods) & GLOBAL_METHODS else None
        design_reports.append({
            "design_seed": design_seed, "fold": fold,
            "train_routing": model.routing_diagnostics(bundle.train),
            "test_routing": model.routing_diagnostics(bundle.test),
            "probe_labeling_online_cost": model.probe_labeling_online_cost,
            **model.report(),
        })
        if checkpoint_output is not None:
            design_path = _fold_design_path(checkpoint_output, fold)
            write_json(design_path, {
                "fold": fold,
                "design_trials": design_reports,
            })
            print(json.dumps({
                "fold_checkpoint": "design",
                "fold": fold,
                "path": str(design_path),
                "num_regions": design_reports[-1].get("num_regions"),
                "probe_regions": design_reports[-1].get("probe_regions"),
            }, ensure_ascii=False), flush=True)
        persist()

        done_baselines = {_baseline_row_key(row) for row in baseline_trials}
        for method in ("bm25", "dense", "hybrid"):
            key = (method, fold, design_seed)
            if key in done_baselines:
                print(json.dumps({
                    "fold_checkpoint": "reuse_baseline", "fold": fold, "method": method,
                }, ensure_ascii=False), flush=True)
                continue
            before = graph_retriever.stats()
            metrics = model.evaluate_base(bundle.test, method)
            baseline_row = {
                "design_seed": design_seed, "fold": fold, "method": method,
                "online_retrieval_cost": graph_retriever.delta(before), **metrics,
            }
            attach_token_efficiency(baseline_row)
            baseline_trials.append(baseline_row)
            persist()

        done_quality = {_quality_row_key(row) for row in rows}
        for method in _ordered_methods(methods):
            key = (method, fold, design_seed)
            if key in done_quality:
                print(json.dumps({
                    "fold_checkpoint": "reuse_quality",
                    "fold": fold, "method": method, "selection": "full_pipeline",
                }, ensure_ascii=False), flush=True)
                continue
            before = graph_retriever.stats()
            if method in REGIONAL_METHODS:
                selected = model.select(method)
                metrics = model.evaluate(bundle.test, selected)
                deployment_cost = aggregate_costs([model.graphs[key].cost for key in selected])
                selected_ids: dict[str, Any] = {
                    "selected_regions": selected,
                    "ranking_formula": method,
                    "warp_selected_count": len(model.select("warp")),
                }
                estimated_cost = sum(model.costs[key] for key in selected)
            else:
                if factory is None:
                    raise RuntimeError("Global baseline factory was not initialized")
                baseline = factory.build(method)
                metrics = evaluate_retrieval(
                    bundle.test, baseline.search, bootstrap_seed=design_seed,
                )
                deployment_cost = baseline.cost
                selected_ids = {
                    "selected_documents": baseline.graph.doc_ids if baseline.graph is not None else [],
                    "core_fraction": (
                        factory.ket_core_fraction if method == "ket_rag" else factory.g2_core_fraction
                    ),
                }
                estimated_cost = (
                    sum(factory.doc_costs[key] for key in selected_ids["selected_documents"])
                    + (baseline.lightweight.cost.selection_cost if baseline.lightweight is not None else 0.0)
                )
            probe_graph_cost = aggregate_costs([outcome.graph.cost for outcome in model.probes.values()])
            labeling_cost = usage_to_cost(model.probe_labeling_online_cost)
            interaction_cost = usage_to_cost(model.interaction_analysis.get("online_analysis_cost"))
            if method == "warp":
                # design.md：design_search = probe 构图 + design-set LLM；
                # first_run = deployment + 未进入部署集的方法专属 design cost。
                design_search_cost = aggregate_costs([probe_graph_cost, labeling_cost, interaction_cost])
                probe_only_cost = aggregate_costs([
                    outcome.graph.cost for region_id, outcome in model.probes.items()
                    if region_id not in selected
                ])
                first_run_cost = aggregate_costs([
                    deployment_cost, probe_only_cost, labeling_cost, interaction_cost,
                ])
                method_design_wall = (
                    model.design_timings["partition_seconds"]
                    + model.design_timings["feature_seconds"]
                    + model.design_timings["predictor_seconds"]
                )
            else:
                design_search_cost = aggregate_costs([])
                first_run_cost = deployment_cost
                method_design_wall = 0.0
            rows.append({
                "method": method,
                "design_seed": design_seed,
                "fold": fold,
                "selection": "full_pipeline",
                **selected_ids,
                "selected_estimated_graph_cost": estimated_cost,
                "deployment_cost": deployment_cost.to_dict(),
                "design_search_cost": design_search_cost.to_dict(),
                "first_run_cost_including_probe": first_run_cost.to_dict(),
                "method_specific_design_wall_seconds": method_design_wall,
                "first_run_wall_seconds": first_run_cost.wall_seconds + method_design_wall,
                "online_retrieval_cost": graph_retriever.delta(before),
                **metrics,
            })
            attach_token_efficiency(rows[-1])
            persist()

        done_graph_baselines = {str(row.get("method")).lower() for row in full_graph_trials}
        if GRAPH_BASELINE_METHODS <= done_graph_baselines:
            print(json.dumps({
                "fold_checkpoint": "reuse_full_graph", "fold": fold,
            }, ensure_ascii=False), flush=True)
            full_cost_payload = next(
                row["actual_construction_cost"] for row in full_graph_trials
                if str(row.get("method")).lower() == "full_graph"
            )
            denominator_tokens = float(
                full_cost_payload["input_tokens"] + full_cost_payload["output_tokens"]
                + full_cost_payload["embedding_tokens"]
            )
            denominator_usd = float(full_cost_payload["estimated_usd"])
        else:
            before = graph_retriever.stats()
            full_metrics = model.evaluate_full_graph(bundle.test)
            full_online = graph_retriever.delta(before)
            before = graph_retriever.stats()
            graph_metrics = model.evaluate_full_graph_only(bundle.test)
            graph_online = graph_retriever.delta(before)
            full_cost = model.full_graph.cost
            full_graph_trials = [
                {"design_seed": design_seed, "fold": fold, "method": "hipporag2", "online_retrieval_cost": graph_online,
                 "actual_construction_cost": full_cost.to_dict(), **graph_metrics},
                {"design_seed": design_seed, "fold": fold, "method": "full_graph", "online_retrieval_cost": full_online,
                 "actual_construction_cost": full_cost.to_dict(), **full_metrics},
            ]
            for graph_row in full_graph_trials:
                attach_token_efficiency(graph_row)
            denominator_tokens = full_cost.selection_cost
            denominator_usd = full_cost.estimated_usd
            persist()
        _attach_cost_fractions(rows, denominator_tokens, denominator_usd)
        for row in rows:
            attach_token_efficiency(row)
        for graph_row in full_graph_trials:
            attach_token_efficiency(graph_row)
        persist()

        reader_config = config["reader"]
        if bool(reader_config["enabled"]):
            reader_top_k = int(reader_config["top_k"])
            done_readers = {_baseline_row_key(row) for row in reader_trials}
            for method in [str(value).lower() for value in reader_config["methods"]]:
                if (method, fold, design_seed) in done_readers:
                    print(json.dumps({
                        "fold_checkpoint": "reuse_reader", "fold": fold, "method": method,
                    }, ensure_ascii=False), flush=True)
                    continue
                before = graph_retriever.stats()
                if method in {"bm25", "dense", "hybrid"}:
                    search = lambda query, k, method=method: model.search_base(query, k, method)
                elif method == "hipporag2":
                    graph = model.materialize_full_graph()
                    search = lambda query, k, graph=graph: fuse_and_rerank(
                        query, [model.graph_retriever.search(query, graph, model.config.candidate_k)],
                        model.reranker, k=k, candidate_k=model.config.candidate_k, source="hipporag2_reranked",
                    )
                elif method == "full_graph":
                    search = model.search_full_graph
                elif method in REGIONAL_METHODS:
                    selected = model.select(method)
                    model.materialize(selected)
                    selected_set = set(selected)
                    search = lambda query, k, selected_set=selected_set: model.search(query, k, selected_set)
                elif method in GLOBAL_METHODS:
                    if factory is None:
                        raise RuntimeError("Global baseline factory was not initialized")
                    baseline = factory.build(method)
                    search = baseline.search
                else:
                    raise ValueError(f"Unknown reader method: {method}")
                reader_metrics = evaluate_hipporag2_reader(
                    bundle.test, search, bundle.documents, model.materialize_full_graph(), reader_top_k,
                )
                source = next(
                    (
                        row for row in [*rows, *baseline_trials, *full_graph_trials]
                        if str(row.get("method")).lower() == method
                        and int(row.get("fold", fold)) == fold
                    ),
                    {},
                )
                reader_row = {
                    "design_seed": design_seed, "fold": fold, "method": method,
                    "online_retrieval_cost": graph_retriever.delta(before),
                    "deployment_cost": source.get("deployment_cost") or source.get("actual_construction_cost"),
                    "first_run_cost_including_probe": source.get("first_run_cost_including_probe"),
                    **reader_metrics,
                }
                attach_token_efficiency(reader_row, extra_online=consumed_tokens(source.get("online_retrieval_cost")))
                reader_trials.append(reader_row)
                persist()

    return persist()


def _run_partition_ablation_fold(
    config: dict[str, Any], bundle: Any, fold: int, mode: str,
) -> dict[str, Any]:
    """Run one partition mode in an isolated process/fold model lifecycle."""
    experiment = config["experiment"]
    ablation_config = experiment.get("partition_ablations")
    if not isinstance(ablation_config, dict):
        raise ValueError("experiment.partition_ablations is required")
    configured_modes = [str(value) for value in ablation_config.get("modes", [])]
    if mode not in configured_modes:
        raise ValueError(f"Partition mode {mode!r} is not configured in {configured_modes}")
    if not bundle.test:
        raise ValueError("A partition ablation requires a non-empty test query split")
    ablation_seed = int(ablation_config["seed"])
    model, graph_retriever = _build_model(config, bundle, ablation_seed, mode)
    selected = model.select("warp")
    before = graph_retriever.stats()
    metrics = model.evaluate(bundle.test, selected)
    ablation = {
        "partition_mode": mode,
        "design_seed": ablation_seed,
        "fold": fold,
        "selection": "full_pipeline",
        "selected_regions": selected,
        "deployment_cost": aggregate_costs([model.graphs[key].cost for key in selected]).to_dict(),
        "online_retrieval_cost": graph_retriever.delta(before),
        "routing": model.routing_diagnostics(bundle.test),
        "num_regions": len(model.regions),
        **metrics,
    }
    attach_token_efficiency(ablation)
    return ablation


def _fold_checkpoint_path(output: Path, fold: int) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.fold-{fold}{suffix}")


def _fold_progress_path(output: Path, fold: int) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.fold-{fold}.progress{suffix}")


def _quality_row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["method"]).lower(),
        int(row["fold"]),
        int(row["design_seed"]),
    )


def _baseline_row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["method"]).lower(), int(row["fold"]), int(row["design_seed"]))


def _empty_fold_progress() -> dict[str, list[dict[str, Any]]]:
    return {
        "design_trials": [],
        "baseline_trials": [],
        "quality_cost_curve": [],
        "partition_ablations": [],
        "reader_evaluation_trials": [],
    }


def _load_fold_progress(
    output: Path | None, fold: int, data_sha256: dict[str, str] | None,
) -> dict[str, list[dict[str, Any]]]:
    empty = _empty_fold_progress()
    if output is None:
        return empty
    path = _fold_progress_path(output, fold)
    if not path.exists():
        return empty
    payload = _read_result(path)
    stored_hash = payload.get("data_sha256")
    if data_sha256 is not None and stored_hash != data_sha256:
        raise ValueError(f"Fold progress {path} was produced with different dataset bytes")
    if int(payload.get("fold", -1)) != fold:
        raise ValueError(f"Fold progress {path} does not belong to fold {fold}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Fold progress {path} is missing result")
    loaded = _empty_fold_progress()
    for key in loaded:
        rows = result.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"Fold progress {path} field {key} must be a list")
        loaded[key] = rows
    loaded["quality_cost_curve"] = [
        row for row in loaded["quality_cost_curve"]
        if str(row.get("selection")) == "full_pipeline"
    ]
    print(json.dumps({
        "fold_checkpoint": "resume_progress",
        "fold": fold,
        "path": str(path),
        "baselines": len(loaded["baseline_trials"]),
        "quality_cost_rows": len(loaded["quality_cost_curve"]),
        "reader_rows": len(loaded["reader_evaluation_trials"]),
    }, ensure_ascii=False), flush=True)
    return loaded


def _write_fold_progress(
    output: Path, fold: int, data_sha256: dict[str, str], fold_result: dict[str, Any],
) -> None:
    write_json(_fold_progress_path(output, fold), {
        "fold": fold,
        "data_sha256": data_sha256,
        "result": fold_result,
    })


def _attach_cost_fractions(
    rows: list[dict[str, Any]], denominator_tokens: float, denominator_usd: float,
) -> None:
    if denominator_tokens <= 0:
        raise ValueError("Full-graph token cost denominator must be positive")
    for row in rows:
        deployed = row["deployment_cost"]
        first_run = row["first_run_cost_including_probe"]
        row["actual_cost_fraction"] = (
            deployed["input_tokens"] + deployed["output_tokens"] + deployed["embedding_tokens"]
        ) / denominator_tokens
        row["first_run_cost_fraction"] = (
            first_run["input_tokens"] + first_run["output_tokens"] + first_run["embedding_tokens"]
        ) / denominator_tokens
        row["actual_usd_fraction"] = (
            deployed["estimated_usd"] / denominator_usd if denominator_usd > 0 else 0.0
        )


def _fold_design_path(output: Path, fold: int) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.fold-{fold}.design{suffix}")


def _partial_output_path(output: Path) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.partial{suffix}")


def _write_fold_checkpoint(
    path: Path, fold: int, data_sha256: dict[str, str], fold_result: dict[str, Any],
) -> None:
    write_json(path, {
        "fold": fold,
        "data_sha256": data_sha256,
        "result": fold_result,
    })


def _load_fold_checkpoint(path: Path, data_sha256: dict[str, str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_result(path)
    if payload.get("data_sha256") != data_sha256:
        raise ValueError(f"Fold checkpoint {path} was produced with different dataset bytes")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Fold checkpoint {path} is missing result")
    return result


def _assemble_experiment_result(
    config: dict[str, Any],
    metadata: dict[str, Any],
    bundles: list[Any],
    fold_results: list[dict[str, Any]],
    phase: str,
    partition_mode: str | None,
) -> dict[str, Any]:
    experiment = config["experiment"]
    seed = int(experiment["seed"])
    baseline_trials = [row for result in fold_results for row in result["baseline_trials"]]
    quality_rows = [row for result in fold_results for row in result["quality_cost_curve"]]
    design_trials = [row for result in fold_results for row in result["design_trials"]]
    partition_rows = [row for result in fold_results for row in result["partition_ablations"]]
    reader_trials = [row for result in fold_results for row in result["reader_evaluation_trials"]]
    completed_folds = sorted({
        int(row["fold"])
        for result in fold_results
        for row in (result["quality_cost_curve"] or result["design_trials"] or result["partition_ablations"])
    })
    held_out_ids = [query.id for bundle in bundles for query in bundle.test]
    if len(held_out_ids) != len(set(held_out_ids)):
        raise ValueError("Cross-fitting must evaluate each query exactly once")
    for row in quality_rows + baseline_trials + partition_rows:
        attach_token_efficiency(row)
    retrieval_comparisons = quality_rows + [
        row for row in baseline_trials if "per_query" in row
    ]
    quality_summary = _crossfit_summary(quality_rows, ("method",), seed) if quality_rows else []
    baseline_summary = _crossfit_summary(baseline_trials, ("method",), seed) if baseline_trials else []
    token_efficiency_rows = []
    seen_methods: set[str] = set()
    for item in quality_summary + baseline_summary:
        method = str(item["method"])
        if method in seen_methods:
            continue
        seen_methods.add(method)
        token_efficiency_rows.append({
            "method": method,
            "complete_evidence@10": item.get("complete_evidence@10"),
            "total_tokens_excluding_design": item.get("total_tokens_excluding_design"),
            "total_tokens_including_design": item.get("total_tokens_including_design"),
            "token_efficiency_excluding_design": item.get("token_efficiency_excluding_design"),
            "token_efficiency_including_design": item.get("token_efficiency_including_design"),
            "formal_metric": False,
        })
    metadata = {
        **metadata,
        "cross_fitting": {
            "folds": int(experiment["cross_fitting_folds"]),
            "completed_folds": completed_folds,
            "incomplete": len(fold_results) < int(experiment["cross_fitting_folds"]),
            "design_queries_per_fold": [len(bundle.train) for bundle in bundles[:len(fold_results)]],
            "test_queries_per_fold": [len(bundle.test) for bundle in bundles[:len(fold_results)]],
            "assignment": "sha256(seed:query_id) ordering followed by round-robin folds",
        },
        "execution_phase": {
            "phase": phase,
            "partition_mode": partition_mode,
        },
        "token_efficiency_is_formal": False,
    }
    return {
        "run_metadata": metadata,
        "design_trials": design_trials,
        "baseline_trials": baseline_trials,
        "baselines": baseline_summary,
        "quality_cost_curve": quality_rows,
        "quality_cost_summary": quality_summary,
        "quality_cost_auc": _quality_cost_auc(quality_rows) if quality_rows else [],
        "token_efficiency": token_efficiency_rows,
        "paired_significance": _significance(
            retrieval_comparisons, int(experiment["randomization_samples"]), seed,
        ) if retrieval_comparisons else [],
        "partition_ablations": partition_rows,
        "partition_ablation_summary": _crossfit_summary(
            partition_rows, ("partition_mode",), seed,
        ) if partition_rows else [],
        "reader_evaluation_trials": reader_trials,
        "reader_evaluation": _summarize(
            reader_trials, ("method",), ("answer_em", "answer_f1"),
        ) if reader_trials else [],
    }


def run_experiment(
    config: dict[str, Any], phase: str = "main", partition_mode: str | None = None,
    output: Path | None = None,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Run deterministic cross-fitting and merge all held-out query results."""
    if phase not in {"main", "partition-ablation"}:
        raise ValueError("phase must be main or partition-ablation")
    if phase == "partition-ablation" and not partition_mode:
        raise ValueError("partition_mode is required for partition-ablation phase")
    if phase == "main" and partition_mode is not None:
        raise ValueError("partition_mode is only valid for partition-ablation phase")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or "cross_fitting_folds" not in experiment:
        raise ValueError("experiment.cross_fitting_folds is required")
    folds = int(experiment["cross_fitting_folds"])
    seed = int(experiment["seed"])
    metadata = reproducibility_metadata(config)
    bundles = load_crossfit_bundles(config["dataset"], folds, seed)
    if max_folds is None:
        run_folds = folds
    else:
        if int(max_folds) < 1:
            raise ValueError("max_folds must be >= 1")
        run_folds = min(int(max_folds), folds)
    metadata["fold_limit"] = {"configured_folds": folds, "run_folds": run_folds}
    fold_results: list[dict[str, Any]] = []
    for fold, bundle in enumerate(bundles[:run_folds]):
        checkpoint_path = _fold_checkpoint_path(output, fold) if output is not None else None
        loaded = _load_fold_checkpoint(checkpoint_path, metadata["data_sha256"]) if checkpoint_path else None
        if loaded is not None:
            print(json.dumps({
                "fold_checkpoint": "reuse", "fold": fold, "path": str(checkpoint_path),
            }, ensure_ascii=False), flush=True)
            fold_results.append(loaded)
        elif phase == "main":
            def on_progress(fold_result: dict[str, Any]) -> None:
                if output is None:
                    return
                write_json(
                    _partial_output_path(output),
                    _assemble_experiment_result(
                        config, metadata, bundles, fold_results + [fold_result], phase, partition_mode,
                    ),
                )
            fold_result = _run_fold_experiment(
                config, bundle, fold, checkpoint_output=output,
                data_sha256=metadata["data_sha256"], on_progress=on_progress,
            )
            fold_results.append(fold_result)
            if checkpoint_path is not None:
                _write_fold_checkpoint(checkpoint_path, fold, metadata["data_sha256"], fold_result)
                print(json.dumps({
                    "fold_checkpoint": "saved", "fold": fold, "path": str(checkpoint_path),
                    "quality_cost_rows": len(fold_result["quality_cost_curve"]),
                }, ensure_ascii=False), flush=True)
        else:
            fold_result = {
                "design_trials": [],
                "baseline_trials": [],
                "quality_cost_curve": [],
                "partition_ablations": [
                    _run_partition_ablation_fold(config, bundle, fold, str(partition_mode))
                ],
                "reader_evaluation_trials": [],
            }
            fold_results.append(fold_result)
            if checkpoint_path is not None:
                _write_fold_checkpoint(checkpoint_path, fold, metadata["data_sha256"], fold_result)
                print(json.dumps({
                    "fold_checkpoint": "saved", "fold": fold, "path": str(checkpoint_path),
                }, ensure_ascii=False), flush=True)
        if output is not None:
            write_json(
                _partial_output_path(output),
                _assemble_experiment_result(config, metadata, bundles, fold_results, phase, partition_mode),
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _assemble_experiment_result(config, metadata, bundles, fold_results, phase, partition_mode)


def _phase_output_path(output: Path, label: str) -> Path:
    """Place durable phase checkpoints beside the final dataset artifact."""
    suffix = output.suffix or ".json"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}.{label}{suffix}")


def _read_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Phase artifact must be a JSON object: {path}")
    return value


def _run_phase_process(
    config_path: str, output: Path, phase: str, partition_mode: str | None = None,
) -> None:
    command = [
        sys.executable, "-m", "warp.run",
        "--config", config_path,
        "--output", str(output),
        "--phase", phase,
    ]
    if partition_mode is not None:
        command.extend(["--partition-mode", partition_mode])
    subprocess.run(command, check=True)


def _orchestrate_all_phases(config_path: str, output: Path) -> dict[str, Any]:
    """Run main and each partition mode in separate GPU-owning processes."""
    config = load_config(config_path)
    ablation_config = config["experiment"].get("partition_ablations")
    if not isinstance(ablation_config, dict):
        raise ValueError("experiment.partition_ablations is required")
    modes = [str(value) for value in ablation_config.get("modes", [])]
    if not modes:
        raise ValueError("At least one partition ablation mode is required")

    main_path = _phase_output_path(output, "main")
    _run_phase_process(config_path, main_path, "main")
    main_result = _read_result(main_path)

    phase_paths: list[Path] = [main_path]
    partition_rows: list[dict[str, Any]] = []
    partition_summaries: list[dict[str, Any]] = []
    expected_hashes = main_result["run_metadata"]["data_sha256"]
    for mode in modes:
        phase_path = _phase_output_path(output, f"partition-{mode}")
        _run_phase_process(config_path, phase_path, "partition-ablation", mode)
        phase_result = _read_result(phase_path)
        if phase_result["run_metadata"]["data_sha256"] != expected_hashes:
            raise ValueError(f"Phase {mode} used different dataset bytes")
        if phase_result["quality_cost_curve"] or phase_result["baseline_trials"]:
            raise ValueError(f"Partition phase {mode} unexpectedly contains main experiment rows")
        partition_rows.extend(phase_result["partition_ablations"])
        partition_summaries.extend(phase_result["partition_ablation_summary"])
        phase_paths.append(phase_path)

    main_result["partition_ablations"] = sorted(
        partition_rows, key=lambda row: (str(row["partition_mode"]), int(row["fold"])),
    )
    main_result["partition_ablation_summary"] = sorted(
        partition_summaries,
        key=lambda row: str(row["partition_mode"]),
    )
    main_result["run_metadata"]["execution_phase"] = {
        "phase": "all",
        "process_isolation": "main and each partition mode ran in separate child processes",
        "phase_artifacts": [str(path) for path in phase_paths],
    }
    return main_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WARP-G selective GraphRAG experiments")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--output", default="outputs/results.json", help="Result JSON path")
    parser.add_argument(
        "--phase", choices=["all", "main", "partition-ablation"], default="all",
        help="Run all phases with subprocess isolation, or one internal phase",
    )
    parser.add_argument(
        "--partition-mode",
        help="Configured partition mode; required only for --phase partition-ablation",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Run only the first N of the configured cross-fitting folds; split assignment stays full-fold",
    )
    parser.add_argument(
        "--skip-reader", action="store_true",
        help="Disable HippoRAG reader EM/F1 for this run",
    )
    parser.add_argument(
        "--skip-interactions", action="store_true",
        help="Skip probe-region interaction measurement during fit",
    )
    return parser


def _apply_run_overrides(config: dict[str, Any], *, skip_reader: bool, skip_interactions: bool) -> dict[str, Any]:
    updated = dict(config)
    if skip_reader:
        reader = dict(updated.get("reader") or {})
        reader["enabled"] = False
        updated["reader"] = reader
    if skip_interactions:
        warp = dict(updated.get("warp") or {})
        warp["interaction_pairs"] = 0
        updated["warp"] = warp
    return updated


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "all":
        if args.partition_mode is not None:
            raise ValueError("--partition-mode is invalid with --phase all")
        if args.max_folds is not None or args.skip_reader or args.skip_interactions:
            raise ValueError("--max-folds/--skip-reader/--skip-interactions require --phase main or partition-ablation")
        result = _orchestrate_all_phases(args.config, Path(args.output))
    else:
        config = _apply_run_overrides(
            load_config(args.config),
            skip_reader=args.skip_reader,
            skip_interactions=args.skip_interactions,
        )
        result = run_experiment(
            config, phase=args.phase, partition_mode=args.partition_mode,
            output=Path(args.output), max_folds=args.max_folds,
        )
    write_json(args.output, result)
    print(json.dumps({"output": args.output, "rows": len(result["quality_cost_curve"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
