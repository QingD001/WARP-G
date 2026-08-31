"""从 YAML 运行完整论文实验并输出自描述 JSON artifact。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import torch

from warp.baselines import GlobalBaselineFactory
from warp.data import load_crossfit_bundles
from warp.eval.construction_cost import aggregate_costs
from warp.eval.reader import evaluate_hipporag2_reader
from warp.eval.retrieval import evaluate_retrieval
from warp.eval.statistics import paired_bootstrap_interval, paired_randomization_pvalue
from warp.graph import HippoRAG2Config, HippoRAG2GraphBuilder, HippoRAG2GraphRetriever
from warp.graph.hipporag2 import SUPPORTED_API_VERSION, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from warp.pipeline import WARPConfig, WARPG
from warp.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever, fuse_and_rerank
from warp.utils import write_json


METRICS = ("evidence_recall@5", "evidence_recall@10", "complete_evidence@5", "complete_evidence@10")
REGIONAL_METHODS = {"warp", "random_region", "frequency_only", "gain_only", "cost_only"}
GLOBAL_METHODS = {"ket_rag", "g2cons"}


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
        output.append(item)
    return output


def _significance(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    budgets = sorted({float(row["budget_fraction"]) for row in rows})
    references = sorted({str(row["method"]) for row in rows if row["method"] != "warp"})
    for budget in budgets:
        candidate_rows = [row for row in rows if row["method"] == "warp"
                          and float(row["budget_fraction"]) == budget]
        candidate_per_query = {
            query_id: values
            for row in candidate_rows for query_id, values in row["per_query"].items()
        }
        for reference_name in references:
            reference_rows = [row for row in rows if row["method"] == reference_name
                              and float(row["budget_fraction"]) == budget]
            reference_per_query = {
                query_id: values
                for row in reference_rows for query_id, values in row["per_query"].items()
            }
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
                    "budget_fraction": budget,
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
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        budget_points: list[tuple[float, dict[str, float]]] = []
        for budget in sorted({float(row["budget_fraction"]) for row in method_rows}):
            trials = [row for row in method_rows if float(row["budget_fraction"]) == budget]
            cost = statistics.fmean(float(row["actual_cost_fraction"]) for row in trials)
            metrics = {metric: statistics.fmean(float(row[metric]) for row in trials) for metric in METRICS}
            budget_points.append((cost, metrics))
        by_cost: dict[float, list[dict[str, float]]] = {}
        for cost, metrics in budget_points:
            by_cost.setdefault(cost, []).append(metrics)
        points = [
            (cost, {metric: statistics.fmean(item[metric] for item in values) for metric in METRICS})
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
            item[f"{metric}_auc"] = sum(
                (right_cost - left_cost) * (left_metrics[metric] + right_metrics[metric]) / 2.0
                for (left_cost, left_metrics), (right_cost, right_metrics) in zip(points, points[1:])
            )
        output.append(item)
    return output


def _run_fold_experiment(config: dict[str, Any], bundle: Any, fold: int) -> dict[str, Any]:
    required = ("dataset", "warp", "graph", "reranker", "experiment", "reader")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing experiment configuration sections: {', '.join(missing)}")
    experiment = config["experiment"]
    for required_key in (
        "budgets", "methods", "seed", "randomization_samples",
        "partition_ablations",
    ):
        if required_key not in experiment:
            raise ValueError(f"experiment.{required_key} is required")
    budgets = [float(value) for value in experiment["budgets"]]
    if any(value < 0.0 or value > 1.0 for value in budgets):
        raise ValueError("All budget fractions must be in [0, 1]")
    methods = [str(value).lower() for value in experiment["methods"]]
    unknown = set(methods) - REGIONAL_METHODS - GLOBAL_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    design_seed = int(experiment["seed"])

    if not bundle.test:
        raise ValueError("A paper experiment requires a non-empty test query split")
    rows: list[dict[str, Any]] = []
    baseline_trials: list[dict[str, Any]] = []
    design_reports: list[dict[str, Any]] = []
    reader_trials: list[dict[str, Any]] = []
    partition_ablation_rows: list[dict[str, Any]] = []

    for design_seed in [design_seed]:
        model, graph_retriever = _build_model(config, bundle, design_seed)
        factory = GlobalBaselineFactory(
            bundle.documents, model.base, model.graph_builder, model.graph_retriever,
            model.reranker, model.config.candidate_k,
        ) if set(methods) & GLOBAL_METHODS else None
        design_reports.append({
            "design_seed": design_seed, "fold": fold,
            "train_routing": model.routing_diagnostics(bundle.train),
            "test_routing": model.routing_diagnostics(bundle.test),
            "probe_labeling_online_cost": model.probe_labeling_online_cost,
            **model.report(),
        })

        for method in ("bm25", "dense", "hybrid"):
            before = graph_retriever.stats()
            metrics = model.evaluate_base(bundle.test, method)
            baseline_trials.append({
                "design_seed": design_seed, "fold": fold, "method": method,
                "online_retrieval_cost": graph_retriever.delta(before), **metrics,
            })

        seed_rows: list[dict[str, Any]] = []
        for budget in budgets:
            for method in methods:
                before = graph_retriever.stats()
                if method in REGIONAL_METHODS:
                    selected = model.select(budget, method)
                    metrics = model.evaluate(bundle.test, selected)
                    deployment_cost = aggregate_costs([model.graphs[key].cost for key in selected])
                    selected_ids: dict[str, Any] = {"selected_regions": selected}
                    estimated_cost = sum(model.costs[key] for key in selected)
                else:
                    if factory is None:
                        raise RuntimeError("Global baseline factory was not initialized")
                    baseline = factory.build(method, budget * model.full_graph_cost)
                    metrics = evaluate_retrieval(
                        bundle.test, baseline.search, bootstrap_seed=design_seed,
                    )
                    deployment_cost = baseline.cost
                    selected_ids = {
                        "selected_documents": baseline.graph.doc_ids if baseline.graph is not None else [],
                    }
                    estimated_cost = (
                        sum(factory.doc_costs[key] for key in selected_ids["selected_documents"])
                        + (baseline.lightweight.cost.selection_cost if baseline.lightweight is not None else 0.0)
                    )
                probe_cost = aggregate_costs([outcome.graph.cost for outcome in model.probes.values()])
                design_search_cost = probe_cost if method == "warp" else aggregate_costs([])
                # probe 图被 materialize 复用不重建：first-run 按物理并集记账，
                # 只加未进入部署集的 probe 图，避免 probe∩selected 双计。
                if method == "warp":
                    probe_only_cost = aggregate_costs([
                        outcome.graph.cost for region_id, outcome in model.probes.items()
                        if region_id not in selected
                    ])
                    first_run_cost = aggregate_costs([deployment_cost, probe_only_cost])
                else:
                    first_run_cost = deployment_cost
                if method == "warp":
                    method_design_wall = sum(model.design_timings.values()) - model.design_timings["base_index_seconds"]
                else:
                    method_design_wall = 0.0
                seed_rows.append({
                    "method": method,
                    "design_seed": design_seed,
                    "fold": fold,
                    "budget_fraction": budget,
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

        before = graph_retriever.stats()
        full_metrics = model.evaluate_full_graph(bundle.test)
        full_online = graph_retriever.delta(before)
        before = graph_retriever.stats()
        graph_metrics = model.evaluate_full_graph_only(bundle.test)
        graph_online = graph_retriever.delta(before)
        full_cost = model.full_graph.cost
        baseline_trials.extend([
            {"design_seed": design_seed, "fold": fold, "method": "hipporag2", "online_retrieval_cost": graph_online,
             "actual_construction_cost": full_cost.to_dict(), **graph_metrics},
            {"design_seed": design_seed, "fold": fold, "method": "full_graph", "online_retrieval_cost": full_online,
             "actual_construction_cost": full_cost.to_dict(), **full_metrics},
        ])
        denominator_tokens = full_cost.selection_cost
        denominator_usd = full_cost.estimated_usd
        for row in seed_rows:
            deployed = row["deployment_cost"]
            first_run = row["first_run_cost_including_probe"]
            row["actual_cost_fraction"] = (
                deployed["input_tokens"] + deployed["output_tokens"] + deployed["embedding_tokens"]
            ) / denominator_tokens
            row["first_run_cost_fraction"] = (
                first_run["input_tokens"] + first_run["output_tokens"] + first_run["embedding_tokens"]
            ) / denominator_tokens
            row["actual_usd_fraction"] = deployed["estimated_usd"] / denominator_usd
        rows.extend(seed_rows)

        reader_config = config["reader"]
        if bool(reader_config["enabled"]):
            reader_budget = float(reader_config["budget"])
            reader_top_k = int(reader_config["top_k"])
            for method in [str(value).lower() for value in reader_config["methods"]]:
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
                    selected = model.select(reader_budget, method)
                    model.materialize(selected)
                    selected_set = set(selected)
                    search = lambda query, k, selected_set=selected_set: model.search(query, k, selected_set)
                elif method in GLOBAL_METHODS:
                    if factory is None:
                        raise RuntimeError("Global baseline factory was not initialized")
                    baseline = factory.build(method, reader_budget * model.full_graph_cost)
                    search = baseline.search
                else:
                    raise ValueError(f"Unknown reader method: {method}")
                reader_trials.append({
                    "design_seed": design_seed, "fold": fold, "method": method,
                    "online_retrieval_cost": graph_retriever.delta(before),
                    **evaluate_hipporag2_reader(
                        bundle.test, search, bundle.documents, model.materialize_full_graph(), reader_top_k,
                    ),
                })

    ablation_config = experiment["partition_ablations"]
    if not isinstance(ablation_config, dict):
        raise ValueError("experiment.partition_ablations is required")
    ablation_seed = int(ablation_config["seed"])
    ablation_budget = float(ablation_config["budget"])
    for mode in [str(value) for value in ablation_config["modes"]]:
        model, graph_retriever = _build_model(config, bundle, ablation_seed, mode)
        selected = model.select(ablation_budget, "warp")
        before = graph_retriever.stats()
        metrics = model.evaluate(bundle.test, selected)
        partition_ablation_rows.append({
            "partition_mode": mode,
            "design_seed": ablation_seed,
            "fold": fold,
            "budget_fraction": ablation_budget,
            "selected_regions": selected,
            "deployment_cost": aggregate_costs([model.graphs[key].cost for key in selected]).to_dict(),
            "online_retrieval_cost": graph_retriever.delta(before),
            "routing": model.routing_diagnostics(bundle.test),
            "num_regions": len(model.regions),
            **metrics,
        })

    return {
        "design_trials": design_reports,
        "baseline_trials": baseline_trials,
        "quality_cost_curve": rows,
        "partition_ablations": partition_ablation_rows,
        "reader_evaluation_trials": reader_trials,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic cross-fitting and merge all held-out query results."""
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or "cross_fitting_folds" not in experiment:
        raise ValueError("experiment.cross_fitting_folds is required")
    folds = int(experiment["cross_fitting_folds"])
    seed = int(experiment["seed"])
    metadata = reproducibility_metadata(config)
    bundles = load_crossfit_bundles(config["dataset"], folds, seed)
    fold_results: list[dict[str, Any]] = []
    for fold, bundle in enumerate(bundles):
        fold_results.append(_run_fold_experiment(config, bundle, fold))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_trials = [row for result in fold_results for row in result["baseline_trials"]]
    quality_rows = [row for result in fold_results for row in result["quality_cost_curve"]]
    design_trials = [row for result in fold_results for row in result["design_trials"]]
    partition_rows = [row for result in fold_results for row in result["partition_ablations"]]
    reader_trials = [row for result in fold_results for row in result["reader_evaluation_trials"]]
    held_out_ids = [query.id for bundle in bundles for query in bundle.test]
    if len(held_out_ids) != len(set(held_out_ids)):
        raise ValueError("Cross-fitting must evaluate each query exactly once")
    metadata["cross_fitting"] = {
        "folds": folds,
        "design_queries_per_fold": [len(bundle.train) for bundle in bundles],
        "test_queries_per_fold": [len(bundle.test) for bundle in bundles],
        "total_unique_held_out_queries": len(held_out_ids),
        "assignment": "sha256(seed:query_id) ordering followed by round-robin folds",
    }
    return {
        "run_metadata": metadata,
        "design_trials": design_trials,
        "baseline_trials": baseline_trials,
        "baselines": _crossfit_summary(baseline_trials, ("method",), seed),
        "quality_cost_curve": quality_rows,
        "quality_cost_summary": _crossfit_summary(
            quality_rows, ("method", "budget_fraction"), seed,
        ),
        "quality_cost_auc": _quality_cost_auc(quality_rows),
        "paired_significance": _significance(
            quality_rows, int(experiment["randomization_samples"]), seed,
        ),
        "partition_ablations": partition_rows,
        "partition_ablation_summary": _crossfit_summary(
            partition_rows, ("partition_mode", "budget_fraction"), seed,
        ),
        "reader_evaluation_trials": reader_trials,
        "reader_evaluation": _summarize(
            reader_trials, ("method",), ("answer_em", "answer_f1"),
        ) if reader_trials else [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WARP-G selective GraphRAG experiments")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--output", default="outputs/results.json", help="Result JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(load_config(args.config))
    write_json(args.output, result)
    print(json.dumps({"output": args.output, "rows": len(result["quality_cost_curve"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
