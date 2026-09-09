import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from warp.eval.cutoffs import RETRIEVAL_KS, metric_names
from warp.eval.multistep import merge_ranked, run_multistep_retrieval
from warp.eval.reader import evaluate_hipporag2_reader
from warp.eval.retrieval import evaluate_retrieval, query_metrics, ranked_from_payload
from warp.models import Document, Query, SearchResult
from warp.partition.boundary import region_coaccess_neighbors, replicate_boundaries
from warp.partition.coaccess_graph import CoaccessGraph, CoaccessGraphBuilder
from warp.partition.leiden import RegionPartitioner
from warp.run import _apply_run_overrides, _hydrate_ranked_cache


class RetrievalCutoffTest(unittest.TestCase):
    def test_query_metrics_include_all_cutoffs(self) -> None:
        query = Query("q1", "who", ["d1", "d2"])
        results = [
            SearchResult("d1", 1.0, "base", 1),
            SearchResult("d3", 0.5, "base", 2),
            SearchResult("d2", 0.4, "base", 3),
        ]
        values = query_metrics(results, query, RETRIEVAL_KS)
        self.assertEqual(set(values), set(metric_names()))
        self.assertEqual(values["evidence_recall@2"], 0.5)
        self.assertEqual(values["complete_evidence@2"], 0.0)
        self.assertEqual(values["complete_evidence@3"], 1.0)
        self.assertEqual(values["complete_evidence@5"], 1.0)
        self.assertEqual(values["complete_evidence@10"], 1.0)


class RetrievalThenReaderWorkflowTest(unittest.TestCase):
    def test_evaluate_retrieval_caches_ranked_results_in_one_pass(self) -> None:
        calls: list[tuple[str, int]] = []

        def search(query: str, k: int) -> list[SearchResult]:
            calls.append((query, k))
            return [SearchResult("d1", 1.0, "base", 1), SearchResult("d2", 0.5, "base", 2)]

        metrics = evaluate_retrieval(
            [Query("q1", "who", ["d1"])], search, ks=(2, 3, 5, 10), bootstrap_samples=10,
        )
        self.assertEqual(calls, [("who", 10)])
        self.assertIn("ranked_results", metrics)
        self.assertEqual(metrics["ranked_results"]["q1"][0]["doc_id"], "d1")
        self.assertEqual(metrics["evidence_recall@2"], 1.0)
        restored = ranked_from_payload(metrics["ranked_results"]["q1"])
        self.assertEqual(restored[0].doc_id, "d1")
        self.assertEqual(len(calls), 1)

    def test_reader_signature_has_no_search_callable(self) -> None:
        params = inspect.signature(evaluate_hipporag2_reader).parameters
        self.assertIn("ranked_by_query", params)
        self.assertNotIn("search", params)

    def test_reader_raises_when_cache_is_missing(self) -> None:
        import sys
        query = Query("q1", "who", ["d1"], answer="Ada")
        documents = [Document("d1", "Ada Lovelace")]
        misc = MagicMock()
        graph = MagicMock()
        graph.backend = MagicMock()
        with patch.dict(sys.modules, {
            "hipporag": MagicMock(),
            "hipporag.utils": MagicMock(),
            "hipporag.utils.misc_utils": misc,
        }):
            with self.assertRaises(KeyError) as ctx:
                evaluate_hipporag2_reader([query], {}, documents, graph)
        self.assertIn("q1", str(ctx.exception))

    def test_hydrate_ranked_cache_prefers_trial_payload(self) -> None:
        cache = _hydrate_ranked_cache(
            [{"method": "warp", "ranked_results": {"q1": [{"doc_id": "d1", "score": 1.0}]}}],
            [{"method": "bm25"}],
        )
        self.assertEqual(cache["warp"]["q1"][0]["doc_id"], "d1")
        self.assertNotIn("bm25", cache)

    def test_skip_multistep_override_sets_zero_steps(self) -> None:
        updated = _apply_run_overrides(
            {"warp": {"retrieval_k": 10}, "reader": {"enabled": True}},
            skip_reader=False, skip_interactions=False, skip_multistep=True,
        )
        self.assertEqual(updated["warp"]["multistep_max_steps"], 0)

    def test_fold_reader_loop_does_not_rebuild_search(self) -> None:
        import warp.run as runmod
        source = inspect.getsource(runmod._run_fold_experiment)
        self.assertIn("evaluate_hipporag2_reader(", source)
        self.assertIn("retrieval_cache.get(method)", source)
        self.assertNotIn("search = lambda", source)
        self.assertNotIn("search = baseline.search", source)
        self.assertNotIn("search = model.search", source)


class RegionRouterTest(unittest.TestCase):
    def test_route_unions_base_hit_router_and_bridge(self) -> None:
        from warp.pipeline import WARPConfig, WARPG
        base = MagicMock()
        base.dense.encode.return_value = [[0.0, 1.0]]
        model = WARPG(
            WARPConfig(region_router_k=1, bridge_k=1),
            MagicMock(), MagicMock(), MagicMock(), base,
        )
        model.doc_to_regions = {"d1": ["r0"]}
        model.region_vectors = {
            "r0": [1.0, 0.0],
            "r1": [0.0, 1.0],
            "r2": [1.0, 1.0],
        }
        model.region_neighbors = {"r0": {"r2": 5.0}, "r1": {"r2": 1.0}}
        trace = model._route_regions(
            "q", [SearchResult("d1", 1.0, "base", 1)], {"r0", "r1", "r2"},
        )
        self.assertEqual(trace["base_hit_regions"], ["r0"])
        self.assertEqual(trace["router_regions"], ["r1"])
        self.assertEqual(trace["bridge_regions"], ["r2"])
        self.assertEqual(trace["routed_regions"], ["r0", "r1", "r2"])


class MultistepLogTest(unittest.TestCase):
    def test_merge_ranked_keeps_higher_score(self) -> None:
        merged = merge_ranked(
            [SearchResult("d1", 0.2, "base", 1, "r1")],
            [SearchResult("d1", 0.9, "graph", 1, "r2"), SearchResult("d2", 0.1, "base", 2)],
        )
        self.assertEqual([item.doc_id for item in merged], ["d1", "d2"])
        self.assertEqual(merged[0].source, "graph")
        self.assertEqual(merged[0].rank, 1)

    def test_stops_on_complete_evidence_and_writes_step_log(self) -> None:
        calls: list[str] = []

        def search(query: str, k: int):
            calls.append(query)
            hits = [SearchResult("d1", 1.0, "base", 1, "r1"), SearchResult("d2", 0.8, "graph", 2, "r2")]
            return hits, {
                "base_hit_regions": ["r1"],
                "router_regions": ["r2"],
                "bridge_regions": [],
                "routed_regions": ["r1", "r2"],
                "tokens": {"logical_input_tokens": 3, "logical_output_tokens": 1},
            }

        generations: list[str] = []

        def generate(prompt: str) -> str:
            generations.append(prompt)
            return "follow up"

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "warp.jsonl"
            summary = run_multistep_retrieval(
                [Query("q1", "who", ["d1", "d2"])],
                search, generate, max_steps=3, retrieval_k=5, ks=(2, 3, 5, 10),
                log_path=log_path, method="warp",
            )
            self.assertEqual(calls, ["who"])
            self.assertEqual(generations, [])
            self.assertEqual(summary["complete_evidence@10"], 1.0)
            rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            first = json.loads(rows[0])
            self.assertEqual(first["step"], 1)
            self.assertEqual(first["stop_reason"], "complete_evidence")
            self.assertEqual(first["original_query"], "who")
            self.assertEqual(first["routed_regions"], ["r1", "r2"])
            self.assertEqual(first["tokens"]["logical_input_tokens"], 3)
            self.assertIn("evidence_recall@2", first["metrics"])

    def test_rewritten_query_is_used_on_next_step(self) -> None:
        seen: list[str] = []

        def search(query: str, k: int):
            seen.append(query)
            if query == "who":
                hits = [SearchResult("d1", 1.0, "base", 1)]
            else:
                hits = [SearchResult("d1", 1.0, "base", 1), SearchResult("d2", 0.9, "graph", 2)]
            return hits, {"base_hit_regions": [], "router_regions": [], "bridge_regions": [], "routed_regions": []}

        def generate(prompt: str) -> str:
            return "missing hop"

        summary = run_multistep_retrieval(
            [Query("q1", "who", ["d1", "d2"])],
            search, generate, max_steps=3, retrieval_k=5,
        )
        self.assertEqual(seen, ["who", "missing hop"])
        self.assertEqual(summary["complete_evidence@10"], 1.0)


class BoundaryAndCoaccessTest(unittest.TestCase):
    def test_rank_idf_downweights_hub_pairs_versus_count(self) -> None:
        documents = [Document("a", "A"), Document("b", "B"), Document("c", "C")]
        queries = [Query("q1", "one", ["a"]), Query("q2", "two", ["a"]), Query("q3", "three", ["a"])]
        query_top_ids = {
            "q1": ["a", "b"],
            "q2": ["a", "c"],
            "q3": ["a", "b", "c"],
        }

        class _Retriever:
            dense = None

        count = CoaccessGraphBuilder(top_k=3, semantic_k=0, semantic_lambda=0.0, weight_mode="count").build(
            documents, queries, _Retriever(), query_top_ids=query_top_ids,
        )
        weighted = CoaccessGraphBuilder(top_k=3, semantic_k=0, semantic_lambda=0.0, weight_mode="rank_idf").build(
            documents, queries, _Retriever(), query_top_ids=query_top_ids,
        )
        self.assertGreater(count.query_edges[("a", "b")], weighted.query_edges[("a", "b")])
        self.assertGreater(count.query_edges[("a", "c")], weighted.query_edges[("a", "c")])

    def test_boundary_replication_keeps_cores_disjoint(self) -> None:
        from warp.models import Region
        regions = [
            Region("r0", ["a", "b"], metadata={"core_doc_ids": ["a", "b"]}),
            Region("r1", ["c"], metadata={"core_doc_ids": ["c"]}),
        ]
        graph = CoaccessGraph(["a", "b", "c"], {}, {("a", "c"): 4.0, ("b", "c"): 1.0}, {}, {})
        updated = replicate_boundaries(regions, graph, 1.0)
        cores = [set(region.metadata["core_doc_ids"]) for region in updated]
        self.assertEqual(len(cores[0] & cores[1]), 0)
        copied = set(updated[0].doc_ids) | set(updated[1].doc_ids)
        self.assertIn("a", copied)
        self.assertTrue(any("a" in region.metadata["boundary_doc_ids"] or "c" in region.metadata["boundary_doc_ids"]
                            for region in updated))
        neighbors = region_coaccess_neighbors(updated, graph)
        self.assertIn("r1", neighbors["r0"])

    def test_expensive_region_is_split(self) -> None:
        graph = CoaccessGraph(
            ["a", "b", "c", "d"],
            {("a", "b"): 1.0, ("c", "d"): 1.0},
            {("a", "b"): 1.0, ("c", "d"): 1.0},
            {},
            {},
        )
        partitioner = RegionPartitioner(min_region_size=1, seed=0)
        calls = {"n": 0}

        def fake_leiden(subgraph):
            calls["n"] += 1
            if calls["n"] == 1:
                return [0] * len(subgraph.nodes)
            mid = max(len(subgraph.nodes) // 2, 1)
            return [0] * mid + [1] * (len(subgraph.nodes) - mid)

        partitioner._leiden = fake_leiden  # type: ignore[method-assign]
        regions = partitioner.partition(graph, cost_fn=lambda nodes: float(len(nodes)), max_cost=2.0)
        self.assertGreaterEqual(len(regions), 2)
        self.assertTrue(all(len(region.doc_ids) <= 2 for region in regions))


if __name__ == "__main__":
    unittest.main()
