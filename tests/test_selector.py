import unittest

from warp.advisor.selector import RegionSelector
from warp.baselines.global_graph import _select_core
from warp.eval.construction_cost import aggregate_costs, attach_token_efficiency, consumed_tokens, usage_to_cost
from warp.models import RegionFeatures
from warp.run import _ordered_methods


def _features(region_id: str, query_freq: float) -> RegionFeatures:
    return RegionFeatures(
        region_id=region_id, num_docs=10, num_tokens=100, query_freq=query_freq,
        base_recall=0.5, failure_rate=0.5, avg_retrieval_entropy=1.0,
        multi_doc_rate=0.2, embedding_dispersion=0.1, coaccess_density=0.1,
    )


class RegionSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {
            "r1": _features("r1", 4.0),
            "r2": _features("r2", 1.0),
            "r3": _features("r3", 0.0),
        }
        self.costs = {"r1": 10.0, "r2": 5.0, "r3": 1.0}
        self.gains = {"r1": 0.2, "r2": -0.1, "r3": 0.4}
        self.selector = RegionSelector(seed=42)

    def test_warp_keeps_positive_score_regions_only(self) -> None:
        selected = self.selector.select("warp", self.features, self.costs, self.gains)
        self.assertEqual(selected, ["r1"])

    def test_controls_only_replace_ranking_and_reuse_warp_count(self) -> None:
        warp_count = len(self.selector.select("warp", self.features, self.costs, self.gains))
        self.assertEqual(warp_count, 1)
        self.assertEqual(
            self.selector.select("gain_only", self.features, self.costs, self.gains, limit=warp_count),
            ["r3"],
        )
        self.assertEqual(
            self.selector.select("frequency_only", self.features, self.costs, self.gains, limit=warp_count),
            ["r1"],
        )
        self.assertEqual(
            self.selector.select("cost_only", self.features, self.costs, self.gains, limit=warp_count),
            ["r3"],
        )
        self.assertEqual(
            len(self.selector.select("random_region", self.features, self.costs, self.gains, limit=warp_count)),
            1,
        )

    def test_control_without_warp_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.selector.select("gain_only", self.features, self.costs, self.gains)

    def test_old_budget_argument_is_gone(self) -> None:
        with self.assertRaises(TypeError):
            self.selector.select("warp", 0.2, self.features, self.costs, self.gains)


class ExperimentOrderTest(unittest.TestCase):
    def test_methods_follow_design_order(self) -> None:
        self.assertEqual(
            _ordered_methods(["ket_rag", "g2cons", "cost_only", "warp", "random_region"]),
            ["warp", "random_region", "cost_only", "ket_rag", "g2cons"],
        )


class NativeCoreSelectionTest(unittest.TestCase):
    def test_ket_style_core_uses_count_fraction_not_token_budget(self) -> None:
        ranked = [f"d{i}" for i in range(10)]
        self.assertEqual(_select_core(ranked, 0.8), ranked[:8])
        self.assertEqual(_select_core(ranked, 1.0), ranked)

    def test_rejects_non_positive_fraction(self) -> None:
        with self.assertRaises(ValueError):
            _select_core(["d1"], 0.0)


class DesignCostMergeTest(unittest.TestCase):
    def test_probe_labeling_tokens_enter_design_cost(self) -> None:
        labeling = usage_to_cost({
            "logical_input_tokens": 11, "logical_output_tokens": 7, "wall_seconds": 1.5,
        })
        probe = usage_to_cost({})
        merged = aggregate_costs([probe, labeling])
        self.assertEqual(merged.input_tokens, 11)
        self.assertEqual(merged.output_tokens, 7)
        self.assertEqual(merged.wall_seconds, 1.5)


class TokenEfficiencyTest(unittest.TestCase):
    def test_counts_deployment_and_online_tokens(self) -> None:
        self.assertEqual(consumed_tokens({
            "input_tokens": 10, "output_tokens": 2, "embedding_tokens": 3,
        }), 15)
        self.assertEqual(consumed_tokens({
            "logical_input_tokens": 8, "logical_output_tokens": 1,
        }), 9)

    def test_writes_both_efficiency_versions(self) -> None:
        row = {
            "complete_evidence@10": 0.5,
            "deployment_cost": {"input_tokens": 20, "output_tokens": 0, "embedding_tokens": 0},
            "first_run_cost_including_probe": {
                "input_tokens": 30, "output_tokens": 0, "embedding_tokens": 0,
            },
            "online_retrieval_cost": {"logical_input_tokens": 5, "logical_output_tokens": 5},
        }
        attach_token_efficiency(row)
        self.assertEqual(row["total_tokens_excluding_design"], 30)
        self.assertEqual(row["total_tokens_including_design"], 40)
        self.assertAlmostEqual(row["token_efficiency_excluding_design"], 0.5 / 30)
        self.assertAlmostEqual(row["token_efficiency_including_design"], 0.5 / 40)

    def test_undefined_when_no_tokens(self) -> None:
        row = {"complete_evidence@10": 0.4, "online_retrieval_cost": {}}
        attach_token_efficiency(row)
        self.assertIsNone(row["token_efficiency_excluding_design"])
        self.assertEqual(row["total_tokens_excluding_design"], 0)


if __name__ == "__main__":
    unittest.main()
