import json
import tempfile
import unittest
from pathlib import Path

from warp.graph.openie_guard import parse_jsonish_key, sanitize_entities, sanitize_triples
from warp.run import (
    _attach_cost_fractions,
    _fold_progress_path,
    _load_fold_progress,
    _quality_cost_auc,
    _quality_row_key,
    _significance,
    _write_fold_progress,
)
from warp.utils import write_json


class SanitizeOpenIETest(unittest.TestCase):
    def test_drops_ellipsis_and_keeps_strings(self) -> None:
        self.assertEqual(sanitize_entities(["Obama", ..., None, "Harvard"]), ["Obama", "Harvard"])

    def test_entities_become_json_serializable(self) -> None:
        payload = {"named_entities": sanitize_entities(["Obama", ...])}
        self.assertEqual(json.dumps(payload), '{"named_entities": ["Obama"]}')

    def test_triples_skip_ellipsis_rows(self) -> None:
        values = [["a", "b", "c"], ..., ["", "x", "y"], ["a", "b", "c"], ["d", ..., "f"]]
        self.assertEqual(sanitize_triples(values), [["a", "b", "c"]])

    def test_parse_jsonish_named_entities(self) -> None:
        text = 'noise {"named_entities": ["Ada", "Lovelace"]} tail'
        self.assertEqual(parse_jsonish_key(text, "named_entities"), ["Ada", "Lovelace"])


class FoldProgressTest(unittest.TestCase):
    def test_round_trip_skips_completed_quality_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "hotpotqa.main.json"
            data_sha256 = {"corpus": "abc", "queries": "def"}
            row = {
                "method": "warp",
                "design_seed": 42,
                "fold": 0,
                "selection": "full_pipeline",
                "deployment_cost": {
                    "input_tokens": 1, "output_tokens": 2, "embedding_tokens": 3, "estimated_usd": 0.1,
                },
                "first_run_cost_including_probe": {
                    "input_tokens": 4, "output_tokens": 5, "embedding_tokens": 6, "estimated_usd": 0.2,
                },
            }
            _write_fold_progress(output, 0, data_sha256, {
                "design_trials": [],
                "baseline_trials": [{"method": "bm25", "fold": 0, "design_seed": 42}],
                "quality_cost_curve": [row],
                "partition_ablations": [],
                "reader_evaluation_trials": [],
            })
            loaded = _load_fold_progress(output, 0, data_sha256)
            self.assertEqual(len(loaded["quality_cost_curve"]), 1)
            self.assertEqual(_quality_row_key(loaded["quality_cost_curve"][0]), ("warp", 0, 42))
            self.assertTrue(_fold_progress_path(output, 0).exists())

    def test_old_budget_curve_rows_are_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "hotpotqa.main.json"
            data_sha256 = {"corpus": "abc", "queries": "def"}
            _write_fold_progress(output, 0, data_sha256, {
                "design_trials": [],
                "baseline_trials": [],
                "quality_cost_curve": [{
                    "method": "warp", "design_seed": 42, "fold": 0, "budget_fraction": 0.2,
                }],
                "partition_ablations": [],
                "reader_evaluation_trials": [],
            })
            loaded = _load_fold_progress(output, 0, data_sha256)
            self.assertEqual(loaded["quality_cost_curve"], [])

    def test_write_json_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_cost_fractions_tolerate_zero_usd(self) -> None:
        rows = [{
            "deployment_cost": {"input_tokens": 1, "output_tokens": 1, "embedding_tokens": 2, "estimated_usd": 0.0},
            "first_run_cost_including_probe": {
                "input_tokens": 2, "output_tokens": 2, "embedding_tokens": 4, "estimated_usd": 0.0,
            },
        }]
        _attach_cost_fractions(rows, 10.0, 0.0)
        self.assertEqual(rows[0]["actual_cost_fraction"], 0.4)
        self.assertEqual(rows[0]["actual_usd_fraction"], 0.0)

    def test_incomplete_quality_snapshot_does_not_crash_summaries(self) -> None:
        row = {
            "method": "ket_rag",
            "selection": "full_pipeline",
            "per_query": {"q1": {
                "evidence_recall@5": 0.0, "evidence_recall@10": 0.0,
                "complete_evidence@5": 0.0, "complete_evidence@10": 0.0,
            }},
        }
        self.assertEqual(_quality_cost_auc([row]), [])
        self.assertEqual(_significance([row], samples=10, seed=42), [])


if __name__ == "__main__":
    unittest.main()
