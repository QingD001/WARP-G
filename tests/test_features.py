import unittest

from warp.advisor.features import RegionFeatureExtractor
from warp.models import Document, Query, Region, SearchResult
from warp.partition.coaccess_graph import CoaccessGraph


class _FakeDense:
    def vector(self, doc_id: str) -> list[float]:
        return [1.0, 0.0] if doc_id.startswith("a") or doc_id == "obama" else [0.0, 1.0]


class _FakeRetriever:
    dense = _FakeDense()


class RegionLocalFeatureTest(unittest.TestCase):
    def test_recall_failure_and_multi_doc_use_region_gold_only(self) -> None:
        regions = [
            Region("rA", ["obama", "other_a"]),
            Region("rB", ["hawaii", "other_b"]),
        ]
        documents = [
            Document("obama", "Barack Obama"),
            Document("other_a", "related A"),
            Document("hawaii", "Honolulu"),
            Document("other_b", "related B"),
        ]
        query = Query("q1", "Where was Obama born?", ["obama", "hawaii"])
        # Base 找回了奥巴马，没找回夏威夷，但两区都被路由到。
        results = [
            SearchResult("obama", 1.0, "base", 1),
            SearchResult("other_a", 0.8, "base", 2),
            SearchResult("other_b", 0.7, "base", 3),
        ]
        coaccess = CoaccessGraph(
            [doc.id for doc in documents], {}, {}, {}, {"q1": ["obama", "other_a", "other_b"]},
        )
        extractor = RegionFeatureExtractor(routing_k=3, candidate_k=3, retrieval_k=3)
        features, region_queries, _ = extractor.extract(
            regions, documents, [query], _FakeRetriever(), coaccess,
            query_results={"q1": results},
        )
        self.assertEqual(region_queries["rA"], ["q1"])
        self.assertEqual(region_queries["rB"], ["q1"])
        self.assertEqual(features["rA"].base_recall, 1.0)
        self.assertEqual(features["rA"].failure_rate, 0.0)
        self.assertEqual(features["rA"].multi_doc_rate, 0.0)
        self.assertEqual(features["rB"].base_recall, 0.0)
        self.assertEqual(features["rB"].failure_rate, 1.0)
        self.assertEqual(features["rB"].multi_doc_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
