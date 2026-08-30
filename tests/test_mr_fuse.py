from __future__ import annotations

import unittest

from multi_retrieval import fuse

IDS = ["A", "B", "C", "D"]


class FuseTest(unittest.TestCase):
    def test_rrf_uses_rank_not_magnitude(self) -> None:
        """A route's own scale must not leak into the pool. A route whose scores
        are in the thousands should not outvote one scoring in decimals."""
        big = {"keyword": {0: 5000.0, 1: 4999.0}}
        small = {"keyword": {0: 0.5, 1: 0.4}}
        weights = {"keyword": 1.0}
        self.assertEqual(fuse.fuse(big, weights), fuse.fuse(small, weights))

    def test_additive_keeps_magnitude(self) -> None:
        routes = {"keyword": {0: 10.0, 1: 1.0}}
        combined = fuse.fuse(routes, {"keyword": 1.0}, mode="additive")
        self.assertGreater(combined[0] - combined[1], 0.5)

    def test_weights_decide_which_route_wins(self) -> None:
        routes = {"keyword": {0: 9.0, 1: 1.0}, "vector": {0: 1.0, 1: 9.0}}
        keyword_heavy = fuse.fuse(routes, {"keyword": 1.0, "vector": 0.1})
        vector_heavy = fuse.fuse(routes, {"keyword": 0.1, "vector": 1.0})
        self.assertGreater(keyword_heavy[0], keyword_heavy[1])
        self.assertGreater(vector_heavy[1], vector_heavy[0])

    def test_a_zero_weight_route_is_ignored(self) -> None:
        routes = {"keyword": {0: 1.0}, "vector": {1: 99.0}}
        combined = fuse.fuse(routes, {"keyword": 1.0, "vector": 0.0})
        self.assertNotIn(1, combined)

    def test_an_empty_route_contributes_nothing(self) -> None:
        self.assertEqual(fuse.fuse({"keyword": {}}, {"keyword": 1.0}), {})

    def test_evidence_from_two_routes_beats_one(self) -> None:
        both = fuse.fuse({"keyword": {0: 1.0}, "vector": {0: 1.0}}, {"keyword": 1.0, "vector": 1.0})
        one = fuse.fuse({"keyword": {0: 1.0}, "vector": {}}, {"keyword": 1.0, "vector": 1.0})
        self.assertGreater(both[0], one[0])

    def test_additive_handles_a_route_where_every_score_is_equal(self) -> None:
        combined = fuse.fuse({"keyword": {0: 3.0, 1: 3.0}}, {"keyword": 1.0}, mode="additive")
        self.assertEqual(combined[0], combined[1])


class OrderTest(unittest.TestCase):
    def test_ties_break_on_parent_asin_so_output_is_reproducible(self) -> None:
        combined = {0: 1.0, 1: 1.0, 2: 1.0}
        self.assertEqual([i for i, _ in fuse.order(combined, IDS, 3)], [0, 1, 2])

    def test_limit_is_respected(self) -> None:
        self.assertEqual(len(fuse.order({0: 3.0, 1: 2.0, 2: 1.0}, IDS, 2)), 2)


if __name__ == "__main__":
    unittest.main()
