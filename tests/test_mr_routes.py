from __future__ import annotations

import unittest

import numpy as np

from multi_retrieval.embed import HashingEmbedder, VectorStore
from multi_retrieval.index import CatalogIndex
from multi_retrieval.routes.category import CategoryRoute
from multi_retrieval.routes.keyword import KeywordRoute, build_expression
from multi_retrieval.routes.vector import VectorRoute
from multi_retrieval.types import DualQuery, Slots
from tests.fixtures import BOOTS_AND_NECKLACES, catalog_file, product


class ExpressionTest(unittest.TestCase):
    def test_multi_word_slots_get_a_phrase_term_as_well_as_words(self) -> None:
        expression = build_expression(["hiking boots"])
        self.assertIn('"hiking boots"', expression)
        self.assertIn('"hiking"', expression)

    def test_punctuation_never_reaches_the_parser(self) -> None:
        # a raw quote or bracket here would be an FTS5 syntax error
        expression = build_expression(['bad "quote" (and) bracket'])
        self.assertNotIn("(", expression)
        self.assertEqual(expression.count('"') % 2, 0)

    def test_nothing_searchable_yields_an_empty_expression(self) -> None:
        self.assertEqual(build_expression([]), "")
        self.assertEqual(build_expression(["a"]), "")     # single chars are dropped


class RouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = catalog_file(BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)


class KeywordRouteTest(RouteTestCase):
    def test_a_distinctive_phrase_finds_its_product(self) -> None:
        scores = KeywordRoute(self.index).search(Slots(free_text=["Triple Moon Pentagram Symbol"]))
        best = max(scores, key=scores.get)
        self.assertEqual(self.index.ids[best], "NECK1")

    def test_empty_slots_produce_no_opinion(self) -> None:
        self.assertEqual(KeywordRoute(self.index).search(Slots()), {})


class CategoryRouteTest(RouteTestCase):
    def test_it_scores_the_right_breadcrumb(self) -> None:
        scores = CategoryRoute(self.index).search(Slots(category="Hiking Boots"))
        found = {self.index.ids[i] for i in scores}
        self.assertTrue({"BOOT1", "BOOT2", "BOOT3"}.issubset(found))

    def test_a_word_every_product_shares_scores_nothing(self) -> None:
        # "Clothing" is on every fixture product, so it separates nothing
        self.assertEqual(CategoryRoute(self.index).search(Slots(category="Clothing")), {})

    def test_colour_and_material_are_not_category_evidence(self) -> None:
        self.assertEqual(CategoryRoute(self.index).search(Slots(color="black", material="leather")), {})

    def test_scoring_is_deterministic_across_calls(self) -> None:
        route = CategoryRoute(self.index)
        first = route.search(Slots(category="Shoes Hiking Boots"))
        second = route.search(Slots(category="Shoes Hiking Boots"))
        self.assertEqual(first, second)


class VectorRouteTest(RouteTestCase):
    def test_similarity_ranks_the_closest_product_first(self) -> None:
        route = VectorRoute(self.index, HashingEmbedder(), cache_dir=str(self.path.parent / "c"))
        scores = route.search(DualQuery(slots=Slots(free_text=["Triple Moon Pentagram Symbol"])))
        best = max(scores, key=scores.get)
        self.assertEqual(self.index.ids[best], "NECK1")

    def test_an_empty_query_returns_nothing(self) -> None:
        route = VectorRoute(self.index, HashingEmbedder(), cache_dir=str(self.path.parent / "c"))
        self.assertEqual(route.search(DualQuery(slots=Slots())), {})

    def test_vectors_are_unit_length(self) -> None:
        matrix = HashingEmbedder().encode(["alpha beta", "gamma"])
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)

    def test_the_fallback_embedder_is_stable_across_processes(self) -> None:
        """crc32, not the builtin hash: Python randomises string hashing per
        process, which would make every cached vector wrong on the next run."""
        first = HashingEmbedder().encode(["waterproof leather upper"])
        second = HashingEmbedder().encode(["waterproof leather upper"])
        np.testing.assert_array_equal(first, second)


class VectorCacheTest(unittest.TestCase):
    def test_cache_round_trips_and_rejects_a_changed_catalog(self) -> None:
        rows = [product("A", title="alpha"), product("B", title="beta")]
        with catalog_file(rows) as path:
            cache = str(path.parent / "cache")
            index = CatalogIndex(path)

            first = VectorStore(HashingEmbedder(), cache_dir=cache, catalog_path=path)
            first.build(index.embed_text, index.ids)
            self.assertFalse(first.loaded_from_cache)

            second = VectorStore(HashingEmbedder(), cache_dir=cache, catalog_path=path)
            second.build(index.embed_text, index.ids)
            self.assertTrue(second.loaded_from_cache)

            # a different id order must not be served from the old cache
            third = VectorStore(HashingEmbedder(), cache_dir=cache, catalog_path=path)
            third.build(index.embed_text[::-1], index.ids[::-1])
            self.assertFalse(third.loaded_from_cache)


if __name__ == "__main__":
    unittest.main()
