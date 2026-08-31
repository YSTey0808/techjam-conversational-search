from __future__ import annotations

import unittest

from multi_retrieval.index import CatalogIndex, flatten, parse_number, tokens
from tests.fixtures import (
    BOOTS_AND_NECKLACES,
    NORMALISED_BOOTS_AND_NECKLACES,
    SIDECAR_BOOTS_AND_NECKLACES,
    catalog_file,
    normalised_catalog_file,
    normalised_product,
    product,
)


class HelpersTest(unittest.TestCase):
    def test_flatten_handles_every_catalog_shape(self) -> None:
        self.assertEqual(flatten(None), "")
        self.assertEqual(flatten("plain"), "plain")
        self.assertEqual(flatten(["a", "b"]), "a b")
        self.assertEqual(flatten({"Material": "alloy"}), "Material alloy")
        self.assertEqual(flatten(12), "12")

    def test_tokens_drop_stopwords_and_single_characters(self) -> None:
        self.assertEqual(tokens("I am looking for a Waterproof Boot"), ["am", "waterproof", "boot"])

    def test_parse_number_fails_softly_on_the_dash_placeholder(self) -> None:
        self.assertIsNone(parse_number("—"))
        self.assertIsNone(parse_number(None))
        self.assertEqual(parse_number("19.99"), 19.99)


class IndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = catalog_file(BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_ids_follow_file_order(self) -> None:
        self.assertEqual(self.index.ids, ["BOOT1", "BOOT2", "BOOT3", "NECK1", "NECK2"])
        self.assertEqual(self.index.size, 5)

    def test_fts_rowid_equals_the_product_index(self) -> None:
        """The whole package addresses products by integer index; if the FTS
        rowid did not agree, every BM25 result would point at the wrong row."""
        hits = self.index.search_bm25('"pentagram" OR "triple"', 5)
        self.assertTrue(hits)
        for product_index, _ in hits:
            self.assertEqual(self.index.ids[product_index], "NECK1")

    def test_bm25_scores_come_back_best_first(self) -> None:
        hits = self.index.search_bm25('"boot" OR "waterproof"', 5)
        scores = [score for _, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_malformed_expression_returns_nothing_rather_than_raising(self) -> None:
        # customer text reaches this; a syntax error must not end the turn
        self.assertEqual(self.index.search_bm25('"unclosed AND (', 5), [])
        self.assertEqual(self.index.search_bm25("", 5), [])

    def test_category_postings_are_sorted_and_deduplicated(self) -> None:
        postings = self.index.category_postings["boots"]
        self.assertEqual(list(postings), sorted(set(postings)))

    def test_idf_zeroes_a_word_every_product_shares(self) -> None:
        # every fixture product sits under "Clothing", exactly like the real catalog
        self.assertAlmostEqual(self.index.idf(self.index.size), 0.0)
        self.assertGreater(self.index.idf(1), 0.0)

    def test_missing_price_is_stored_as_none_not_zero(self) -> None:
        self.assertIsNone(self.index.price[2])          # BOOT3 has price "—"
        self.assertEqual(self.index.price[3], 19.99)

    def test_embed_text_is_bounded(self) -> None:
        rows = [product("LONG", title="x" * 5000)]
        with catalog_file(rows) as path:
            self.assertLessEqual(len(CatalogIndex(path).embed_text[0]), 600)


class NormalisedIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = normalised_catalog_file(
            NORMALISED_BOOTS_AND_NECKLACES, raw_rows=SIDECAR_BOOTS_AND_NECKLACES
        )
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(
            self.path, raw_catalog_path=self.path.with_name("catalog.jsonl")
        )

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_the_faceted_shape_is_detected(self) -> None:
        self.assertEqual(self.index.format, "normalised")
        self.assertEqual(self.index.ids, ["BOOT1", "BOOT2", "BOOT3", "NECK1", "NECK2"])

    def test_audience_and_brand_get_their_own_postings(self) -> None:
        self.assertEqual(set(self.index.audience_postings), {"men", "unisex", "women"})
        self.assertEqual(list(self.index.brand_postings["moonco"]), [3])
        self.assertEqual({self.index.ids[i] for i in self.index.audience_postings["women"]},
                         {"NECK1", "NECK2"})

    def test_the_filter_gate_uses_the_clean_enum(self) -> None:
        # category_enum_postings: tokens of the 8-value `category` enum only
        self.assertEqual({self.index.ids[i] for i in self.index.category_enum_postings["jewelry"]},
                         {"NECK1", "NECK2"})
        self.assertNotIn("necklaces", self.index.category_enum_postings)

    def test_the_category_route_gets_path_tokens_from_the_sidecar(self) -> None:
        # category_postings: fine-grained tokens of the raw categories path
        self.assertIn("necklaces", self.index.category_postings)
        self.assertEqual({self.index.ids[i] for i in self.index.category_postings["necklaces"]},
                         {"NECK1", "NECK2"})
        self.assertIn("hiking", self.index.category_postings)

    def test_ratings_and_reviews_load_from_the_sidecar(self) -> None:
        self.assertTrue(self.index.has_ratings)
        self.assertEqual(self.index.reviews[0], 500.0)     # BOOT1
        self.assertEqual(self.index.reviews[4], 1000.0)    # NECK2
        self.assertAlmostEqual(self.index.rating[3], 4.8)  # NECK1

    def test_price_comes_from_the_normalised_row(self) -> None:
        self.assertEqual(self.index.price[0], 89.0)
        self.assertIsNone(self.index.price[2])             # BOOT3, price null

    def test_the_brand_is_searchable_through_bm25(self) -> None:
        hits = self.index.search_bm25('"moonco"', 5)
        self.assertTrue(hits)
        self.assertEqual(self.index.ids[hits[0][0]], "NECK1")

    def test_lexical_text_comes_from_the_sidecar(self) -> None:
        """A phrase from the product title/features -- absent from the faceted
        table -- is searchable because the sidecar text feeds the FTS columns."""
        hits = self.index.search_bm25('"pentagram" OR "triple"', 5)
        self.assertTrue(hits)
        self.assertEqual(self.index.ids[hits[0][0]], "NECK1")
        self.assertTrue(self.index.embed_text[3])          # NECK1 embed text non-empty

    def test_a_misaligned_sidecar_is_rejected(self) -> None:
        shuffled = list(reversed(SIDECAR_BOOTS_AND_NECKLACES))
        with normalised_catalog_file(NORMALISED_BOOTS_AND_NECKLACES, raw_rows=shuffled) as path:
            with self.assertRaises(RuntimeError):
                CatalogIndex(path, raw_catalog_path=path.with_name("catalog.jsonl"))

    def test_without_a_sidecar_ratings_are_absent_not_zero(self) -> None:
        with normalised_catalog_file(NORMALISED_BOOTS_AND_NECKLACES) as path:
            index = CatalogIndex(path)
            self.assertFalse(index.has_ratings)
            self.assertTrue(all(r is None for r in index.rating))
            self.assertEqual(index.price[0], 89.0)         # price still parsed

    def test_empty_markers_do_not_become_postings(self) -> None:
        rows = [normalised_product("X", brand="unknown", audience="unknown")]
        with normalised_catalog_file(rows) as path:
            index = CatalogIndex(path)
            self.assertEqual(index.brand_postings, {})
            self.assertEqual(index.audience_postings, {})

    def test_embed_text_is_bounded_for_facets_too(self) -> None:
        rows = [normalised_product("LONG", material=["x" * 5000])]
        with normalised_catalog_file(rows) as path:
            self.assertLessEqual(len(CatalogIndex(path).embed_text[0]), 600)


if __name__ == "__main__":
    unittest.main()
