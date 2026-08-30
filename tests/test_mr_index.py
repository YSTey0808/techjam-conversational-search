from __future__ import annotations

import unittest

from multi_retrieval.index import CatalogIndex, flatten, parse_number, tokens
from tests.fixtures import BOOTS_AND_NECKLACES, catalog_file, product


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


if __name__ == "__main__":
    unittest.main()
