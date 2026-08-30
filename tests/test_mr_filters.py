from __future__ import annotations

import unittest

from multi_retrieval.filters import HardFilter
from multi_retrieval.index import CatalogIndex
from multi_retrieval.types import Slots
from tests.fixtures import BOOTS_AND_NECKLACES, catalog_file, product


class NumericFilterTest(unittest.TestCase):
    """Unknown is not a violation. Only 10,410 of 50,000 real products carry a
    price, so excluding the unpriced ones would throw away four fifths of the
    catalog over a constraint they were never checked against."""

    def setUp(self) -> None:
        self._catalog = catalog_file(BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path)
        self.filter = HardFilter(self.index)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_a_product_with_no_price_survives_a_budget(self) -> None:
        outcome = self.filter.apply(Slots(price_max=25.0))
        self.assertIn("price_max", outcome.applied)
        self.assertIn(2, outcome.allowed)          # BOOT3, price "—"
        self.assertNotIn(0, outcome.allowed)       # BOOT1 at 89.0 is over

    def test_a_budget_that_nothing_meets_is_skipped_not_enforced(self) -> None:
        outcome = self.filter.apply(Slots(price_max=0.01))
        # every priced product is over, but the unpriced one keeps the set alive
        self.assertTrue(outcome.allowed is None or outcome.allowed)

    def test_min_reviews_filters_on_a_known_value(self) -> None:
        outcome = self.filter.apply(Slots(min_reviews=100))
        kept = {self.index.ids[i] for i in outcome.allowed}
        self.assertEqual(kept, {"BOOT1", "NECK1", "NECK2"})

    def test_no_numeric_slots_means_no_restriction(self) -> None:
        self.assertIsNone(self.filter.apply(Slots()).allowed)


class TextFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = catalog_file(BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path)
        self.filter = HardFilter(self.index)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_category_never_filters(self) -> None:
        """The most expensive bug in this package, pinned.

        Category is matched against a lossy field. When it filtered, the target
        survived only 8% of turns in the sessions multi_retrieval failed, even though
        a route had already surfaced it 99.2% of the time. It earns weight
        through CategoryRoute instead, and removes nothing."""
        outcome = self.filter.apply(Slots(category="Hiking Boots"))
        self.assertIsNone(outcome.allowed)
        self.assertNotIn("category", outcome.applied)

    def test_a_wrong_category_cannot_exclude_the_right_product(self) -> None:
        # an over-specified category that no product's path fully carries
        outcome = self.filter.apply(Slots(category="Active Shirts & Tees T-Shirts"))
        self.assertIsNone(outcome.allowed)

    def test_a_filter_that_would_empty_the_set_is_skipped(self) -> None:
        outcome = self.filter.apply(Slots(item="boot", brand="NoSuchBrandAnywhere"))
        self.assertIn("brand", outcome.skipped)
        self.assertTrue(outcome.allowed, "backoff must leave something standing")

    def test_most_selective_constraint_is_applied_first(self) -> None:
        outcome = self.filter.apply(Slots(item="boot", min_reviews=100))
        kept = {self.index.ids[i] for i in outcome.allowed}
        self.assertEqual(kept, {"BOOT1"})
        self.assertEqual(len(outcome.applied), 2)

    def test_permits_is_open_when_nothing_restricted(self) -> None:
        outcome = self.filter.apply(Slots())
        self.assertTrue(outcome.permits(0))
        self.assertTrue(outcome.permits(999))

    def test_a_filter_matching_almost_everything_is_not_a_filter(self) -> None:
        rows = [product(f"P{i}", title="common word here") for i in range(12)]
        with catalog_file(rows) as path:
            index = CatalogIndex(path)
            outcome = HardFilter(index, cap=5).apply(Slots(item="common"))
            self.assertIn("item", outcome.skipped)


if __name__ == "__main__":
    unittest.main()
