from __future__ import annotations

import unittest

from multi_retrieval.filters import HardFilter
from multi_retrieval.index import CatalogIndex
from multi_retrieval.types import Slots
from tests.fixtures import (
    NORMALISED_BOOTS_AND_NECKLACES,
    SIDECAR_BOOTS_AND_NECKLACES,
    normalised_catalog_file,
    normalised_product,
)


class FacetGateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = normalised_catalog_file(
            NORMALISED_BOOTS_AND_NECKLACES, raw_rows=SIDECAR_BOOTS_AND_NECKLACES
        )
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path, raw_catalog_path=self.path.with_name("catalog.jsonl"))
        self.filter = HardFilter(self.index)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def kept(self, outcome) -> set[str]:
        return {self.index.ids[i] for i in outcome.allowed}


class BrandGateTest(FacetGateTestCase):
    def test_brand_keeps_only_that_brand(self) -> None:
        outcome = self.filter.apply(Slots(brand="moonco"))
        self.assertIn("brand", outcome.applied)
        self.assertEqual(self.kept(outcome), {"NECK1"})

    def test_brand_folds_case_and_spacing(self) -> None:
        outcome = self.filter.apply(Slots(brand="  MoonCo "))
        self.assertEqual(self.kept(outcome), {"NECK1"})

    def test_an_unknown_brand_is_skipped_not_enforced(self) -> None:
        outcome = self.filter.apply(Slots(brand="NoSuchBrandAnywhere"))
        self.assertIn("brand", outcome.skipped)
        self.assertIsNone(outcome.allowed)          # backoff leaves the pool open


class DepartmentGateTest(FacetGateTestCase):
    def test_department_maps_to_the_audience_enum(self) -> None:
        outcome = self.filter.apply(Slots(department="women"))
        self.assertIn("department", outcome.applied)
        self.assertEqual(self.kept(outcome), {"NECK1", "NECK2"})

    def test_department_synonyms_resolve(self) -> None:
        for phrasing in ("womens", "ladies", "for women"):
            outcome = self.filter.apply(Slots(department=phrasing))
            self.assertEqual(self.kept(outcome), {"NECK1", "NECK2"}, phrasing)

    def test_an_unmappable_department_is_skipped(self) -> None:
        outcome = self.filter.apply(Slots(department="astronauts"))
        self.assertIn("department", outcome.skipped)
        self.assertIsNone(outcome.allowed)


class CategoryGateTest(FacetGateTestCase):
    def test_category_does_nothing_unless_it_is_trusted(self) -> None:
        outcome = self.filter.apply(Slots(category="Jewelry"))
        self.assertIsNone(outcome.allowed)
        self.assertNotIn("category", outcome.applied)

    def test_a_verbatim_category_gates_on_the_enum(self) -> None:
        outcome = self.filter.apply(Slots(category="Jewelry", category_trusted=True))
        self.assertIn("category", outcome.applied)
        self.assertEqual(self.kept(outcome), {"NECK1", "NECK2"})

    def test_a_multi_word_category_intersects_its_tokens(self) -> None:
        rows = [
            normalised_product("A", category="Bags & Luggage", brand="a"),
            normalised_product("B", category="Accessories", brand="b"),
            normalised_product("C", category="Bags & Luggage", brand="c"),
        ]
        with normalised_catalog_file(rows) as path:
            index = CatalogIndex(path)
            # non_selective_fraction=1.0: this test is about token intersection,
            # not selectivity, and 2 of 3 rows would trip the default cap.
            outcome = HardFilter(index, non_selective_fraction=1.0).apply(
                Slots(category="bags luggage", category_trusted=True)
            )
            self.assertEqual({index.ids[i] for i in outcome.allowed}, {"A", "C"})

    def test_an_unknown_category_word_cannot_exclude_anything(self) -> None:
        outcome = self.filter.apply(
            Slots(category="Necklaces Pendants", category_trusted=True)
        )
        # neither word is one of the 8 category enums -> nothing to gate on
        self.assertIsNone(outcome.allowed)


class NonSelectiveTest(unittest.TestCase):
    def test_a_facet_matching_most_of_the_catalog_is_not_a_gate(self) -> None:
        rows = [normalised_product(f"W{i}", audience="women", brand=f"b{i}") for i in range(9)]
        rows += [normalised_product("M0", audience="men", brand="bm")]
        with normalised_catalog_file(rows) as path:
            index = CatalogIndex(path)
            outcome = HardFilter(index).apply(Slots(department="women"))   # 90% of rows
            self.assertIn("department", outcome.skipped)
            self.assertIsNone(outcome.allowed)


class NumericGateTest(FacetGateTestCase):
    def test_min_reviews_reads_the_sidecar(self) -> None:
        outcome = self.filter.apply(Slots(min_reviews=100))
        self.assertEqual(self.kept(outcome), {"BOOT1", "NECK1", "NECK2"})

    def test_min_rating_reads_the_sidecar(self) -> None:
        outcome = self.filter.apply(Slots(min_rating=4.3))
        self.assertEqual(self.kept(outcome), {"BOOT1", "NECK1"})

    def test_a_product_with_no_price_survives_a_budget(self) -> None:
        outcome = self.filter.apply(Slots(price_max=25.0))
        self.assertIn("price_max", outcome.applied)
        self.assertIn(2, outcome.allowed)          # BOOT3, price null
        self.assertNotIn(0, outcome.allowed)       # BOOT1 at 89.0 is over

    def test_a_budget_that_nothing_meets_is_skipped_not_enforced(self) -> None:
        outcome = self.filter.apply(Slots(price_max=0.01))
        self.assertTrue(outcome.allowed is None or outcome.allowed)


class CombinationTest(FacetGateTestCase):
    def test_the_most_selective_constraint_is_applied_first(self) -> None:
        outcome = self.filter.apply(Slots(brand="vibram", department="men"))
        self.assertEqual(self.kept(outcome), {"BOOT1"})
        self.assertEqual(outcome.applied[0], "brand")   # 1 row beats 2

    def test_a_gate_that_would_empty_the_set_is_skipped(self) -> None:
        # men's brand "chainco" (women's jewelry) -> intersection is empty
        outcome = self.filter.apply(Slots(brand="chainco", department="men"))
        self.assertIn("department", outcome.skipped)
        self.assertEqual(self.kept(outcome), {"NECK2"})

    def test_no_slots_means_no_restriction(self) -> None:
        outcome = self.filter.apply(Slots())
        self.assertIsNone(outcome.allowed)
        self.assertTrue(outcome.permits(0))
        self.assertTrue(outcome.permits(999))


class NoSidecarTest(unittest.TestCase):
    """Without the raw sidecar the rating columns are empty. The rating gates
    must degrade quietly, never crash and never wrongly empty the pool."""

    def setUp(self) -> None:
        self._catalog = normalised_catalog_file(NORMALISED_BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.index = CatalogIndex(self.path)          # no raw_catalog_path
        self.filter = HardFilter(self.index)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_min_rating_is_inert_without_ratings(self) -> None:
        outcome = self.filter.apply(Slots(min_rating=4.0))
        # every rating is None -> "unknown is not a violation" keeps all rows
        self.assertEqual(len(outcome.allowed or range(self.index.size)), self.index.size)

    def test_min_reviews_backs_off_rather_than_returning_nothing(self) -> None:
        outcome = self.filter.apply(Slots(min_reviews=100))
        self.assertTrue(outcome.allowed is None or outcome.allowed)

    def test_brand_still_gates(self) -> None:
        outcome = self.filter.apply(Slots(brand="moonco"))
        self.assertEqual({self.index.ids[i] for i in outcome.allowed}, {"NECK1"})


if __name__ == "__main__":
    unittest.main()
