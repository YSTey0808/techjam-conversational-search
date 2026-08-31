from __future__ import annotations

import unittest

from multi_retrieval import DualQuery, DualTrackRetriever, Slots
from multi_retrieval.embed import HashingEmbedder
from multi_retrieval.types import DEFAULT_TRACKS, TrackConfig
from tests.fixtures import BOOTS_AND_NECKLACES, catalog_file


class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._catalog = catalog_file(BOOTS_AND_NECKLACES)
        self.path = self._catalog.__enter__()
        self.retriever = DualTrackRetriever(self.path)

    def tearDown(self) -> None:
        self._catalog.__exit__(None, None, None)

    def test_a_distinctive_phrase_is_found(self) -> None:
        result = self.retriever.retrieve(DualQuery(
            slots=Slots(category="Jewelry Necklaces", free_text=["Triple Moon Pentagram Symbol"]),
            intent="buying",
        ))
        self.assertEqual(result.parent_asins[0], "NECK1")

    def test_it_never_returns_an_empty_list(self) -> None:
        """An empty list is a guaranteed miss; a loose one costs a turn."""
        for query in (
            DualQuery(slots=Slots()),
            DualQuery(slots=Slots(category="nothing matches this at all")),
            DualQuery(slots=Slots(free_text=["zzzz qqqq"])),
        ):
            self.assertTrue(self.retriever.retrieve(query).items, query)

    def test_repeated_calls_are_identical(self) -> None:
        query = DualQuery(slots=Slots(category="Hiking Boots"), intent="browsing")
        first = self.retriever.retrieve(query)
        second = self.retriever.retrieve(query)
        self.assertEqual(first.parent_asins, second.parent_asins)

    def test_no_state_leaks_between_calls(self) -> None:
        boots = DualQuery(slots=Slots(category="Hiking Boots"))
        necklaces = DualQuery(slots=Slots(category="Jewelry Necklaces"))
        first = self.retriever.retrieve(boots).parent_asins
        self.retriever.retrieve(necklaces)
        self.assertEqual(self.retriever.retrieve(boots).parent_asins, first)

    def test_top_k_is_respected(self) -> None:
        result = self.retriever.retrieve(DualQuery(slots=Slots(category="Hiking Boots"), top_k=2))
        self.assertLessEqual(len(result.items), 2)

    def test_intent_changes_the_ranking(self) -> None:
        """Buying leans on keyword evidence, browsing on category. If the two
        tracks produced identical output the routing would be decorative."""
        slots = Slots(category="Hiking Boots", free_text=["waterproof leather upper"])
        buying = self.retriever.retrieve(DualQuery(slots=slots, intent="buying", top_k=5))
        browsing = self.retriever.retrieve(DualQuery(slots=slots, intent="browsing", top_k=5))
        self.assertNotEqual(
            [c.score for c in buying.items], [c.score for c in browsing.items],
        )

    def test_an_unknown_intent_falls_back_to_browsing(self) -> None:
        self.assertEqual(DualQuery(slots=Slots(), intent="nonsense").intent, "browsing")

    def test_layering_can_be_switched_off(self) -> None:
        slots = Slots(category="Hiking Boots", min_reviews=100)
        layered = DualTrackRetriever(self.path, layered=True)
        flat = DualTrackRetriever(self.path, layered=False)
        self.assertTrue(layered.retrieve(DualQuery(slots=slots)).filters_applied)
        self.assertFalse(flat.retrieve(DualQuery(slots=slots)).filters_applied)

    def test_result_reports_which_routes_ran(self) -> None:
        result = self.retriever.retrieve(DualQuery(
            slots=Slots(category="Hiking Boots"), intent="buying",
        ))
        self.assertIn("keyword", result.route_sizes)
        self.assertIn("category", result.route_sizes)
        self.assertNotIn("vector", result.route_sizes)   # no embedder supplied

    def test_the_vector_route_joins_when_an_embedder_is_given(self) -> None:
        retriever = DualTrackRetriever(
            self.path, embedder=HashingEmbedder(), cache_dir=str(self.path.parent / "c"),
        )
        result = retriever.retrieve(DualQuery(slots=Slots(category="Hiking Boots")))
        self.assertIn("vector", result.route_sizes)

    def test_custom_track_weights_are_honoured(self) -> None:
        only_category = {
            "buying": TrackConfig(keyword=0.0, category=1.0, vector=0.0),
            "browsing": DEFAULT_TRACKS["browsing"],
        }
        retriever = DualTrackRetriever(self.path, tracks=only_category)
        result = retriever.retrieve(DualQuery(
            slots=Slots(category="Hiking Boots", free_text=["waterproof"]), intent="buying",
        ))
        self.assertNotIn("keyword", result.route_sizes)


if __name__ == "__main__":
    unittest.main()
