"""Category route: search the inverted index over category paths.

Scores by IDF-weighted overlap rather than plain token count, because category
paths are wildly uneven. Every product in this catalog sits under
"Clothing, Shoes & Jewelry", so the word "jewelry" appears in all 50,000 paths
and carries no information at all — IDF drives it to zero automatically, where
a plain count would treat it as evidence.

This route only ever contributes score. It never removes a candidate; that is
``filters.py``'s job, and only for slots marked hard.
"""

from __future__ import annotations

from ..index import CatalogIndex, tokens
from ..types import Slots

ROUTE_LIMIT = 500

# Slots that describe what kind of thing the customer wants. Colour or material
# would only add noise here — they say nothing about which category path a
# product sits under.
CATEGORY_SLOTS = ("category", "item", "department")


class CategoryRoute:
    name = "category"

    def __init__(self, index: CatalogIndex, *, limit: int = ROUTE_LIMIT) -> None:
        self.index = index
        self.limit = limit

    def search(self, slots: Slots) -> dict[int, float]:
        query_tokens = self._tokens(slots)
        if not query_tokens:
            return {}

        weights = {
            token: self.index.idf(self.index.category_document_frequency(token))
            for token in query_tokens
        }
        total = sum(weights.values())
        if total <= 0:
            # Every query word appears in every category path, so none of them
            # separates anything. Better to say nothing than to rank noise.
            return {}

        accumulated: dict[int, float] = {}
        for token, weight in weights.items():
            if weight <= 0:
                continue
            postings = self.index.category_postings.get(token)
            if postings is None:
                continue
            for product in postings:
                accumulated[product] = accumulated.get(product, 0.0) + weight

        scored = {product: value / total for product, value in accumulated.items()}
        ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        return dict(ordered[: self.limit])

    def _tokens(self, slots: Slots) -> list[str]:
        """Sorted and deduplicated.

        Sorted matters: the float additions below happen in iteration order, and
        a set's order varies between processes because Python randomises string
        hashing. Without this the last bits of a score drift between runs, which
        is enough to flip a tie.
        """
        found: set[str] = set()
        for name in CATEGORY_SLOTS:
            value = getattr(slots, name, "")
            if value:
                found.update(tokens(value))
        return sorted(found)


__all__ = ["CategoryRoute", "CATEGORY_SLOTS"]
