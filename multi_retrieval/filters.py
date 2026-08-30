"""The hard-constraint layer.

This is the diagram's second open question made switchable instead of assumed:
should retrieval run hard constraints first and softer ones second?

With ``layered=True`` the hard slots and numeric limits decide which products
are eligible at all, and the routes then rank inside that set. With
``layered=False`` every slot simply feeds the routes and nothing is excluded.
Both are measurable, so the question gets an answer rather than a guess.

Two rules apply to every filter here:

* **A filter that would empty the candidate set is skipped, not enforced.**
  Returning nothing is a guaranteed miss; returning a loose list costs one turn.
* **Backoff protects against an empty pool, not a wrong one.** A filter can
  narrow to a healthy-looking set that simply excludes the right answer, and
  nothing here will notice. That is why `category` is not a hard slot: see the
  note on HARD_SLOTS in types.py.
* **Unknown is not a violation.** Only 10,410 of 50,000 products carry a price,
  so treating a missing price as "over budget" would discard four fifths of the
  catalog for a constraint it was never checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import CatalogIndex, tokens
from .routes.keyword import build_expression
from .types import Slots

# How many products a text filter may resolve to before we stop treating it as
# selective. A "filter" matching half the catalog is not filtering.
TEXT_FILTER_CAP = 8000


@dataclass
class FilterOutcome:
    allowed: set[int] | None = None          # None means "no restriction"
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def permits(self, product: int) -> bool:
        return self.allowed is None or product in self.allowed


class HardFilter:
    def __init__(self, index: CatalogIndex, *, cap: int = TEXT_FILTER_CAP) -> None:
        self.index = index
        self.cap = cap

    # ------------------------------------------------------------------ text

    def _category_match(self, value: str) -> set[int] | None:
        """Products whose category path carries every word of the slot."""
        words = sorted(set(tokens(value)))
        if not words:
            return None
        current: set[int] | None = None
        for word in words:
            postings = self.index.category_postings.get(word)
            if postings is None:
                continue                      # unknown word: not evidence of absence
            found = set(postings)
            current = found if current is None else (current & found)
            if not current:
                return None                   # over-narrowed; let the caller skip it
        return current

    def _text_match(self, value: str) -> set[int] | None:
        """Products whose indexed text contains the slot value."""
        expression = build_expression([value])
        if not expression:
            return None
        # cap + 1, so the caller can tell "exactly at the cap" from "more than
        # the cap". Fetching only `cap` rows would make the over-broad check
        # below unreachable, and a filter matching half the catalog would be
        # applied as though it were selective.
        hits = self.index.search_bm25(expression, self.cap + 1)
        return {product for product, _ in hits} or None

    # --------------------------------------------------------------- numeric

    def _numeric_match(self, slots: Slots) -> dict[str, set[int]]:
        found: dict[str, set[int]] = {}
        every = range(self.index.size)

        if slots.price_max is not None:
            limit = float(slots.price_max)
            # A product with no price is kept: unknown is not a violation.
            found["price_max"] = {
                i for i in every
                if self.index.price[i] is None or self.index.price[i] <= limit
            }
        if slots.min_rating is not None:
            limit = float(slots.min_rating)
            found["min_rating"] = {
                i for i in every
                if self.index.rating[i] is None or self.index.rating[i] >= limit
            }
        if slots.min_reviews is not None:
            limit = float(slots.min_reviews)
            found["min_reviews"] = {i for i in every if self.index.reviews[i] >= limit}
        return found

    # ------------------------------------------------------------------ apply

    def apply(self, slots: Slots) -> FilterOutcome:
        """Narrow the eligible set, most selective constraint first."""
        outcome = FilterOutcome()

        candidates: list[tuple[str, set[int]]] = []
        for name, value in slots.hard.items():
            matched = self._category_match(value) if name == "category" else self._text_match(value)
            if matched is None:
                outcome.skipped.append(name)
                continue
            if len(matched) > self.cap:
                outcome.skipped.append(name)   # too broad to be a filter
                continue
            candidates.append((name, matched))

        for name, matched in self._numeric_match(slots).items():
            candidates.append((name, matched))

        # Smallest set first: the pool shrinks fastest on the strongest evidence,
        # and anything dropped by backoff is the least selective constraint.
        candidates.sort(key=lambda item: len(item[1]))

        for name, matched in candidates:
            merged = matched if outcome.allowed is None else (outcome.allowed & matched)
            if not merged:
                outcome.skipped.append(name)
                continue
            outcome.allowed = merged
            outcome.applied.append(name)

        return outcome


__all__ = ["HardFilter", "FilterOutcome"]
