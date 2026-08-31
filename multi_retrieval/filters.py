"""The hard-constraint layer.

This is the diagram's second open question made switchable instead of assumed:
should retrieval run hard constraints first and softer ones second?

With ``layered=True`` the hard slots and numeric limits decide which products
are eligible at all, and the routes then rank inside that set. With
``layered=False`` every slot simply feeds the routes and nothing is excluded.
Both are measurable, so the question gets an answer rather than a guess.

**What the gate is allowed to intersect on.** Only the clean, dense, low-
cardinality facets of the normalised catalog:

* ``brand``      -- near unique; the strongest possible gate when stated
* ``department`` -- mapped onto the ``audience`` enum (women / men / kids ...)
* ``category``   -- the 8-value enum, and only when the customer said it
  verbatim (``Slots.category_trusted``)

``item`` and the descriptive facets (``color``, ``material``, ``style``,
``use_case``) are deliberately NOT gates: they are sparse and multi-valued, so
an exact match would delete the ~40-60% of products that simply never state the
attribute. They earn their weight through the routes instead.

Three rules apply to every filter here:

* **A filter that would empty the candidate set is skipped, not enforced.**
  Returning nothing is a guaranteed miss; returning a loose list costs one turn.
* **A filter matching most of the catalog is not a filter.** A lone facet that
  keeps more than ``non_selective_fraction`` of the rows is dropped -- it would
  narrow to a healthy-looking set that could still exclude the right answer,
  and nothing here would notice.
* **Unknown is not a violation.** A product with no price is kept under a
  budget; a brand or audience the catalog never lists is skipped, not enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import CatalogIndex, tokens
from .types import Slots

# A lone facet keeping more than this share of the catalog is treated as
# non-selective and dropped. The normalised enums are clean, so this sits well
# above the old text-match cap: "audience = women" (56%) and "category =
# Clothing" (46%) are still legitimate customer-stated constraints.
NON_SELECTIVE_FRACTION = 0.65

# Free-form department phrasings -> the `audience` enum in the normalised
# catalog. Anything not here maps nowhere and the department gate is skipped.
DEPARTMENT_TO_AUDIENCE = {
    "women": "women", "woman": "women", "womens": "women", "women's": "women",
    "female": "women", "females": "women", "ladies": "women", "lady": "women",
    "men": "men", "man": "men", "mens": "men", "men's": "men",
    "male": "men", "males": "men", "gentlemen": "men",
    "girls": "girls", "girl": "girls", "girls'": "girls",
    "boys": "boys", "boy": "boys", "boys'": "boys",
    "baby": "baby", "babies": "baby", "infant": "baby", "infants": "baby",
    "newborn": "baby", "toddler": "kids", "toddlers": "kids",
    "kids": "kids", "kid": "kids", "children": "kids", "child": "kids",
    "youth": "kids", "junior": "kids", "juniors": "kids",
    "unisex": "unisex", "everyone": "unisex",
}


@dataclass
class FilterOutcome:
    allowed: set[int] | None = None          # None means "no restriction"
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def permits(self, product: int) -> bool:
        return self.allowed is None or product in self.allowed


class HardFilter:
    def __init__(
        self,
        index: CatalogIndex,
        *,
        non_selective_fraction: float = NON_SELECTIVE_FRACTION,
    ) -> None:
        self.index = index
        self.non_selective_fraction = non_selective_fraction

    @property
    def _cap(self) -> int:
        fraction = self.non_selective_fraction
        if 0.0 < fraction < 1.0:
            return int(fraction * self.index.size)
        return self.index.size

    # ------------------------------------------------------------------ facets

    def _category_match(self, value: str) -> set[int] | None:
        """Products whose `category` enum carries every word of the slot.

        Intersection, not union: "bags luggage" should resolve to the
        "Bags & Luggage" enum only, not everything tagged "bags". An unknown
        word is skipped -- it is our vocabulary gap, not evidence of absence.
        """
        words = sorted(set(tokens(value)))
        if not words:
            return None
        current: set[int] | None = None
        for word in words:
            postings = self.index.category_enum_postings.get(word)
            if postings is None:
                continue
            found = set(postings)
            current = found if current is None else (current & found)
            if not current:
                return None                   # over-narrowed; caller skips it
        return current

    def _department_match(self, value: str) -> set[int] | None:
        """Products whose `audience` enum matches the stated department."""
        wanted = {
            DEPARTMENT_TO_AUDIENCE[word]
            for word in tokens(value)
            if word in DEPARTMENT_TO_AUDIENCE
        }
        if not wanted:
            return None
        matched: set[int] = set()
        for audience in wanted:
            postings = self.index.audience_postings.get(audience)
            if postings is not None:
                matched.update(postings)
        return matched or None

    def _brand_match(self, value: str) -> set[int] | None:
        """Products whose `brand` is exactly the stated one (case/space-folded)."""
        candidates = {value.strip().lower(), " ".join(tokens(value))}
        for key in candidates:
            postings = self.index.brand_postings.get(key)
            if postings is not None:
                return set(postings)
        return None

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
        cap = self._cap

        matchers = {
            "category": self._category_match,
            "department": self._department_match,
            "brand": self._brand_match,
        }

        candidates: list[tuple[str, set[int]]] = []
        for name, value in slots.hard.items():
            matcher = matchers.get(name)
            matched = matcher(value) if matcher else None
            if matched is None:
                outcome.skipped.append(name)
                continue
            if len(matched) > cap:
                outcome.skipped.append(name)   # keeps most of the catalog: not a gate
                continue
            candidates.append((name, matched))

        # Explicit numeric limits are always considered -- they are stated
        # constraints, not fuzzy matches, so the cap does not apply to them.
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


__all__ = ["HardFilter", "FilterOutcome", "DEPARTMENT_TO_AUDIENCE"]
