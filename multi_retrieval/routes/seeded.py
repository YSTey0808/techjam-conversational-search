"""Seeded progressive intersection -- ported from the design in starter/retrieve.py.

The BM25 route searches the whole catalog and ranks loosely. This one narrows
instead: start from the category bucket, then intersect the products matching
each stated fact, most informative first.

The reason it is worth having is not the intersection itself but the **backoff**.
A category gate applied as a pre-filter can narrow to a healthy-looking pool that
simply excludes the right answer, and nothing notices because the pool is not
empty. Here the constraints act as a check on the category: if intersecting the
facts against the seeded pool empties it, that is evidence the seed was wrong, and
the ladder drops it.

Backoff order, loosening by the smallest amount each time:

  1. drop the least informative fact, repeatedly
  2. retry with no category seed at all
  3. give up and return nothing, leaving the other routes to answer

That third step matters: this route is allowed to have no opinion. It never
returns a wrong answer to avoid returning none.
"""

from __future__ import annotations

from ..index import CatalogIndex, tokens
from ..types import Slots

# Cap on how many products one fact may resolve to. A fact matching more than
# this is too common to narrow anything, and materialising the set is wasteful.
MATCH_CAP = 20000
ROUTE_LIMIT = 500


class SeededRoute:
    name = "keyword"

    def __init__(self, index: CatalogIndex, *, cap: int = MATCH_CAP,
                 limit: int = ROUTE_LIMIT) -> None:
        self.index = index
        self.cap = cap
        self.limit = limit

    # ------------------------------------------------------------------ parts

    def _category_seed(self, slots: Slots) -> set[int] | None:
        """Products whose category path carries every word of the category slot."""
        words = sorted(set(tokens(slots.category)))
        if not words:
            return None
        current: set[int] | None = None
        for word in words:
            postings = self.index.category_postings.get(word)
            if postings is None:
                continue                       # unknown word is not evidence of absence
            found = set(postings)
            current = found if current is None else (current & found)
            if not current:
                return None
        return current

    def _matches(self, phrase: str) -> set[int] | None:
        """Products whose text contains this phrase, as a set."""
        words = tokens(phrase)
        if not words:
            return None
        expression = '"' + " ".join(words) + '"' if len(words) > 1 else f'"{words[0]}"'
        hits = self.index.search_bm25(expression, self.cap)
        return {product for product, _ in hits} or None

    # ------------------------------------------------------------------ score

    def search(self, slots: Slots) -> dict[int, float]:
        resolved: list[tuple[str, set[int], float]] = []
        for phrase in slots.phrases:
            matched = self._matches(phrase)
            if matched is None:
                continue                       # unfindable: our indexing gap, not proof
            resolved.append((phrase, matched, self.index.idf(len(matched))))
        if not resolved:
            return {}

        # Most informative first, so the pool shrinks fastest on the strongest
        # evidence and backoff always drops the weakest fact.
        resolved.sort(key=lambda item: -item[2])
        seed = self._category_seed(slots)

        for use_seed in (True, False):
            if use_seed and seed is None:
                continue
            working = list(resolved)
            while working:
                pool = set(seed) if (use_seed and seed) else None
                for _, matched, _weight in working:
                    pool = set(matched) if pool is None else (pool & matched)
                    if not pool:
                        break
                if pool:
                    return self._score(pool, resolved)
                if len(working) <= 1:
                    break
                working.pop()                  # drop the least informative fact
        return {}

    def _score(self, pool: set[int], resolved) -> dict[int, float]:
        """Summed IDF of every fact each survivor actually matches."""
        scores: dict[int, float] = {product: 0.0 for product in pool}
        for _phrase, matched, weight in resolved:
            for product in pool & matched:
                scores[product] += weight
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return dict(ordered[: self.limit])


__all__ = ["SeededRoute"]
