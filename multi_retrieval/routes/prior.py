"""Popularity prior, applied to the fused pool.

Not a retrieval route: it fetches nothing and has no opinion about the query.
It breaks ties the text routes cannot, which is why it is applied after fusion
rather than mixed in as a fourth route -- under RRF a route contributes by rank,
and this needs to contribute by magnitude.

The hidden targets are real purchase records, so they skew heavily towards
products people actually buy. On a first turn, when nothing has been said but a
category, "most bought here" is a genuinely strong guess, and even later it
separates candidates that matched identical evidence.

Measured on the public set: 0.8174 without it, 0.8396 at the shipped weight.

On the weight, an honest note. Letting popularity dominate outright scores
marginally higher (0.8450 at weight 0.8, and 0.8850 in a variant that reordered
the final ten by popularity alone) because it lifts hit rate. It also overrules
real evidence -- with a dominant prior, a heavily reviewed necklace outranks the
one carrying the exact phrase the customer just quoted, and MRR falls from 0.659
to 0.620. The +0.005 is inside the noise band measured elsewhere in this repo,
so the shipped weight keeps the prior a tie-break and lets evidence win.
"""

from __future__ import annotations

import math

from ..index import CatalogIndex

DEFAULT_WEIGHT = 0.2


class PopularityPrior:
    name = "prior"

    def __init__(self, index: CatalogIndex, *, weight: float = DEFAULT_WEIGHT) -> None:
        self.weight = weight
        values = [math.log1p(count) for count in index.reviews]
        largest = max(values) if values else 0.0
        self.scores = [value / largest for value in values] if largest else [0.0] * len(values)

    def blend(self, combined: dict[int, float], *, shortlist: int = 200) -> dict[int, float]:
        """Re-order the strongest candidates by fused evidence plus popularity.

        Two details matter, and getting either wrong makes popularity beat
        evidence rather than break its ties.

        **Only the shortlist.** Applied across the whole pool, a heavily reviewed
        but irrelevant product climbs into the top ten and displaces a relevant
        one. Their ``rank.py`` gets this right the same way: ``retrieve()`` hands
        it a few hundred candidates and popularity only ever reorders those.

        **Normalise the fused scores first.** RRF puts rank 1 and rank 2 about
        0.0003 apart, so an unscaled prior of 0.01 simply overrules the routes.
        Rescaling the shortlist to [0, 1] means a decisive route hit still wins
        and the prior only decides between candidates the routes could not
        separate.
        """
        if self.weight <= 0.0 or not combined:
            return combined

        # Tie-break explicitly on product index. This sort decides which
        # candidates survive the cut, not merely how they are displayed -- under
        # RRF many products tie exactly, so without a stated rule membership at
        # the boundary falls out of dict insertion order.
        ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:shortlist]
        values = [score for _, score in ranked]
        largest, smallest = max(values), min(values)
        spread = largest - smallest

        blended: dict[int, float] = {}
        for product, score in ranked:
            base = (score - smallest) / spread if spread > 0 else 1.0
            blended[product] = base + self.weight * self.scores[product]
        return blended


__all__ = ["PopularityPrior", "DEFAULT_WEIGHT"]
