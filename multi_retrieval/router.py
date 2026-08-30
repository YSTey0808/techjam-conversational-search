"""The public entry point: intent in, ranked candidates out.

Follows the architecture diagram's retrieval stage. Intent selects how heavily
each route counts; the routes run; their results merge into one candidate pool.

The diagram draws vector search under Browsing only. Here all three routes are
available to both tracks and intent changes the *weighting* — a Buying turn can
still lean on semantic similarity once its keyword evidence thins out, and a
Browsing turn should not be blind to an exact word match if the customer
happens to produce one.

There is no reranking stage. The fused pool order is the output, which puts
more weight on fusion than the diagram implies.
"""

from __future__ import annotations

from pathlib import Path

from . import fuse
from .filters import HardFilter
from .index import CatalogIndex
from .routes.category import CategoryRoute
from .routes.keyword import KeywordRoute
from .routes.prior import DEFAULT_WEIGHT, PopularityPrior
from .routes.seeded import SeededRoute
from .types import (
    DEFAULT_TRACKS,
    Candidate,
    DualQuery,
    DualResult,
    TrackConfig,
)

POOL_LIMIT = 200


class DualTrackRetriever:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        tracks: dict[str, TrackConfig] | None = None,
        fusion: str = "rrf",
        keyword_mode: str = "bm25",   # "bm25" | "seeded"
        layered: bool = True,
        pool_limit: int = POOL_LIMIT,
        embedder=None,
        prior_weight: float = DEFAULT_WEIGHT,
        cache_dir: str = ".cache/multi_retrieval",
    ) -> None:
        self.index = CatalogIndex(catalog_path)
        self.tracks = dict(tracks or DEFAULT_TRACKS)
        self.fusion = fusion
        self.layered = layered
        self.pool_limit = pool_limit

        self.keyword_mode = keyword_mode
        self.keyword = (SeededRoute(self.index) if keyword_mode == "seeded"
                        else KeywordRoute(self.index))
        self.category = CategoryRoute(self.index)
        self.filter = HardFilter(self.index)
        self.prior = PopularityPrior(self.index, weight=prior_weight)

        # The vector route is optional: building it encodes the whole catalog,
        # which is the one slow step in this package. Pass an embedder to enable
        # it; leave it out and the two lexical routes carry the query.
        self.vector = None
        if embedder is not None:
            from .routes.vector import VectorRoute
            self.vector = VectorRoute(self.index, embedder, cache_dir=cache_dir)

    # --------------------------------------------------------------- retrieve

    def retrieve(self, query: DualQuery) -> DualResult:
        weights = self.tracks.get(query.intent, DEFAULT_TRACKS[query.intent]).as_dict()

        # Layer one: which products are eligible at all.
        outcome = self.filter.apply(query.slots) if self.layered else None

        # Layer two: the routes rank whatever is eligible.
        routes: dict[str, dict[int, float]] = {}
        if weights.get("keyword", 0.0) > 0:
            routes["keyword"] = self.keyword.search(query.slots)
        if weights.get("category", 0.0) > 0:
            routes["category"] = self.category.search(query.slots)
        if self.vector is not None and weights.get("vector", 0.0) > 0:
            routes["vector"] = self.vector.search(query)

        raw_sizes = {name: len(scores) for name, scores in routes.items()}
        if outcome is not None and outcome.allowed is not None:
            routes = {
                name: {p: s for p, s in scores.items() if p in outcome.allowed}
                for name, scores in routes.items()
            }
            # If filtering silenced every route, the constraints were wrong about
            # this catalog rather than about the customer. Fall back to unfiltered
            # results: a loose list costs a turn, an empty one costs the session.
            if not any(routes.values()):
                routes = {
                    "keyword": self.keyword.search(query.slots),
                    "category": self.category.search(query.slots),
                }
                if self.vector is not None and weights.get("vector", 0.0) > 0:
                    routes["vector"] = self.vector.search(query)
                outcome.skipped.extend(outcome.applied)
                outcome.applied.clear()

        combined = fuse.fuse(routes, weights, mode=self.fusion)
        if not combined:
            combined = self._fallback(query)
        # Applied after fusion, not as a fourth route: under RRF a route votes by
        # rank, and a tie-break prior has to act on magnitude to do its job.
        combined = self.prior.blend(combined, shortlist=self.pool_limit)

        ranked = fuse.order(combined, self.index.ids, min(query.top_k, self.pool_limit))
        items = [
            Candidate(
                parent_asin=self.index.ids[product],
                score=score,
                routes={name: scores[product] for name, scores in routes.items() if product in scores},
            )
            for product, score in ranked
        ]

        return DualResult(
            items=items,
            intent=query.intent,
            pool_size=len(combined),
            filtered_size=len(outcome.allowed) if outcome and outcome.allowed is not None else self.index.size,
            route_sizes=raw_sizes,
            filters_applied=list(outcome.applied) if outcome else [],
            filters_skipped=list(outcome.skipped) if outcome else [],
        )

    def _fallback(self, query: DualQuery) -> dict[int, float]:
        """Nothing matched at all. Return the most-reviewed products.

        Never hand back an empty list: an empty list is a guaranteed miss, while
        a mediocre one still has a chance and costs at most a turn.
        """
        ranked = sorted(range(self.index.size), key=lambda i: (-self.index.reviews[i], i))
        top = ranked[: self.pool_limit]
        return {product: float(len(top) - position) for position, product in enumerate(top)}

    @property
    def diagnostics(self) -> dict:
        detail = self.index.diagnostics()
        detail["vector_route"] = type(self.vector).__name__ if self.vector else None
        detail["fusion"] = self.fusion
        detail["keyword_mode"] = self.keyword_mode
        detail["prior_weight"] = self.prior.weight
        detail["layered"] = self.layered
        return detail


__all__ = ["DualTrackRetriever"]
