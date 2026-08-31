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

import json
from pathlib import Path

from . import fuse
from .embed import DEFAULT_EMBEDDINGS_DIR
from .filters import HardFilter
from .index import CatalogIndex
from .routes.category import CategoryRoute
from .routes.keyword import KeywordRoute
from .routes.prior import DEFAULT_WEIGHT, PopularityPrior
from .types import (
    DEFAULT_TRACKS,
    Candidate,
    DualQuery,
    DualResult,
    TrackConfig,
)

POOL_LIMIT = 200


def _numeric_sidecar(
    catalog_path: str | Path,
    explicit: str | Path | None,
) -> str | Path | None:
    """The raw catalog to read average_rating / rating_number from.

    Explicit wins. Otherwise, when the primary catalog is the normalised table,
    probe for the original ``catalog.jsonl`` beside it -- present in every repo
    checkout -- so the popularity prior and rating gates keep working without
    the caller wiring the second path by hand.
    """
    if explicit is not None:
        return explicit
    primary = Path(catalog_path)
    sibling = primary.with_name("catalog.jsonl")
    if sibling != primary and sibling.exists():
        return sibling
    return None


class DualTrackRetriever:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog_normalised.jsonl",
        *,
        raw_catalog_path: str | Path | None = None,
        tracks: dict[str, TrackConfig] | None = None,
        fusion: str = "rrf",
        layered: bool = True,
        pool_limit: int = POOL_LIMIT,
        embedder=None,
        embeddings_dir: str | Path | None = DEFAULT_EMBEDDINGS_DIR,
        prior_weight: float = DEFAULT_WEIGHT,
        cache_dir: str = ".cache/multi_retrieval",
    ) -> None:
        self.index = CatalogIndex(
            catalog_path,
            raw_catalog_path=_numeric_sidecar(catalog_path, raw_catalog_path),
        )
        self.tracks = dict(tracks or DEFAULT_TRACKS)
        self.fusion = fusion
        self.layered = layered
        self.pool_limit = pool_limit

        self.keyword = KeywordRoute(self.index)
        self.category = CategoryRoute(self.index)
        self.filter = HardFilter(self.index)
        self.prior = PopularityPrior(self.index, weight=prior_weight)

        # Three routes by default. The vector route:
        #   * `embedder` given        -> encode the catalog live with it
        #     (tests pass HashingEmbedder; slow with a real model)
        #   * otherwise               -> adopt the precomputed nomic matrix in
        #     `embeddings_dir` and encode queries with NomicEmbedder
        # It falls back to two routes -- no error -- when the embedding files are
        # absent, sentence-transformers is not installed, or the catalog is not
        # the one the vectors were built for (a test fixture).
        self.vector = None
        if embedder is not None:
            from .routes.vector import VectorRoute
            self.vector = VectorRoute(self.index, embedder, cache_dir=cache_dir)
        elif embeddings_dir is not None:
            self.vector = self._precomputed_vector_route(embeddings_dir, cache_dir)

    def _precomputed_vector_route(self, embeddings_dir, cache_dir):
        base = Path(embeddings_dir)
        matrix_path = base / "v2_nomic.npy"
        ids_path = base / "v2_nomic_ids.json"
        if not (matrix_path.exists() and ids_path.exists()):
            return None
        try:
            stored = len(json.loads(ids_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
        if stored != self.index.size:
            return None                       # not this catalog's vectors
        try:
            from .embed import NomicEmbedder
            from .routes.vector import VectorRoute
            return VectorRoute(
                self.index, NomicEmbedder(), cache_dir=cache_dir,
                precomputed=(str(matrix_path), str(ids_path)),
            )
        except ImportError:
            return None                       # deps not installed: run without it

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
        detail["prior_weight"] = self.prior.weight
        detail["layered"] = self.layered
        return detail


__all__ = ["DualTrackRetriever"]
