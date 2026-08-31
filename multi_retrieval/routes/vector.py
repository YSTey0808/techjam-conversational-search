"""Vector route: dense cosine similarity over the whole catalog.

Available to both tracks. The architecture diagram draws this under Browsing
only, but intent should change how heavily a route counts, not whether it
exists — a Buying turn whose keyword evidence runs thin still benefits from
semantic neighbours, and a Browsing turn should not ignore an exact word match
if the customer happens to produce one.

The catalog matrix is normally the precomputed nomic file (``precomputed=``);
without it the route encodes ``index.embed_text`` live with whatever embedder is
passed. Vectors are L2-normalised, so a dot product is the cosine and the whole
search is one matmul over 50,000 x 768 — sub-millisecond, no index structure.
"""

from __future__ import annotations

import numpy as np

from ..embed import Embedder, VectorStore
from ..index import CatalogIndex
from ..types import DualQuery

ROUTE_LIMIT = 500


class VectorRoute:
    name = "vector"

    def __init__(
        self,
        index: CatalogIndex,
        embedder: Embedder,
        *,
        cache_dir: str = ".cache/multi_retrieval",
        limit: int = ROUTE_LIMIT,
        precomputed: tuple[str, str] | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.limit = limit
        self.store = VectorStore(embedder, cache_dir=cache_dir, catalog_path=index.catalog_path)
        if precomputed is not None:
            matrix_path, ids_path = precomputed
            self.store.load_precomputed(matrix_path, ids_path, index.ids)
            if self.store.matrix.shape[1] != embedder.dimension:
                raise RuntimeError(
                    f"query encoder is {embedder.dimension}-dim but the "
                    f"precomputed matrix is {self.store.matrix.shape[1]}-dim"
                )
        else:
            self.store.build(index.embed_text, index.ids)

    def query_text(self, query: DualQuery) -> str:
        """What we actually embed.

        Slot values first, then the raw sentence. The raw message is included
        because the customer's own phrasing carries intent that the slots have
        already flattened away — and this is the one route that can use it.
        """
        parts = list(query.slots.phrases)
        if query.raw_message:
            parts.append(query.raw_message)
        return " ".join(parts).strip()

    def search(self, query: DualQuery) -> dict[int, float]:
        text = self.query_text(query)
        if not text:
            return {}
        vector = self.embedder.encode([text])[0]
        scores = self.store.similarity(vector)

        count = min(self.limit, scores.shape[0])
        # argpartition finds the top-N without sorting all 50,000.
        top = np.argpartition(-scores, count - 1)[:count]
        return {int(i): float(scores[i]) for i in top}


__all__ = ["VectorRoute"]
