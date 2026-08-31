"""Dual-track multi-route retrieval, built from the team architecture diagram.

A second, independent implementation of the retrieval stage. It shares no code
with ``retrieval/`` — its own catalog parsing, its own indexes, its own types —
so the two can be measured against each other honestly.

    from multi_retrieval import DualTrackRetriever, DualQuery, Slots

    # normalised faceted table; the original catalog.jsonl beside it is picked
    # up automatically as the average_rating / rating_number sidecar
    retriever = DualTrackRetriever("data/catalog_normalised.jsonl")
    result = retriever.retrieve(DualQuery(
        slots=Slots(category="Jewelry Necklaces", material="alloy"),
        intent="buying",
        top_k=10,
    ))
"""

from .router import DualTrackRetriever
from .types import (
    BROWSING,
    BUYING,
    Candidate,
    DualQuery,
    DualResult,
    Slots,
    TrackConfig,
)

__all__ = [
    "DualTrackRetriever",
    "DualQuery",
    "DualResult",
    "Slots",
    "Candidate",
    "TrackConfig",
    "BUYING",
    "BROWSING",
]
