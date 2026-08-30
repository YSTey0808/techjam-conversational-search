"""Data contract for the dual-track retriever.

Follows the team architecture diagram: the state component fills in `Slots` plus
an intent, and retrieval routes on that intent.

This package shares no code with ``retrieval/``. It is a second, independent
implementation of the retrieval stage so the two can be measured against each
other; duplication between them is the point, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BUYING = "buying"
BROWSING = "browsing"
INTENTS = (BUYING, BROWSING)

# The diagram asks whether retrieval should run hard constraints first and the
# softer ones second. These two groups are what that question is about; the
# split is a default that DualTrackRetriever can override.
#
# `category` is NOT here by default, and this was the single most expensive
# mistake in this package. With an INFERRED category filtering the pool, the
# target survived only 8% of turns in the sessions multi_retrieval failed -- while a
# route had already surfaced it 99.2% of the time. Requiring a product's path to
# carry every word of a guessed "Active Shirts & Tees T-Shirts" throws away the
# right answer, and backoff never notices because the pool is not empty, just
# wrong. Measured inside the pipeline: 0.7478 filtering, 0.8601 not.
#
# A VERBATIM category is a different thing and is worth filtering on: measured
# 0.7573 not filtering against 0.8396 filtering. Callers opt in per query with
# Slots.category_trusted.
HARD_SLOTS = ("item", "brand", "department")
SOFT_SLOTS = ("color", "material", "size", "fit", "occasion",
              "season", "performance", "care", "style", "pattern")


@dataclass
class Slots:
    """What the customer has told us, in structured form.

    Mirrors the slot list on the architecture diagram. Anything the extractor
    could not place goes in ``free_text`` — it still reaches the keyword and
    vector routes, so an unslotted phrase is never simply thrown away.
    """

    item: str = ""
    category: str = ""
    brand: str = ""
    department: str = ""              # gender / age, usually from the profile

    color: str = ""
    material: str = ""
    size: str = ""
    fit: str = ""
    occasion: str = ""
    season: str = ""
    performance: str = ""
    care: str = ""
    style: str = ""
    pattern: str = ""

    price_max: float | None = None
    min_rating: float | None = None
    min_reviews: int | None = None

    free_text: list[str] = field(default_factory=list)

    # Whether `category` came from the customer word-for-word, or was inferred.
    # Only a verbatim category is safe to filter on -- see the note above.
    category_trusted: bool = False

    # ------------------------------------------------------------------ views

    def named(self, names: tuple[str, ...]) -> dict[str, str]:
        """The named text slots that actually hold a value."""
        found: dict[str, str] = {}
        for name in names:
            value = getattr(self, name, "")
            if isinstance(value, str) and value.strip():
                found[name] = value.strip()
        return found

    @property
    def hard(self) -> dict[str, str]:
        """Slots allowed to remove candidates.

        `category` joins only when the caller vouches that the customer said it
        verbatim. Filtering on an inferred category cost 0.11: the target
        survived 8% of turns while a route had already found it 99.2% of the
        time. Filtering on a verbatim one is worth about the same in the other
        direction, so the distinction is trust, not the field.
        """
        names = HARD_SLOTS + (("category",) if self.category_trusted else ())
        return self.named(names)

    @property
    def soft(self) -> dict[str, str]:
        return self.named(SOFT_SLOTS)

    @property
    def phrases(self) -> list[str]:
        """Every text value plus free text, for the keyword and vector routes."""
        values = list(self.hard.values()) + list(self.soft.values())
        values.extend(text.strip() for text in self.free_text if text and text.strip())
        return values

    @property
    def numeric(self) -> dict[str, float]:
        found: dict[str, float] = {}
        if self.price_max is not None:
            found["price_max"] = float(self.price_max)
        if self.min_rating is not None:
            found["min_rating"] = float(self.min_rating)
        if self.min_reviews is not None:
            found["min_reviews"] = float(self.min_reviews)
        return found

    def is_empty(self) -> bool:
        return not self.phrases and not self.numeric


@dataclass
class DualQuery:
    """One retrieval request. ``intent`` decides how the routes are weighted."""

    slots: Slots = field(default_factory=Slots)
    intent: str = BROWSING
    raw_message: str = ""
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            self.intent = BROWSING


@dataclass(frozen=True)
class TrackConfig:
    """How much each route counts for one intent.

    All three routes are available to both tracks. The diagram draws vector
    under Browsing only, but multi-route retrieval means intent changes the
    weighting, not which routes exist — a Buying turn can still benefit from
    semantic similarity once its keyword evidence runs out.
    """

    keyword: float = 1.0
    category: float = 0.5
    vector: float = 0.3

    def as_dict(self) -> dict[str, float]:
        return {"keyword": self.keyword, "category": self.category, "vector": self.vector}


# Swept against the public set with scripts/score_multi_retrieval.py. The first
# guesses (buying 1.0/0.5/0.3, browsing 0.3/0.7/1.0) scored 0.7617; these score
# 0.8174. Keyword evidence deserves far more weight than the diagram's emphasis
# suggested, and a browsing turn leaning hard on vector measurably hurts (0.806).
#
# Honest caveat on the difference between the two tracks: separating them is
# worth +0.0012 over using identical weights everywhere, which is inside the
# run-to-run noise band. The mechanism is real and the routing works, but on
# this evaluator it is not currently earning its keep. Do not read these two
# rows as evidence that dual-track routing pays.
DEFAULT_TRACKS = {
    BUYING: TrackConfig(keyword=1.0, category=0.2, vector=0.1),
    BROWSING: TrackConfig(keyword=1.0, category=0.6, vector=0.1),
}


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    score: float
    routes: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DualResult:
    """Ranked items plus enough detail to explain and debug one turn."""

    items: list[Candidate]
    intent: str
    pool_size: int                        # candidates the routes produced
    filtered_size: int                    # survivors of the hard layer
    route_sizes: dict[str, int] = field(default_factory=dict)
    filters_applied: list[str] = field(default_factory=list)
    filters_skipped: list[str] = field(default_factory=list)

    @property
    def parent_asins(self) -> list[str]:
        return [item.parent_asin for item in self.items]
