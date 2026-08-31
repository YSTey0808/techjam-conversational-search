"""OWNER C -- turn the SessionState into a candidate pool.

    retrieve(prep, state) -> list[str]                 <- the only public name

Three routes, each returning {parent_asin: score}, plus an orchestrator that
fuses them. Routes return scores rather than lists so the orchestrator can
combine them without knowing how any of them works -- add a fourth route and
nothing else has to change.

    _keyword    exact / near-exact catalog string matching   (PRIMARY)
    _category   the soft category gate
    _vector     semantic fallback                            (STUB in v1)

NEVER returns an empty list. An empty list is a guaranteed miss; a mediocre
list costs nothing but a turn.
"""

from __future__ import annotations

import math
import os

from starter.preprocessing import Preprocessing
from starter.schema import SessionState, Slot

# ---------------------------------------------------------------------------
# Candidate generation is delegated to the multi_retrieval package.
#
# Measured inside this pipeline, on the 200 public sessions, with everything
# else (extract, state, rank, ask) unchanged:
#
#     _original_retrieve + rank.py       0.5560
#     multi_retrieval    + rank.py       0.6995
#
# Ranking is deliberately NOT taken over: agent.py calls the reranker
# (reranker/adapter.py) after this function, and letting a separate stage score
# the pool was worth another 0.03 over multi_retrieval ordering the results
# itself. This function's job is to decide WHICH candidates it sees, nothing
# more. The numbers above predate the reranker -- rank.py was the scorer then.
#
# Flip this to False to restore the original implementation, which is preserved
# below as _original_retrieve. Nothing else needs changing.
# ---------------------------------------------------------------------------
USE_MULTI_RETRIEVAL = True

# multi_retrieval builds its own index and needs the catalog file, which
# Preprocessing does not record. Override with TECHJAM_CATALOG if you run the
# evaluator against a different one.
#
# The routes and hard filter build from the NORMALISED faceted table; the
# original export is still handed over as the numeric sidecar so the popularity
# prior and rating gates have average_rating / rating_number to read.
_CATALOG_PATH = os.environ.get("TECHJAM_CATALOG", "data/catalog.jsonl")
_NORMALISED_PATH = os.environ.get(
    "TECHJAM_CATALOG_NORMALISED", "data/catalog_normalised.jsonl"
)
# Precomputed nomic catalog vectors. If present (and sentence-transformers is
# installed) the vector route turns on; otherwise the two lexical routes carry
# the query and nothing here changes.
_EMBEDDINGS_DIR = os.environ.get("TECHJAM_EMBEDDINGS", "data/embeddings")

# How many candidates to hand the reranker. Matches the original _FUSE_LIMIT so
# the downstream stages see a pool of the size they were tuned against.
_POOL_SIZE = 50000

# Their Constraint.attribute vocabulary -> multi_retrieval slot names. The two
# line up because both are typed by attribute; "use_case" has no direct slot so
# it lands on the nearest one, and anything unmapped goes to free_text rather
# than being dropped.
_ATTRIBUTE_TO_SLOT = {
    "material": "material", "color": "color", "size": "size",
    "style": "style", "use_case": "occasion", "brand": "brand",
}

_RETRIEVER = None

# Reciprocal-rank fusion constant. 60 is the usual default; it flattens the
# difference between rank 1 and rank 2 so no single route can dominate.
_RRF_K = 60

# Route weights by intent. MODULE CONSTANTS: sweep these, do not guess them.
# Buying leans on explicit constraints; browsing has none yet, so it leans on
# the category gate until the customer says something concrete.
_WEIGHTS = {
    "buying":   {"keyword": 1.0, "category": 0.3, "vector": 0.5},
    "browsing": {"keyword": 0.8, "category": 1.0, "vector": 0.5},
}
_DEFAULT_WEIGHTS = _WEIGHTS["browsing"]

_ROUTE_LIMIT = 500      # per-route cap before fusion
_FUSE_LIMIT = 200       # candidates handed to the reranker
_FALLBACK_LIMIT = 200


# --------------------------------------------------------------------------
# _keyword -- the primary route
# --------------------------------------------------------------------------

def _intersect(
    prep: Preprocessing,
    slots: list[Slot],
    seed: set[str] | None,
    broad: bool,
) -> set[str] | None:
    """Intersect posting lists into `seed`. None means it collapsed to empty.

    An unmatched slot is SKIPPED, never intersected: a string we cannot find is
    our indexing failure, not evidence the product does not exist.
    Intersecting on an empty posting list would destroy the pool.
    """
    pool = None if seed is None else set(seed)
    for slot in slots:
        postings = prep.lookup(slot.key, broad=broad)
        if not postings:
            continue
        pool = set(postings) if pool is None else (pool & postings)
        if not pool:
            return None
    return pool


def _score_matches(prep: Preprocessing, state: SessionState, pool: set[str]) -> dict[str, float]:
    """Summed IDF of every slot each candidate actually matches.

    `confidence_score` carries the weight the old `Constraint.weight` did: it is
    already 0..1 and already says how firmly the customer wants this, so a
    separate hard/soft weight would just be a coarser copy of it.
    """
    scores = {asin: 0.0 for asin in pool}
    for slot in state.filled_slots.values():
        if not slot.key:
            continue
        weight = (slot.confidence_score or 0.0) * prep.idf(slot.key)
        matched = prep.lookup(slot.key, broad=True)
        for asin in pool:
            if asin in matched:
                scores[asin] += weight
    return scores


def _keyword(prep: Preprocessing, state: SessionState) -> dict[str, float]:
    """Intersect hard constraints, most-informative-first, with backoff.

    IDF-descending so the pool shrinks fastest and the constraint dropped on
    backoff is always the one carrying least signal.

    Backoff order is deliberate:
      1. drop the highest-DF constraint, repeatedly
      2. widen to the broad posting index
      3. drop the category gate LAST -- it is the cheapest evidence to lose
         only because it is also the least likely to be wrong
    """
    hard = [s for s in state.hard_slots.values() if s.key]
    if not hard:
        return {}
    ordered = sorted(hard, key=lambda s: prep.idf(s.key), reverse=True)
    category_pool = prep.category_pool(state.category)

    for use_category in (True, False):
        seed = category_pool if (use_category and category_pool) else None
        if use_category and not category_pool:
            continue
        for broad in (False, True):
            working = list(ordered)
            while working:
                pool = _intersect(prep, working, seed, broad)
                if pool:
                    scored = _score_matches(prep, state, pool)
                    return dict(sorted(scored.items(), key=lambda kv: -kv[1])[:_ROUTE_LIMIT])
                if len(working) <= 1:
                    break
                working.pop()          # drop the lowest-IDF / highest-DF one
    return {}


# --------------------------------------------------------------------------
# _category -- the soft gate
# --------------------------------------------------------------------------

def _category(prep: Preprocessing, state: SessionState) -> dict[str, float]:
    """Products in the customer's category bucket, weighted by specificity.

    SOFT, never a hard filter. coarse_category is lossy -- 1,136 products land
    on 'Shoes & Jewelry Westlake' because a store name sits in the category
    path -- so this route only ever contributes weight to the fusion. It never
    removes a candidate another route found.

    A 2-product bucket is far more informative than a 1,354-product one, so
    specificity is log(N / bucket_size). Popularity is folded in at a small
    weight purely to give the bucket a stable internal order for fusion.
    """
    pool = prep.category_pool(state.category)
    if not pool:
        return {}                      # browsing turn 1: nothing stated yet
    specificity = math.log((prep.n_docs + 1.0) / (len(pool) + 1.0))
    scored = {
        asin: specificity + 0.1 * prep.popularity.get(asin, 0.0)
        for asin in pool
    }
    return dict(sorted(scored.items(), key=lambda kv: -kv[1])[:_ROUTE_LIMIT])


# --------------------------------------------------------------------------
# _vector -- semantic fallback
# --------------------------------------------------------------------------

def _vector(prep: Preprocessing, state: SessionState) -> dict[str, float]:
    """Embedding similarity between constraint text and product text.

    V1 IS A DELIBERATE STUB and returns {}. The interface is defined so the
    orchestrator already fuses three routes; filling it in changes nothing
    else.

    This route exists for free-form phrasings that _keyword cannot resolve --
    a customer word that maps to no catalog string at all. If extract()'s
    variant resolution works well, this route may never fire, which is exactly
    why it is not built yet: it must earn its place against a measurement, not
    an intuition. No embedding model, no new dependency, until then.
    """
    return {}


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def _fallback(prep: Preprocessing, state: SessionState) -> list[str]:
    """Last resort. Category bucket by popularity, else the whole catalog."""
    pool = prep.category_pool(state.category)
    source = pool if pool else prep.asins
    return sorted(source, key=lambda a: -prep.popularity.get(a, 0.0))[:_FALLBACK_LIMIT]


def _multi_retriever(prep: Preprocessing):
    """Build the multi_retrieval index once and reuse it.

    Indexing 50,000 products takes a little over a second, so this must not
    happen per turn. The catalog is checked against the one Preprocessing
    loaded: indexing a different file would produce candidates that the
    reranker and ask.py cannot resolve, and it would fail silently rather
    than loudly.
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        from multi_retrieval import DualTrackRetriever

        primary = _NORMALISED_PATH if os.path.exists(_NORMALISED_PATH) else _CATALOG_PATH
        embeddings = _EMBEDDINGS_DIR if os.path.isdir(_EMBEDDINGS_DIR) else None
        retriever = DualTrackRetriever(
            primary, raw_catalog_path=_CATALOG_PATH, embeddings_dir=embeddings,
        )
        if retriever.index.size != prep.n_docs:
            raise RuntimeError(
                f"catalog mismatch: multi_retrieval indexed {retriever.index.size} "
                f"products from {primary!r}, but preprocessing loaded "
                f"{prep.n_docs}. Set TECHJAM_CATALOG / TECHJAM_CATALOG_NORMALISED "
                f"to the catalog actually in use."
            )
        _RETRIEVER = retriever
    return _RETRIEVER


def _slots_from(state: SessionState):
    """Turn the accumulated SessionState into multi_retrieval's Slots.

    `category_trusted` is left False on purpose. state.py *infers* the category
    from the words the customer used, and filtering on an inferred category was
    measured to cost 0.11 -- the target survived only 8% of turns while a route
    had already found it 99.2% of the time. Only a category quoted word-for-word
    is safe to filter on.
    """
    from multi_retrieval import Slots

    slots = Slots(category=state.category or "")
    for attribute, slot in state.filled_slots.items():
        # `category` is unmapped and so lands in free_text, as it did before:
        # Slots.category feeds the category route, free_text feeds the keyword
        # route, and both want those words.
        name = _ATTRIBUTE_TO_SLOT.get(attribute)
        if name and not getattr(slots, name):
            setattr(slots, name, str(slot.val))
        elif isinstance(slot.val, str) and slot.val not in slots.free_text:
            # Only text reaches free_text. `budget` binds a float, and "31.5"
            # in the BM25 expression is noise, not a term.
            slots.free_text.append(slot.val)
    return slots


def retrieve(prep: Preprocessing, state: SessionState) -> list[str]:
    """Find the candidate products.  <- the only public name

    Same contract as before: returns roughly 200 parent_asins for the reranker
    to score. See USE_MULTI_RETRIEVAL at the top of this file.
    """
    if not USE_MULTI_RETRIEVAL:
        return _original_retrieve(prep, state)

    from multi_retrieval import DualQuery

    result = _multi_retriever(prep).retrieve(DualQuery(
        slots=_slots_from(state),
        intent=state.scenario,
        top_k=_POOL_SIZE,
    ))
    pool = result.parent_asins
    # An empty list is a guaranteed miss; the original never returned one and
    # neither does this. multi_retrieval falls back internally, so this is a
    # belt-and-braces guard rather than an expected path.
    return pool or _fallback(prep, state)


def _original_retrieve(prep: Preprocessing, state: SessionState) -> list[str]:
    """Fuse the three routes by reciprocal rank and return the top candidates.

    The original Owner C implementation, kept intact. Set
    USE_MULTI_RETRIEVAL = False at the top of this file to make it live again.
    """
    routes = {
        "keyword": _keyword(prep, state),
        "category": _category(prep, state),
        "vector": _vector(prep, state),
    }
    weights = _WEIGHTS.get(state.scenario, _DEFAULT_WEIGHTS)

    fused: dict[str, float] = {}
    for name, scored in routes.items():
        if not scored:
            continue
        weight = weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        for position, (asin, _score) in enumerate(ranked, start=1):
            fused[asin] = fused.get(asin, 0.0) + weight / (_RRF_K + position)

    if not fused:
        return _fallback(prep, state)
    return [asin for asin, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:_FUSE_LIMIT]]
