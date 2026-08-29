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

from starter.preprocessing import Preprocessing
from starter.schema import Constraint, SessionState

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
_FUSE_LIMIT = 200       # candidates handed to rank.py
_FALLBACK_LIMIT = 200


# --------------------------------------------------------------------------
# _keyword -- the primary route
# --------------------------------------------------------------------------

def _intersect(
    prep: Preprocessing,
    constraints: list[Constraint],
    seed: set[str] | None,
    broad: bool,
) -> set[str] | None:
    """Intersect posting lists into `seed`. None means it collapsed to empty.

    An unmatched constraint is SKIPPED, never intersected: a string we cannot
    find is our indexing failure, not evidence the product does not exist.
    Intersecting on an empty posting list would destroy the pool.
    """
    pool = None if seed is None else set(seed)
    for constraint in constraints:
        postings = prep.lookup(constraint.key, broad=broad)
        if not postings:
            continue
        pool = set(postings) if pool is None else (pool & postings)
        if not pool:
            return None
    return pool


def _score_matches(prep: Preprocessing, state: SessionState, pool: set[str]) -> dict[str, float]:
    """Summed IDF of every constraint each candidate actually matches."""
    scores = {asin: 0.0 for asin in pool}
    for constraint in state.constraints:
        if not constraint.key:
            continue
        weight = constraint.weight * prep.idf(constraint.key)
        matched = prep.lookup(constraint.key, broad=True)
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
    hard = [c for c in state.constraints if c.hard and c.key]
    if not hard:
        return {}
    ordered = sorted(hard, key=lambda c: prep.idf(c.key), reverse=True)
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


def retrieve(prep: Preprocessing, state: SessionState) -> list[str]:
    """Fuse the three routes by reciprocal rank and return the top candidates."""
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
