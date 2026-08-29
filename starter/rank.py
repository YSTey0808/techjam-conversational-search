"""OWNER D -- order the candidate pool.

    rank(prep, state, pool, top_k) -> list[str]        <- the only public name

    score = sum over matched constraints of weight * idf(key)
          + POP_W * popularity
          - contradiction penalty

SCORE THE FULL POOL, THEN CUT. Never rank an already-truncated list: the
candidate you want is often not in the first slice, and truncating before
scoring throws it away with no way to get it back.

POP_W and CONTRADICTION_W are MODULE CONSTANTS so they can be swept. Do not
hand-tune them against a handful of examples -- popularity in particular is a
prior about the catalog, and a prior that is wrong for the real customer
distribution will quietly cost more than it gains.
"""

from __future__ import annotations

import re

from starter.preprocessing import Preprocessing
from starter.schema import SessionState

# Weight on log1p(rating_number). Sweep, do not guess.
POP_W = 0.35
# Penalty per unit of relative overage against a stated budget cap.
CONTRADICTION_W = 2.0

_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _budget_cap(state: SessionState) -> float | None:
    """The tightest price ceiling the customer has stated, if any."""
    caps: list[float] = []
    for constraint in state.constraints:
        if constraint.attribute != "budget":
            continue
        match = _PRICE_RE.search(constraint.text)
        if match:
            caps.append(float(match.group(1)))
    return min(caps) if caps else None


def _matched_sets(prep: Preprocessing, state: SessionState) -> list[tuple[float, set[str]]]:
    """(weight * idf, posting set) for every constraint that resolved.

    Computed once per turn rather than per candidate -- otherwise scoring is
    O(pool * constraints) lookups instead of O(constraints).
    """
    out: list[tuple[float, set[str]]] = []
    for constraint in state.constraints:
        if not constraint.key:
            continue
        out.append((
            constraint.weight * prep.idf(constraint.key),
            prep.lookup(constraint.key, broad=True),
        ))
    return out


def _penalty(prep: Preprocessing, asin: str, cap: float | None) -> float:
    """Contradiction penalty.

    Only priced products can contradict a budget: a null price is unknown, not
    a violation, and 79% of the catalog has no price at all. Penalising the
    unknown would bury most of the catalog for anyone who mentions money.
    """
    if cap is None or cap <= 0:
        return 0.0
    price = prep.price.get(asin)
    if price is None or price <= cap:
        return 0.0
    return CONTRADICTION_W * min((price - cap) / cap, 2.0)


def rank(prep: Preprocessing, state: SessionState, pool: list[str], top_k: int) -> list[str]:
    """Order `pool` best-first and return at most `top_k` parent_asins."""
    if not pool or top_k <= 0:
        return []

    matched = _matched_sets(prep, state)
    cap = _budget_cap(state)

    scored: list[tuple[float, str]] = []
    for asin in pool:                                   # the FULL pool
        score = sum(weight for weight, postings in matched if asin in postings)
        score += POP_W * prep.popularity.get(asin, 0.0)
        score -= _penalty(prep, asin, cap)
        scored.append((score, asin))

    scored.sort(key=lambda pair: -pair[0])
    return [asin for _score, asin in scored[:top_k]]     # cut only at the end
