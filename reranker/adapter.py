"""Bridge between the starter pipeline and reranker.py.

    rerank(prep, state, pool, top_k) -> list[str]      <- the only public name

Same signature and same contract as the rank.rank() it replaces, so agent.py
changes by one line. Everything pipeline-shaped lives here; reranker.py stays
self-contained and knows nothing about Preprocessing or SessionState.

CONFIG["llm_mock"] is currently True, so the LLM step is an identity function:
the turn's ordering is Stage B's, with no API call and no cost.

Once the mock is off, the LLM step runs only when a credential is visible
(ANTHROPIC_API_KEY, from the environment or .env). With no key the turn is pure
Stage B - no network, no per-turn latency. Set RERANKER_USE_LLM=0 to force it
off even with a key.
"""

from __future__ import annotations

import os

from reranker.reranker import CONFIG, llm_rerank, load_env, shrink_pool
from starter.preprocessing import Preprocessing
from starter.schema import SessionState


def _use_llm() -> bool:
    load_env()
    if os.environ.get("RERANKER_USE_LLM", "1").strip().lower() in {"0", "false", "no"}:
        return False
    # Mock mode needs no credential: llm_rerank short-circuits to the Stage B
    # order before touching the network, so the path runs at zero cost.
    if CONFIG["llm_mock"]:
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _constraint_rows(prep: Preprocessing, state: SessionState) -> list[tuple[dict, set[str]]]:
    """(request constraint, posting set) for every resolved constraint.

    Unresolved constraints (empty `key`) are skipped for the same reason
    retrieve._intersect skips them: a string we cannot find in the catalog is our
    indexing failure, not evidence against a product. Including them would leave
    every candidate "unknown" and silently disable the safety rule.

    Attributes are suffixed with their index because two constraints often share
    one attribute ("feature"), and constraint_status is keyed by attribute.

    Posting sets are looked up once per turn here, not once per candidate.
    """
    return [
        ({"attribute": f"{c.attribute}:{i}", "value": c.text}, prep.lookup(c.key, broad=True))
        for i, c in enumerate(state.constraints)
        if c.key
    ]


def _query(state: SessionState) -> str:
    """The customer's ask, reassembled - SessionState keeps no raw utterance."""
    parts = [state.category or ""]
    parts += [c.text for c in state.constraints if c.text]
    return "; ".join(part for part in parts if part)


def build_request(prep: Preprocessing, state: SessionState, pool: list[str]) -> dict:
    """Pipeline objects -> the reranker's request schema."""
    rows = _constraint_rows(prep, state)
    return {
        "query": _query(state),
        "constraints": [entry for entry, _postings in rows],
        "user_profile": state.profile or None,
        "pool": [
            {
                "parent_asin": asin,
                "title": prep.title.get(asin, ""),
                "text": prep.text.get(asin, ""),
                "retrieval_rank": position,
                "constraint_status": {
                    entry["attribute"]: ("matched" if asin in postings else "unknown")
                    for entry, postings in rows
                },
                "store_rating": prep.average_rating.get(asin),
                "n_store_ratings": prep.rating_number.get(asin),
            }
            for position, asin in enumerate(pool, start=1)
        ],
    }


def rerank(prep: Preprocessing, state: SessionState, pool: list[str], top_k: int) -> list[str]:
    """Order `pool` best-first and return at most `top_k` parent_asins."""
    if not pool or top_k <= 0:
        return []

    request = build_request(prep, state, pool)
    shrunk = shrink_pool(request)

    if not _use_llm():
        return [row["parent_asin"] for row in shrunk["products"][:top_k]]

    result = llm_rerank(request, shrunk, top_k=top_k)
    return [row["parent_asin"] for row in result["products"]]
