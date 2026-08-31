"""Bridge between the starter pipeline and reranker.py.

    rerank(prep, state, pool, top_k) -> list[str]      <- the only public name

Same signature and same contract as the rank.rank() it replaces, so agent.py
changes by one line. Everything pipeline-shaped lives here; reranker.py stays
self-contained and knows nothing about Preprocessing or SessionState.

The Stage C LLM step runs only when a provider is configured -- see
llm_client.get_client (ANTHROPIC_API_KEY or GROQ_API_KEY, from the environment
or .env). With none the turn is pure Stage B: no network, no per-turn latency.
CONFIG["llm_mock"]=True or RERANKER_USE_LLM=0 forces Stage B even with a key.
"""

from __future__ import annotations

import dataclasses
import os

from reranker.reranker import CONFIG, llm_rerank, load_env, shrink_pool
from starter.preprocessing import Preprocessing
from starter.schema import SessionState


def _use_llm() -> bool:
    load_env()
    if os.environ.get("RERANKER_USE_LLM", "1").strip().lower() in {"0", "false", "no"}:
        return False
    if CONFIG["llm_mock"]:
        return False
    provider = (os.environ.get("TECHJAM_LLM_PROVIDER") or "").strip().lower()
    if provider in {"anthropic", "groq"}:
        return True
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("GROQ_API_KEY")
    )


def _constraint_rows(prep: Preprocessing, state: SessionState) -> list[tuple[dict, set[str]]]:
    """(request constraint, posting set) for every resolved slot.

    Unresolved slots (empty `key`) are skipped for the same reason
    retrieve._intersect skips them: a string we cannot find in the catalog is our
    indexing failure, not evidence against a product. Including them would leave
    every candidate "unknown" and silently disable the safety rule.

    Attributes keep their positional suffix. It is redundant now that slots are
    keyed by attribute, but constraint_status must stay unique per entry for
    reranker._is_guaranteed, and the suffix is what guarantees that whatever the
    slot map does next.

    Posting sets are looked up once per turn here, not once per candidate.
    """
    return [
        ({"attribute": f"{attribute}:{i}", "value": str(slot.val)},
         prep.lookup(slot.key, broad=True))
        for i, (attribute, slot) in enumerate(state.filled_slots.items())
        if slot.key
    ]


def _query(state: SessionState) -> str:
    """The customer's ask, reassembled - SessionState keeps no raw utterance."""
    parts = [state.category or ""]
    parts += [str(slot.val) for slot in state.filled_slots.values() if slot.val]
    return "; ".join(part for part in parts if part)


def build_request(prep: Preprocessing, state: SessionState, pool: list[str]) -> dict:
    """Pipeline objects -> the reranker's request schema."""
    rows = _constraint_rows(prep, state)
    return {
        "query": _query(state),
        "constraints": [entry for entry, _postings in rows],
        # reranker.py reads this with .get() (reranker.py:98, :106), but
        # user_profile is a UserProfile dataclass since the schema migration.
        # asdict() is the boundary conversion; a bare dataclass would silently
        # disable the preference and rating-style judges.
        "user_profile": dataclasses.asdict(state.user_profile) if state.user_profile else None,
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
