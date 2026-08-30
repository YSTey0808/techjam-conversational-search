"""Stage B candidate shrinking + Stage C LLM rerank.

Self-contained: no imports from the rest of the repo, no dataset paths.
`shrink_pool` is pure stdlib and never touches the network.
`llm_rerank` lazy-imports `anthropic` so Stage B works without it installed.
"""

import json
import os
from pathlib import Path

CONFIG = {
    # --- Stage B ---
    "skip_max_pool": 150,        # <= this -> skip branch
    "light_max_pool": 500,       # <= this (and > skip_max_pool) -> light branch
    "output_size": 100,          # hard cap on shrink_pool output
    "rrf_k": 60,                 # RRF denominator constant
    "rating_prior_count": 15,    # Bayesian smoothing pseudo-count
    "rating_prior_value": 4.0,   # Bayesian smoothing prior mean
    "w_preference": 0.4,         # heavy-branch preference weight
    "w_rating_critical": 0.5,    # w_r when rating_style contains "critical"
    "w_rating_default": 0.3,     # w_r otherwise
    # --- Stage C (LLM) ---
    # >>> MOCK MODE IS ON <<<  No API call, no tokens, no cost. The mock returns
    # the Stage B order unchanged, so what you measure is Stage B alone. Set this
    # to False (and provide a key) to make the real Claude call live again.
    "llm_mock": True,
    "llm_model": "claude-sonnet-5",
    "llm_max_tokens": 16000,
    "llm_effort": "high",        # low | medium | high | xhigh | max
    "llm_thinking": True,        # adaptive thinking on/off
    "llm_top_k": 10,             # how many products the LLM returns
    "llm_text_chars": 400,       # per-product text truncation; None = send full text
    "llm_timeout_s": 120.0,
    "llm_max_retries": 2,
}

# preference tag -> extra surface forms that also count as a match. Extend freely.
SYNONYMS = {
    "comfort": ["comfortable", "cozy", "soft"],
    "fit": ["fitted", "slim fit", "relaxed fit"],
    "durable": ["durability", "long lasting", "heavy duty"],
    "durability": ["durable", "long lasting", "heavy duty"],
    "waterproof": ["water resistant", "water-resistant", "weatherproof"],
    "lightweight": ["light weight", "featherweight"],
    "budget": ["affordable", "value for money", "inexpensive"],
    "warm": ["insulated", "thermal", "fleece"],
    "breathable": ["breathability", "moisture wicking", "ventilated"],
}


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------

def _smoothed_rating(item):
    """Bayesian-smoothed store rating. Missing rating gets exactly the prior."""
    rating = item.get("store_rating")
    if rating is None:
        return CONFIG["rating_prior_value"]
    n = item.get("n_store_ratings") or 0
    prior_n, prior_v = CONFIG["rating_prior_count"], CONFIG["rating_prior_value"]
    return (prior_n * prior_v + rating * n) / (prior_n + n)


def _retrieval_order(pool):
    """Pool indices sorted by upstream retrieval rank."""
    return sorted(range(len(pool)), key=lambda i: pool[i].get("retrieval_rank", 0))


def _ranks_from_order(order, size):
    """Turn an ordered list of pool indices into 1-based ranks parallel to the pool."""
    ranks = [0] * size
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _rating_ranks(pool):
    """1-based rank by smoothed rating, descending. Ties keep retrieval order."""
    order = sorted(_retrieval_order(pool), key=lambda i: -_smoothed_rating(pool[i]))
    return _ranks_from_order(order, len(pool))


def _pref_score(item, tags):
    """Fraction of preference tags whose text (or a synonym) shows up in the item."""
    haystack = f"{item.get('title') or ''} {item.get('text') or ''}".lower()
    matched = 0
    for tag in tags:
        forms = [tag] + SYNONYMS.get(tag.lower(), [])
        if any(form.lower() in haystack for form in forms):
            matched += 1
    return matched / len(tags)


def _preference_ranks(pool, profile):
    """1-based rank by preference match, descending. None when there is no profile."""
    tags = (profile or {}).get("preference_tags") or []
    if not tags:
        return None
    order = sorted(_retrieval_order(pool), key=lambda i: -_pref_score(pool[i], tags))
    return _ranks_from_order(order, len(pool))


def _rating_weight(profile):
    style = ((profile or {}).get("rating_style") or "").lower()
    return CONFIG["w_rating_critical"] if "critical" in style else CONFIG["w_rating_default"]


def _is_guaranteed(item, constraints):
    """True when every requested constraint is explicitly 'matched' for this item."""
    status = item.get("constraint_status") or {}
    return all(status.get(c.get("attribute")) == "matched" for c in constraints)


def _row(item, score, guaranteed, rating_rank, pref_rank):
    return {
        "parent_asin": item.get("parent_asin"),
        "final_rank": 0,  # assigned after the final ordering is known
        "rrf_score": score,
        "guaranteed": guaranteed,
        "judge_ranks": {
            "retrieval": item.get("retrieval_rank"),
            "rating": rating_rank,
            "preference": pref_rank,
        },
    }


def shrink_pool(request):
    """Shrink a candidate pool to at most CONFIG['output_size'] products.

    Never mutates `request` or anything inside it.
    """
    pool = request.get("pool") or []
    constraints = request.get("constraints") or []
    profile = request.get("user_profile")
    pool_size_in = len(pool)

    if pool_size_in <= CONFIG["skip_max_pool"]:
        branch = "skip"
    elif pool_size_in <= CONFIG["light_max_pool"]:
        branch = "light"
    else:
        branch = "heavy"

    if branch == "skip":
        rows = [
            _row(pool[i], 0.0, _is_guaranteed(pool[i], constraints), None, None)
            for i in _retrieval_order(pool)[: CONFIG["output_size"]]
        ]
    else:
        k = CONFIG["rrf_k"]
        w_r = _rating_weight(profile)
        rating_ranks = _rating_ranks(pool)
        pref_ranks = _preference_ranks(pool, profile) if branch == "heavy" else None

        rows = []
        for i, item in enumerate(pool):
            score = 1.0 / (k + item.get("retrieval_rank", 0)) + w_r / (k + rating_ranks[i])
            pref_rank = pref_ranks[i] if pref_ranks else None
            if pref_rank is not None:
                score += CONFIG["w_preference"] / (k + pref_rank)
            rows.append(
                _row(item, score, _is_guaranteed(item, constraints), rating_ranks[i], pref_rank)
            )

        def by_score(row):
            return (-row["rrf_score"], row["judge_ranks"]["retrieval"])

        guaranteed = sorted((r for r in rows if r["guaranteed"]), key=by_score)
        rest = sorted((r for r in rows if not r["guaranteed"]), key=by_score)
        rows = (guaranteed + rest)[: CONFIG["output_size"]]

    for rank, row in enumerate(rows, start=1):
        row["final_rank"] = rank

    return {"products": rows, "branch_used": branch, "pool_size_in": pool_size_in}


# --------------------------------------------------------------------------
# Stage C - LLM rerank
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You rerank shopping search results. Given a query, any hard constraints, the "
    "shopper's profile, and a list of candidate products, order the candidates from "
    "best to worst match. Prefer products that satisfy every stated constraint and fit "
    "the shopper's stated preferences; use the rating only to break near-ties. Return "
    "parent_asin values drawn only from the candidate list, most relevant first."
)

RANKING_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}


def load_env(path=None):
    """Read KEY=VALUE lines from a .env file into os.environ. Existing vars win.

    Dependency-free stand-in for python-dotenv. Searches the reranker folder and
    its parents for `.env` when no path is given. Returns the file used, or None.
    """
    if path is None:
        here = Path(__file__).resolve()
        path = next((p / ".env" for p in here.parents if (p / ".env").is_file()), None)
        if path is None:
            return None
    path = Path(path)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return path


def _build_payload(request, shrunk):
    """Compact per-candidate records, in Stage B order, for the model to rank."""
    by_asin = {item.get("parent_asin"): item for item in (request.get("pool") or [])}
    limit = CONFIG["llm_text_chars"]
    payload = []
    for row in shrunk["products"]:
        item = by_asin.get(row["parent_asin"], {})
        text = item.get("text") or ""
        payload.append(
            {
                "parent_asin": row["parent_asin"],
                "title": item.get("title") or "",
                "text": text[:limit] if limit else text,
                "store_rating": item.get("store_rating"),
                "n_store_ratings": item.get("n_store_ratings"),
                "constraint_status": item.get("constraint_status") or {},
            }
        )
    return payload


def _fallback(shrunk, error, top_k=None):
    """Stage B ordering, used whenever the LLM call cannot be trusted or completed."""
    top_k = CONFIG["llm_top_k"] if top_k is None else top_k
    products = [
        {"parent_asin": row["parent_asin"], "final_rank": rank}
        for rank, row in enumerate(shrunk["products"][:top_k], start=1)
    ]
    return {"products": products, "model": CONFIG["llm_model"], "llm_used": False, "error": error}


def _apply_ranking(ranking, shrunk, top_k=None):
    """Keep only real ASINs, drop duplicates, then backfill anything the model dropped."""
    top_k = CONFIG["llm_top_k"] if top_k is None else top_k
    allowed = [row["parent_asin"] for row in shrunk["products"]]
    allowed_set = set(allowed)
    ordered, seen = [], set()
    for asin in ranking:
        if asin in allowed_set and asin not in seen:
            ordered.append(asin)
            seen.add(asin)
    ordered += [asin for asin in allowed if asin not in seen]
    return [
        {"parent_asin": asin, "final_rank": rank}
        for rank, asin in enumerate(ordered[:top_k], start=1)
    ]


def llm_rerank(request, shrunk, client=None, top_k=None):
    """Ask Claude to order the shrunk candidates. Falls back to Stage B order on any failure.

    `top_k` overrides CONFIG["llm_top_k"] for this call, so a caller whose list
    width varies per turn does not have to mutate the module-level CONFIG.
    """
    top_k = CONFIG["llm_top_k"] if top_k is None else top_k
    try:
        import anthropic
    except ImportError:
        return _fallback(shrunk, "anthropic package not installed", top_k)

    if not shrunk["products"]:
        return _fallback(shrunk, None, top_k)

    user_content = json.dumps(
        {
            "query": request.get("query"),
            "constraints": request.get("constraints") or [],
            "user_profile": request.get("user_profile"),
            "candidates": _build_payload(request, shrunk),
            "return_top_k": top_k,
        },
        ensure_ascii=False,
    )

    output_config = {
        "effort": CONFIG["llm_effort"],
        "format": {"type": "json_schema", "schema": RANKING_SCHEMA},
    }
    kwargs = {
        "model": CONFIG["llm_model"],
        "max_tokens": CONFIG["llm_max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "output_config": output_config,
    }
    if CONFIG["llm_thinking"]:
        kwargs["thinking"] = {"type": "adaptive"}

    # .env is read once here so callers do not have to; a real environment
    # variable always wins. Zero-arg client then resolves ANTHROPIC_API_KEY /
    # ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
    load_env()
    client = client or anthropic.Anthropic()
    try:
        response = client.with_options(
            timeout=CONFIG["llm_timeout_s"], max_retries=CONFIG["llm_max_retries"]
        ).messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        return _fallback(shrunk, f"bad request: {e.message}", top_k)
    except anthropic.AuthenticationError:
        return _fallback(shrunk, "authentication failed - check ANTHROPIC_API_KEY", top_k)
    except anthropic.RateLimitError:
        return _fallback(shrunk, "rate limited", top_k)
    except anthropic.APIStatusError as e:
        return _fallback(shrunk, f"api error {e.status_code}", top_k)
    except anthropic.APIConnectionError:
        return _fallback(shrunk, "connection error", top_k)

    if response.stop_reason == "refusal":
        return _fallback(shrunk, "model refused", top_k)

    text = next((b.text for b in response.content if b.type == "text"), None)
    try:
        ranking = json.loads(text)["ranking"]
    except (TypeError, ValueError, KeyError):
        return _fallback(shrunk, "unparseable response", top_k)

    return {
        "products": _apply_ranking(ranking, shrunk, top_k),
        "model": response.model,
        "llm_used": True,
        "error": None,
    }
