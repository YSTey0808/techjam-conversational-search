# reranker

Stage B (deterministic candidate shrinking) and Stage C (LLM rerank) for the product
recommendation pipeline. The engine lives in [reranker.py](reranker.py); all tunables are in
the `CONFIG` dict at the top of that file.

- `shrink_pool(request)` — pure stdlib, no network, never mutates its input. Shrinks a
  candidate pool to at most 100 products.
- `llm_rerank(request, shrunk, client=None, top_k=None)` — asks Claude for the final ordering.
  `import anthropic` happens inside the function, so Stage B works with the package absent.
- [adapter.py](adapter.py) `rerank(prep, state, pool, top_k)` — the pipeline bridge. Keeps
  `reranker.py` free of any `starter` import.

## Pipeline wiring

`starter/agent.py` calls `adapter.rerank(...)` where it used to call `rank.rank(...)`; the
signature and return type are identical (`list[str]`, at most `top_k`), so nothing else in the
agent changed. `starter/rank.py` has been deleted — its IDF + popularity + budget scoring is
**not** carried over, so ordering now comes from RRF fusion (and the LLM step when enabled).

The adapter maps pipeline objects onto the request schema:

| Request field | Source |
|---|---|
| `query` | `state.category` + every constraint's `text`, joined (SessionState keeps no raw utterance) |
| `constraints` | resolved constraints only (`c.key` non-empty), attribute suffixed with its index so two `feature` constraints don't collide |
| `user_profile` | `state.profile` — the evaluator's profile already matches this schema field for field |
| `pool[].title` / `.text` | `prep.title` / `prep.text` |
| `pool[].retrieval_rank` | 1-based position in the pool `retrieve()` returned |
| `pool[].constraint_status` | `matched` when the ASIN is in `prep.lookup(c.key, broad=True)`, else `unknown` |
| `pool[].store_rating` / `.n_store_ratings` | `prep.average_rating` / `prep.rating_number` |

Unresolved constraints are skipped for the same reason `retrieve._intersect` skips them: a
string we can't find in the catalog is an indexing failure, not evidence against a product.
Including them would mark every candidate `unknown` and silently disable the safety rule.

`prep.title`, `prep.text` and `prep.average_rating` were added to `starter/preprocessing.py`
for this — three read-only dicts filled during the existing catalog pass. No existing
behaviour changed; without them the rating judge would be dead (every item on the 4.0 prior)
and the LLM would see no product titles.

## The LLM step is currently MOCKED

`CONFIG["llm_mock"] = True`. `llm_rerank` returns the Stage B ordering unchanged and never
touches the network — no API call, no tokens, no cost, no added latency. **What the pipeline
produces today is Stage B's ranking, full stop.** The mock is still routed through
(`adapter._use_llm()` returns True under it), so the whole path runs end to end:

```
agent.respond -> adapter.rerank -> shrink_pool -> llm_rerank -> _mock_rerank -> identity
```

The real Claude call is intact directly below the mock gate in `llm_rerank`, marked and
unreachable rather than commented out, so it stays syntax-valid and diffable.

### Turning the real call on

1. `CONFIG["llm_mock"] = False`
2. `pip install anthropic`
3. `cp .env.example .env` and put your key in `ANTHROPIC_API_KEY`.

With the mock off, the adapter runs the LLM step **only when a key is visible**; with no key
every turn falls back to pure Stage B. `RERANKER_USE_LLM=0` forces it off even with a key.
`.env` is gitignored and read by `load_env()` (a dependency-free dotenv reader — a real
environment variable always wins).

Rough cost before you flip it: ~14K input tokens per call at 100 candidates × 400 chars. The
last recorded eval run took 636 turns, so order of 9M tokens ≈ **$20 per full run** on
Sonnet 5, plus seconds of latency per turn.

## Usage

Standalone use (the pipeline calls `adapter.rerank` instead — see above):

```python
from reranker import shrink_pool, llm_rerank

request = {
    "query": "warm waterproof jacket for hiking",
    "constraints": [{"attribute": "color", "value": "black"}],
    "user_profile": {
        "preference_tags": ["comfort", "waterproof"],
        "rating_style": "critical",
        "average_prior_rating": 3.4,
        "summary": "Buys technical outdoor gear, dislikes bulky fits.",
    },
    "pool": [
        {
            "parent_asin": "B00EXAMPLE1",
            "title": "Alpine Shell Jacket",
            "text": "Water resistant, cozy fleece lining, relaxed fit.",
            "retrieval_rank": 1,
            "constraint_status": {"color": "matched"},
            "store_rating": 4.6,
            "n_store_ratings": 812,
        },
        # ... hundreds more
    ],
}

shrunk = shrink_pool(request)          # <= 100 products, ordered best-first
final = llm_rerank(request, shrunk)    # top-K, LLM-ordered (falls back to shrunk order)

for p in final["products"]:
    print(p["final_rank"], p["parent_asin"])
```

`llm_rerank` needs `pip install anthropic` and a credential in the environment
(`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile — the zero-arg
client resolves all three). Without either, it returns the Stage B order with
`llm_used: False` and an `error` string rather than raising.

## `shrink_pool` — input

```python
{
  "query": str,
  "constraints": [{"attribute": str, "value": str}],
  "user_profile": {                # may be None -> treated as no profile
      "preference_tags": [str],
      "rating_style": str,         # e.g. "critical", "usually positive"
      "average_prior_rating": float,
      "summary": str,
  },
  "pool": [{
      "parent_asin": str,
      "title": str,
      "text": str,                 # concatenated description / bullets / attributes
      "retrieval_rank": int,       # 1-based rank from upstream retrieval
      "constraint_status": {"<attribute>": "matched" | "unknown"},
      "store_rating": float | None,
      "n_store_ratings": int | None,
  }],
}
```

## `shrink_pool` — output

```python
{
  "products": [{                   # ordered best-first, len <= CONFIG["output_size"]
      "parent_asin": str,
      "final_rank": int,           # 1-based, contiguous
      "rrf_score": float,          # 0.0 on the skip branch
      "guaranteed": bool,          # survived via the safety rule
      "judge_ranks": {"retrieval": int, "rating": int | None, "preference": int | None},
  }],
  "branch_used": "skip" | "light" | "heavy",
  "pool_size_in": int,
}
```

Only `parent_asin` and the ordering are consumed downstream; the rest is debug info.

## How the ordering is produced

| Pool size | Branch | Score |
|---|---|---|
| `<= 150` | `skip` | none — sorted by `retrieval_rank`, cut to 100 |
| `151–500` | `light` | `1/(60 + retrieval_rank) + w_r * 1/(60 + rating_rank)` |
| `> 500` | `heavy` | light, plus `0.4 * 1/(60 + preference_rank)` |

`w_r` is `0.5` when `rating_style` contains "critical" (case-insensitive), else `0.3`.

**Rating rank** — items sorted descending by the Bayesian-smoothed rating
`(15 * 4.0 + store_rating * n_store_ratings) / (15 + n_store_ratings)`. Ties keep retrieval
order (stable sort).

**Preference rank** (heavy only) — `matched_tags / len(preference_tags)`, descending. A tag
matches when the tag itself or any of its `SYNONYMS` entries appears (case-insensitively) in
the item's `title` or `text`. Extend `SYNONYMS` by adding keys — nothing else needs to change.

**Safety rule** — an item whose `constraint_status` is `"matched"` for *every* requested
constraint is guaranteed a slot. Guaranteed items come first (ordered by score), then the
rest by score, cut to 100. If guaranteed items alone exceed 100 they are ranked among
themselves and cut.

### Edge cases

| Situation | Behaviour |
|---|---|
| `user_profile` is `None` or missing | `w_r` falls back to `0.3`; the preference term is omitted, so a heavy pool scores like a light one |
| `preference_tags` empty or missing | Same — preference term omitted, `judge_ranks["preference"]` is `None` |
| `store_rating` is `None` | Smoothed rating is exactly the `4.0` prior |
| `n_store_ratings` is `None` or `0` | Treated as 0 ratings, which also yields the prior |
| `constraints` is empty | Every item is vacuously guaranteed; ordering reduces to pure score |
| A constraint attribute missing from `constraint_status` | Counts as unknown, not matched |
| Pool of 150 on the skip branch | Sorted by `retrieval_rank` and **cut to 100** — the `<= 100` contract holds on every branch |
| Empty pool | `{"products": [], "branch_used": "skip", "pool_size_in": 0}` |

## `llm_rerank` — output

```python
{
  "products": [{"parent_asin": str, "final_rank": int}],   # len <= CONFIG["llm_top_k"]
  "model": str,
  "llm_used": bool,      # False when the Stage B fallback was used
  "error": str | None,   # why the fallback fired
}
```

The model's reply is pinned to `{"ranking": [asin, ...]}` via structured outputs, then
sanitised: ASINs not in the shrunk set are dropped, duplicates removed, and anything the
model omitted is appended in Stage B order before the top-K cut. A hallucinated or truncated
ranking degrades to Stage B order — it can never lose or invent products.

Every failure path (package missing, auth, rate limit, API error, connection error, refusal,
unparseable response) returns the Stage B ordering with `llm_used: False`, so the pipeline
degrades instead of crashing.

Note on cost: per-product `text` is truncated to `CONFIG["llm_text_chars"]` (400) before being
sent, since 100 full product descriptions is a lot of tokens for a ranking judgment. Set it to
`None` to send the full text. Prompt caching is deliberately not used — the candidate list
changes every request and the stable system prefix is below the minimum cacheable size, so a
breakpoint would buy nothing.

## `CONFIG` keys (for the eval sweep)

| Key | Default | Meaning |
|---|---|---|
| `skip_max_pool` | 150 | `<=` this uses the skip branch |
| `light_max_pool` | 500 | `<=` this (and above `skip_max_pool`) uses the light branch |
| `output_size` | 100 | Hard cap on `shrink_pool` output |
| `rrf_k` | 60 | RRF denominator constant |
| `rating_prior_count` | 15 | Smoothing pseudo-count |
| `rating_prior_value` | 4.0 | Smoothing prior mean |
| `w_preference` | 0.4 | Heavy-branch preference weight |
| `w_rating_critical` | 0.5 | `w_r` for critical raters |
| `w_rating_default` | 0.3 | `w_r` otherwise |
| `llm_model` | `claude-sonnet-5` | Model for the rerank call |
| `llm_max_tokens` | 16000 | Response cap |
| `llm_effort` | `high` | `low`\|`medium`\|`high`\|`xhigh`\|`max` — the first cost lever |
| `llm_thinking` | `True` | Adaptive thinking on/off |
| `llm_top_k` | 10 | Products returned by `llm_rerank` |
| `llm_text_chars` | 400 | Per-product text truncation; `None` sends full text |
| `llm_timeout_s` | 120.0 | Per-call timeout |
| `llm_max_retries` | 2 | SDK retry count (429s and 5xx) |
