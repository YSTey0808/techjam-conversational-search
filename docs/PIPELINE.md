# The Pipeline, End to End

Follow one customer message from the raw data files all the way to ten ranked
products — naming the exact file, the exact function, and the exact shape of the
data at every hop.

If you want to know *why* a decision was made, read [ARCHITECTURE.md](ARCHITECTURE.md).
If you want the retrieval stage's measurement history, read [RETRIEVAL.md](RETRIEVAL.md).
This document is the *what happens, in order* one. It repeats nothing important by
reference — everything you need to follow the flow is here.

---

## 0. The whole thing on one screen

```
data/catalog.jsonl  ──┐
data/public_set.jsonl ┴──►  evaluator/local_evaluator.py : main()
                                       │   builds the fake customer,
                                       │   then drives up to 10 turns
                                       ▼
                        Agent.__init__()  ──►  preprocessing.build()      ONCE, at startup
                        Agent.reset()     ──►  a fresh SessionState       once per session
                        Agent.respond()   ──►  one turn                   up to 10x per session
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                             │
   extract.extract() ─► state.update() ─► retrieve.retrieve() ─► ask.decide() ─► rank.rank()
     Owner A             Owner B           Owner C                Owner E         Owner D
     English             remember          ~200 candidates        the question    the 10
     → Extraction        → SessionState    → list[asin]           → TurnPolicy    → list[asin]
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
              {"message", "ask_attribute", "recommendations", "usage"}
```

Everything below is a zoom-in on one of those boxes.

---

## 1. Where the data comes in

Two files. They do completely different jobs, and mixing them up is the most common
early confusion.

| File | Rows | Read by | What it is |
|---|---|---|---|
| `data/catalog.jsonl` | 50,000 | **the agent and the evaluator** | the products we search |
| `data/public_set.jsonl` | 200 | **the evaluator only** | test sessions + the hidden answer |

`catalog.jsonl` is not in git — it's a GitHub Release download. One product per line:

```jsonc
{
  "parent_asin": "B09PYB7B6Z",                  // the ID everything is keyed on
  "title": "QIAN0813 Celttic Knot Triple Moon Pentagram ... Pagan Jewelry",
  "features": ["Material:alloy", "Triple Moon Pentagram Symbol", ...],
  "description": ["..."],
  "details": {"Package Dimensions": "...", ...},
  "categories": ["Clothing, Shoes & Jewelry", "Boys", "Jewelry", "Necklaces"],
  "store": "QIAN0813",
  "price": 9.99,                                // null on 79% of rows
  "average_rating": 4.6,
  "rating_number": 490
}
```

`public_set.jsonl` is one test session per line:

```jsonc
{
  "sample_id": "public_0001",
  "scenario_type": "buying",                    // buying | browsing | intent_override | boundary
  "ground_truth": {"parent_asin": "B09PYB7B6Z"},   // THE ANSWER
  "user_profile": {"preference_tags": ["fit","comfort","durability"], ...},
  "category_bucket": "clothing",
  "difficulty_bucket": "easy"
}
```

**The agent never sees `ground_truth`.** It only ever receives English sentences that
the evaluator synthesises *from* the target product. Finding that product is the game.

---

## 2. Who calls the agent — the evaluator loop

`evaluator/local_evaluator.py` is both the harness and the simulated customer. Read this
section first: it explains *why* the agent is shaped the way it is.

```
main()                                      lines 298-308
  ├─ load_jsonl("data/public_set.jsonl")    the 200 sessions
  ├─ catalog_index("data/catalog.jsonl")    ids, categories, full products
  └─ evaluate(Agent(catalog), ...)          Agent.__init__ runs here — indexing happens once
```

Then, for each of the 200 sessions (`evaluate()`, lines 216-295):

**Step 1 — build the secret shopping list.** `materialize_hidden_fields()` → `intent_card()`
(lines 52-71) reads the *target product's own* `features` and `details` and turns them
into what the customer "wants":

```python
{"target_category":  "<the product title>",
 "hard_constraints": ["Material:alloy", "Triple Moon Pentagram Symbol"],   # first 2
 "soft_preferences": ["<feature 3>", "<feature 4>"]}                        # next 2
```

This is the key insight about this evaluator: **the customer quotes the product's own
catalog strings back at us.** That is precisely why `preprocessing.py` indexes raw
feature strings verbatim, and why `extract.py` tries to resolve customer words *into*
catalog vocabulary.

**Step 2 — reset.** `agent.reset(session_id, sample["user_profile"])` (line 228).

**Step 3 — the opening line.** `initial_message()` (lines 154-163) writes turn 1:

| Scenario | Sentence |
|---|---|
| `buying` | `"I'm looking for {category}. A key requirement is: {hard_constraints[0]}."` |
| `browsing` | `"I'm looking for {category}, but I'm still exploring."` |
| `intent_override` | `"I'm looking for {category}. {old_value}"` |

where `{category}` is `coarse_category()` of the target's category path — e.g.
`"Jewelry Necklaces"`.

**Step 4 — the turn loop** (lines 238-268), at most `MAX_TURNS = 10`:

```python
response = agent.respond(session_id, user_message, turn, TOP_K)   # TOP_K = 10
ranked   = normalize_recommendations(response["recommendations"], catalog_ids)
if target in ranked:            # HIT — record the rank, stop the session
    best_rank, hit_turn = ranked.index(target) + 1, turn
    break
user_message, _ = customer_reply(sample, response["ask_attribute"], disclosed, ...)
```

`normalize_recommendations()` (lines 95-109) drops anything that isn't a real catalog
ID, drops duplicates, and **cuts at 10**. Returning more than ten costs nothing and
gains nothing.

**Step 5 — the customer answers our question.** `customer_reply()` (lines 166-185):

- we asked nothing (`ask_attribute is None`) → *"Those options are not quite right yet.
  Ask me about one specific attribute."* — a wasted turn
- we asked something → the customer looks through the undisclosed items on their secret
  list, keeps those whose `classify_constraint()` matches our attribute, and hands over
  up to two of them: *"For that, what matters is: Material:alloy."*
- nothing on the list matches → *"I don't have an additional preference for {attribute}."*
- a `boundary` session declines once: *"I don't have a preference for {attribute};
  please use your judgment."*

**So a good question is literally how you obtain the next clue.** That is `ask.py`'s
entire reason to exist.

**Step 6 — `intent_override` sessions** (lines 258-264) swap the requirement at turn 3
or 4: *"Actually, ignore my earlier preference. What I need is: …"*, and hits before
that point don't count.

**Scoring** (lines 279-280):

```
efficiency = (11 - MTTC) / 10                          MTTC = mean turns to first hit
score      = 0.50*hit_rate@10 + 0.30*MRR + 0.20*efficiency
```

Hit rate dominates, but MRR means rank 1 is worth ten times rank 10, and efficiency
means finding it on turn 2 beats finding it on turn 6.

---

## 3. Startup — `preprocessing.build()`, run exactly once

```python
# starter/agent.py:29-34
def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
    self.prep = preprocessing.build(catalog_path)          # ~50k rows, one pass
    self._sessions: dict[str, SessionState] = defaultdict(SessionState)
```

`starter/preprocessing.py:build()` (lines 271-311) makes one pass over the catalog and
builds everything read-only. Nothing here is ever rebuilt per turn.

| What it builds | Line | Who uses it later |
|---|---|---|
| `asins` — every `parent_asin`, in file order | 285 | fallbacks |
| `popularity[asin]` = `log1p(rating_number)` | 290 | `rank.py`, `_category`, `_fallback` |
| `price[asin]` via `safe_price()` | 291 | `rank._penalty()` |
| `coarse[asin]` + `cat_index[category]` via `coarse_category()` | 293-295 | `state._infer_category()` |
| `product_keys[asin]` — every indexed string, in order | 299 | `ask._split_quality()` |
| `postings[key]` — **broad**: every string of every product | 301 | `extract._resolve()`, `rank` |
| `canon_postings[key]` — **narrow**: only each product's first 4 strings | 302-303 | precise matching |
| `sorted_keys` | 308 | prefix-tolerant `lookup()` |

On the real catalog that comes out as **50,000 products, 225,228 broad keys, 59,767
narrow keys, 1,115 category buckets.**

Where the keys come from: `constraint_candidates()` (lines 110-137) — every `features`
entry, every `details` entry, plus three synthesised strings (a bare material word,
`"color: black"`, `"budget around $9.99"`). These are *exactly* the strings the
evaluator's `intent_card()` builds its customer from, which is why verbatim matching
works at all.

Three lookups the rest of the codebase lives on:

**`lookup(key, broad)`** (lines 207-225) — exact hit first; on a miss it prefix-matches
in both directions via `bisect` over `sorted_keys`. A long feature string can arrive
truncated, so exact matching alone would miss. Guarded by `MIN_PREFIX_LEN = 12` (so
`"leather"` can't prefix-match half the catalog) and `PREFIX_FANOUT_CAP = 64`.

**`idf(key)`** (lines 233-244) — `log((n_docs + 1) / (df + 1))`, cached. Not optional:
`"Imported"` sits on ~13,900 products and `"Machine Wash"` on ~8,900. Counting matches
without weighting by rarity is close to meaningless.

**`attribute_of(key)`** (lines 251-255) — keyword classifier (`_classify`, lines 153-167)
labelling a string `budget|material|color|size|use_case|style|feature`. Used only by
`ask.py`.

One oddity worth knowing: `preprocessing.active()` (lines 258-268) is a module-level
handle on the most recent index. It exists because `extract()` needs the index but its
signature takes no `prep` argument.

**Cost:** `preprocessing.build()` plus `multi_retrieval`'s own index (§7) is a couple of
seconds, paid once. Per turn, zero indexing.

---

## 4. `Agent.respond()` — five calls, no logic

```python
# starter/agent.py:42-55
def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
    state  = self._sessions[session_id]
    ex     = extract.extract(user_message, turn, state)        # A
    state  = state_mod.update(state, ex)                       # B
    pool   = retrieve.retrieve(self.prep, state)               # C
    policy = ask.decide(self.prep, state, pool, turn)          # E
    items  = rank.rank(self.prep, state, pool, policy.list_width)   # D
    self._sessions[session_id] = state
    return {
        "message":         policy.message,
        "ask_attribute":   policy.ask_attribute,
        "recommendations": [{"parent_asin": a} for a in items],
        "usage":           ex.usage,
    }
```

That is the entire pipeline. **No branching, no fallbacks in this file** — anything that
needs a decision belongs to the module that owns it.

Two contract details that live right here:

- **`top_k` is accepted and never read.** The evaluator always passes `10`
  (`local_evaluator.py:16`, and `docs/agent_api_contract.json` pins it as
  `{"const": 10}`), but list width actually comes from `ask.py:_LIST_WIDTH = 10` via
  `policy.list_width`. Same number, two independent sources.
- **`message` must be a `str`.** If it isn't, `local_evaluator.py:243-244` throws away
  the entire turn — recommendations included.

Note the call order: **`ask.decide()` runs before `rank.rank()`**, and it looks at the
full ~200 candidate pool, not the final ten.

---

## 5. Stage A — `extract.extract(message, turn, state) → Extraction`

`starter/extract.py:425-439`. Turns one English sentence into structured requirements.

The hard problem: **a shopper and a product listing do not use the same words.** They
say *"waterproof"*; the catalog says *"Water Resistant"*. They say *"stretchy"*; the
catalog says *"Spandex"*.

### Path 1 — the LLM (only if you configure one)

```
extract()
  └─ _cached_call(message, turn, known)          LRU, 512 entries
       └─ _client()                              picks a provider from env vars
            ├─ _NullClient        always raises  ← THE DEFAULT
            ├─ _OllamaClient      POST /api/generate
            └─ _OpenAICompatClient POST /chat/completions
       └─ _validate(raw)                         strict schema check, else None
```

The prompt (`_PROMPT`, lines 133-158) asks for JSON with, per constraint, a `text`, an
`attribute`, a `hard` flag, and — the part that matters — **`variants`: 2-5 literal
phrasings a product page might use.** Timeout 2s, one retry. Anything malformed is
rejected outright rather than half-trusted. Cached on `(message, turn, what we already
know)` — the last part matters because it changes the prompt.

```bash
TECHJAM_LLM_PROVIDER=ollama          # or: openai
TECHJAM_LLM_MODEL=llama3.1
TECHJAM_LLM_BASE_URL=http://localhost:11434
TECHJAM_LLM_API_KEY=...              # openai-compatible only
TECHJAM_LLM_TIMEOUT=2.0
```

**With nothing configured you get `_NullClient`, which always fails.** So out of the
box — and in CI — the path below is what actually runs.

### Path 2 — `_degraded()`, the rule-based reader (lines 320-375)

A real extractor, not a token dump:

1. `_find_budget()` — `$40`, `"fifty bucks"`, `"under 30"` → a hard `budget` constraint
2. colour words → hard `color`, with variants `["color: black", "black"]`
3. material words → hard `material`, variants `["cotton", "100% cotton"]`
4. use-case words → hard `use_case`, expanded through `_SYNONYMS`
   (`waterproof → ["water resistant", "waterproof", "weatherproof"]`)
5. size cues → soft `size`
6. **everything left over** — any token longer than 3 chars that isn't a stopword —
   becomes a weak `feature` constraint, so a word we have no rule for still gets a shot

Plus `override` detection (`"actually"`, `"forget"`, `"never mind"`, …) and
`no_preference` detection (`"doesn't matter"`, `"up to you"`, …).

### The vocabulary bridge — `_resolve()` (lines 382-395)

```python
for candidate in [*variants, text]:
    key = normalize(candidate)
    if key and prep.lookup(key, broad=True):
        return key          # first phrasing that genuinely exists in the catalog
return ""                   # nothing matched — NORMAL, not an error
```

**`key == ""` is a normal outcome.** The constraint is still carried; it just can't be
looked up, and may still match later. Everything downstream searches on `key`.

### Output

```python
Extraction(
  constraints = [Constraint(text, key, attribute, hard, weight, turn), ...],
  intent      = "buying" | "browsing",
  override    = bool,          # they retracted something
  no_preference = str | None,  # an attribute they declined
  usage       = {"prompt_tokens": int, "completion_tokens": int},
)
```

`weight` is `1.0` when `hard`, else `_SOFT_WEIGHT = 0.35`.

**`extract()` never raises.** Nested try/except; the worst case is an empty `Extraction`.

---

## 6. Stage B — `state.update(state, extraction) → SessionState`

`starter/state.py:110-134`. Folds one turn into the running conversation. Three
operations, in the order a real conversation needs them:

**1. Wipe** — only when `extraction.override` is set. `_wipe()` (90-96) drops previously
held constraints on the same attributes the customer just spoke about. This is the
`intent_override` scenario's escape hatch.

**2. Replace** — `_replace_or_append()` (99-107). Same key or same text → refresh it.
Otherwise, if the attribute is in

```python
_SINGLE_VALUED = {"color", "size", "budget", "material"}
```

the new value **replaces** the old one: a person has one favourite colour at a time, so
"black" then "white" should swap, not accumulate.

**3. Append** — everything else piles up.

Two more things happen:

- `no_preference` → `state.dead_attributes`, so `ask.py` never asks about it again
- `_infer_category()` (48-80) — a token vote. Take every word from this turn's
  constraints, look up which of the 1,115 `cat_index` buckets contain that word, count
  the votes, take the winner. Deliberately conservative: ties resolve to the *smallest*
  bucket, and if the smallest is still over 2,000 products it stays uncommitted. Set
  **once** (`if state.category is None`) and never revised.

The resulting `SessionState`:

```python
SessionState(session_id, profile, turn, scenario, category,
             constraints=[Constraint, ...], asked=[...], dead_attributes={...})
```

**`user_profile` is soft signal only.** It never becomes a constraint and never filters
anything — a profile-derived hard filter would start excluding correct answers for no
gain.

---

## 7. Stage C — `retrieve.retrieve(prep, state) → list[str]`

The biggest stage. It has two halves: a thin adapter in `starter/`, and the actual
retrieval engine in `multi_retrieval/`.

```
starter/retrieve.py : retrieve(prep, state)
   ├─ _slots_from(state)        SessionState  →  Slots
   ├─ _multi_retriever(prep)    build the engine once, cache it
   └─ DualTrackRetriever.retrieve(DualQuery(slots, intent, top_k=200))
          │
          ├─ 0.  CatalogIndex            FTS5 + category postings   (built at startup)
          ├─ 1.  HardFilter.apply()      who is eligible at all
          ├─ 2.  KeywordRoute.search()   BM25            ─┐
          │      CategoryRoute.search()  IDF overlap      ├─ each → {product_int: score}
          │      VectorRoute.search()    cosine (OFF)    ─┘
          ├─ 3.  fuse.fuse()             merge by reciprocal rank
          ├─ 4.  PopularityPrior.blend() tie-break the top 200
          └─ 5.  fuse.order()            sort, cut at 200
          │
          ▼
      DualResult(items=[Candidate(asin, score, routes)], pool_size, route_sizes, ...)
          │
          └─ .parent_asins  →  list[str]      ← scores are dropped HERE
```

### 7a. The adapter — `starter/retrieve.py`

**The switch.** `USE_MULTI_RETRIEVAL = True` (line 43). Set it to `False` and
`_original_retrieve()` (lines 284-310) takes over — the earlier hand-rolled
`_keyword`/`_category`/`_vector` fusion, kept intact. Measured inside this pipeline with
everything else unchanged: `_original_retrieve` **0.5560**, `multi_retrieval` **0.6995**.

**Build once.** `_multi_retriever(prep)` (217-237) constructs `DualTrackRetriever` on
first use and caches it in a module global. Indexing 50,000 products takes ~1.8s, so it
must not happen per turn. It also asserts the engine indexed the same catalog
`Preprocessing` loaded — a mismatch would produce candidates the rest of the pipeline
can't resolve, and would fail silently rather than loudly. Override the path with
`TECHJAM_CATALOG`.

**Translate.** `_slots_from(state)` (240-258) maps constraints into typed slots:

```python
_ATTRIBUTE_TO_SLOT = {"material": "material", "color": "color", "size": "size",
                      "style": "style", "use_case": "occasion", "brand": "brand"}
```

First value wins per slot; everything unmapped — including every `feature` constraint —
goes into `free_text`, where the keyword and vector routes still see it. Nothing is
thrown away.

**`category_trusted` stays `False`, on purpose.** `state.py` *infers* the category from
the customer's words, and filtering on an inferred category was measured to cost **0.11**:
the target survived only 8% of turns, while some route had already found it 99.2% of the
time. A category quoted word-for-word would be safe to filter on; a guessed one is not.

**Call and guard.**

```python
result = _multi_retriever(prep).retrieve(DualQuery(
    slots=_slots_from(state), intent=state.scenario, top_k=_POOL_SIZE,   # 200
))
return result.parent_asins or _fallback(prep, state)
```

`_fallback()` (210-214) is the category bucket by popularity, else the whole catalog by
popularity, capped at 200. **An empty list is a guaranteed miss; a mediocre list costs
one turn.** The engine already falls back internally, so this is belt-and-braces.

### 7b. Step 0 — the index (`multi_retrieval/index.py`)

Built once, at `DualTrackRetriever.__init__`. One pass over the catalog:

- an in-memory **SQLite FTS5** virtual table over `title, categories, features, details,
  store, description`, inserted with an explicit `rowid` equal to the product's integer
  index — so BM25 hands back the same numbering every other route uses, with no
  translation layer. **Products are integer indexes throughout this package**; ASINs
  only reappear at the very end.
- `category_postings[token] → array of product ints`, from the category path only
- parallel arrays `price`, `rating`, `reviews` (`parse_number()` fails softly — some
  catalog prices are the string `"-"`)
- `embed_text` — `title + features + description`, truncated to 600 chars, only used if
  the vector route is on

BM25 column weights (`BM25_WEIGHTS`, line 30) are
`(0, 6, 4, 2.5, 2.5, 1.5, 1)` — `parent_asin` unindexed, then **title heaviest,
description lightest** because description is mostly marketing copy.

`search_bm25()` (155-176) negates SQLite's score (bm25 is lower-is-better; everything in
this package agrees higher-is-better) and **returns nothing rather than raising on a
malformed expression** — query text comes from customer input and must never crash a turn.

### 7c. Step 1 — the hard filter (`multi_retrieval/filters.py`)

Decides who is *eligible*, before any ranking. Only these slots may exclude anything:

```python
HARD_SLOTS = ("item", "brand", "department")     # + "category" ONLY if category_trusted
# plus numeric: price_max, min_rating, min_reviews
```

`HardFilter.apply()` (108-138) resolves each to a product set, sorts them **smallest
first** (so the pool shrinks fastest on the strongest evidence), and intersects. Three
rules govern it:

- **A filter that would empty the set is skipped, not enforced.** Returning nothing is a
  guaranteed miss.
- **Unknown is not a violation.** Only 10,410 of 50,000 products carry a price, so
  treating a missing price as "over budget" would discard four fifths of the catalog.
- **A filter matching more than `TEXT_FILTER_CAP = 8000` products isn't filtering** — it
  is skipped as unselective.

It returns `FilterOutcome(allowed, applied, skipped)`, where `allowed is None` means "no
restriction".

> **In the shipped configuration this stage usually does nothing.** The rule-based
> extractor labels almost everything `feature`, and `feature` maps to no hard slot — so
> `slots.hard` comes back empty and nothing is filtered. It only comes alive with a
> configured LLM producing typed slots.

### 7d. Step 2 — the routes

Each route independently returns `{product_int: score}`. Scores are never compared
across routes.

**`routes/keyword.py` — BM25 over FTS5.** The primary route.
`build_expression()` (26-50) turns slot values into one MATCH expression: for every
phrase, a quoted whole phrase (so *"hiking boot"* beats a product mentioning hiking in
one place and boots in another) plus each individual word, OR-ed together, capped at
`MAX_TERMS = 40`. **The expression is rebuilt from re-tokenised words**, so raw
punctuation in customer text can never reach the fragile FTS5 parser. `ROUTE_LIMIT = 500`.

**`routes/category.py` — IDF-weighted overlap** on the `(category, item, department)`
slots against the category-path index. Weighting by IDF is what makes this work: every
product in this catalog sits under *"Clothing, Shoes & Jewelry"*, so `"jewelry"` appears
in all 50,000 paths and IDF drives it to exactly zero, where a plain token count would
treat it as evidence. Scores are normalised to sum-of-weights, capped at 500. **This
route only ever adds score — it never removes a candidate.**

**`routes/vector.py` — dense similarity.** Constructed **only if an `embedder` is passed
in**, and `starter/retrieve.py` passes none. **So the shipped agent runs two routes, not
three.** The wiring exists so turning it on changes nothing else.

**Backoff** (`router.py:91-107`): if the hard filter silenced every route, the routes are
re-run unfiltered and the filters are recorded as skipped. The constraints were wrong
about this catalog, not about the customer.

### 7e. Step 3 — fusion (`multi_retrieval/fuse.py`)

Reciprocal rank fusion. Each route contributes by **rank**, not magnitude:

```
combined[product] += weight / (RRF_K + position)          RRF_K = 60
```

That is the property that matters when BM25 scores, IDF overlap fractions and cosine
similarities all arrive on completely different scales. (`mode="additive"` also exists,
min-max normalising each route first, so both can be swept.)

The weights come from the intent — this is the "dual track" part
(`types.py:DEFAULT_TRACKS`):

| intent | keyword | category | vector |
|---|---|---|---|
| `buying` | 1.0 | 0.2 | 0.1 |
| `browsing` | 1.0 | 0.6 | 0.1 |

A buying turn has explicit words to match, so keyword dominates. A browsing turn has
little to go on, so the category signal counts for three times as much. *Honest caveat
from the source: separating the two tracks is worth +0.0012 over identical weights,
which is inside the noise band.*

If fusion produced nothing at all, `_fallback()` (`router.py:136-144`) returns the 200
most-reviewed products. **Never empty.**

### 7f. Step 4 — the popularity prior (`routes/prior.py`)

Not a route: it fetches nothing and has no opinion about the query. It breaks ties the
text routes can't. The hidden targets are real purchase records, so they skew heavily
towards products people actually buy.

`blend()` (42-75) does two things that are easy to get wrong:

- **Only the shortlist** (top 200). Applied to the whole pool, a heavily reviewed but
  irrelevant product climbs into the top ten and displaces a relevant one.
- **Normalise the fused scores to [0,1] first.** RRF puts rank 1 and rank 2 about 0.0003
  apart, so an unscaled prior of 0.2 would simply overrule the routes.

```
blended = normalised_fused_score + 0.2 * normalised_log1p(review_count)
```

Measured: 0.8174 without it, 0.8396 with. Letting popularity dominate scores marginally
higher on hit rate but drops MRR from 0.659 to 0.620 — so it stays a tie-break and lets
evidence win.

### 7g. Step 5 — order and cut

```python
ranked = fuse.order(combined, self.index.ids, min(query.top_k, self.pool_limit))
```

Sorted best-first, **ties broken on `parent_asin`** so the output is reproducible run to
run. `min(200, 200) = 200`.

### 7h. So what does the pool actually look like?

**It is a cap of 200, not a fixed size of 200.** `_POOL_SIZE = 200` goes in as
`DualQuery.top_k`; the router cuts at `min(top_k, POOL_LIMIT)`. You get fewer whenever
fusion found fewer. Measured on the real catalog:

| Query | Pool size |
|---|---|
| `"Kandinsky"` | **1** |
| `"pentagram wicca"` | **21** |
| a typical session turn | **200** |
| `"zzzznotaword"` (no route matched) | **200**, via the popularity fallback |

**And it carries no scores by the time `rank.py` sees it.** Inside the engine every
candidate is fully scored and explained:

```python
Candidate(parent_asin="B01DS32E9U", score=1.0215,
          routes={"keyword": 15.003, "category": 1.0})
DualResult(items=[...], pool_size=200, filtered_size=50000,
           route_sizes={"keyword": 500, "category": 500},
           filters_applied=[], filters_skipped=[])
```

But `starter/retrieve.py` returns `result.parent_asins` — a bare `list[str]`. **Every
score and every diagnostic is discarded at that one line.**

That's deliberate. `rank.py` re-scores the pool from scratch against the accumulated
`SessionState`, and letting it do so measured **+0.03** over `multi_retrieval` ordering
the final ten itself. Retrieval's job is to decide *which* candidates `rank.py` sees;
its ordering is a membership signal, not a prior. (If you ever want to fuse the
retrieval score into ranking, the plumbing is already there — you just have to stop
throwing it away.)

---

## 8. Stage E — `ask.decide(prep, state, pool, turn) → TurnPolicy`

`starter/ask.py:94-112`. Runs **before** ranking, on the **full ~200 pool**.

Roughly two solid constraints is what collapses the candidate set, so the objective is
reaching that in as few turns as possible. That means asking about the attribute that
best **splits** the pool — not the one that sounds most natural. An attribute every
candidate shares tells us nothing.

```
1. _candidates(state)        REACHABLE_ATTRIBUTES − already asked − dead_attributes
                             = material, color, size, style, use_case, budget, feature
2. _split_quality(prep, pool, attribute)  for each, over the first 400 candidates:

       coverage  = (candidates having ANY value for this attribute) / sample size
       impurity  = 1 − Σ p²          (Gini: max when values are evenly spread, 0 when
                                      every candidate gives the same answer)
       score     = coverage × impurity

3. best score > _MIN_SPLIT (0.05)  →  ask it, and append it to state.asked
   otherwise                       →  ask_attribute = None, say nothing, let the
                                      customer lead rather than burn a turn
```

Values come from `prep.product_keys[asin]` filtered by `prep.attribute_of(key)`.

**Never asks `brand` or `category`** — they're deliberately absent from
`REACHABLE_ATTRIBUTES`. Category is already inferred from what the customer said, and
brand isn't something the index can act on as a constraint.

Returns:

```python
TurnPolicy(ask_attribute="material", list_width=10,
           message="What material are you after?")
```

`_message()` (81-91) is a template lookup today, and is the obvious seam for an LLM —
whatever replaces it **must still return a plain `str`**, or the evaluator discards the
whole turn.

---

## 9. Stage D — `rank.rank(prep, state, pool, top_k) → list[str]`

`starter/rank.py:78-94`. The last step.

```
score(asin) =  Σ  weight × idf(key)      over every RESOLVED constraint matching asin
             + 0.35 × popularity(asin)                        POP_W
             − budget contradiction penalty
```

- `_matched_sets()` (46-60) precomputes `(weight × idf, posting set)` **once per turn**,
  not once per candidate — otherwise scoring is `O(pool × constraints)` index lookups
  instead of `O(constraints)`.
- Rare matches count far more than common ones. Matching `"Triple Moon Pentagram Symbol"`
  is worth vastly more than matching `"Imported"`.
- `_penalty()` (63-75) bites **only** on a product with a *known* price above the
  tightest stated budget cap, scaled by relative overage and clamped at 2×
  (`CONTRADICTION_W = 2.0`). A null price is unknown, not a violation — 79% of the
  catalog has no price, and penalising the unknown would bury most of it the moment
  anyone mentions money.

```python
scored.sort(key=lambda pair: -pair[0])
return [asin for _score, asin in scored[:top_k]]     # the ONLY truncation
```

**Score the full pool, then cut.** Never rank an already-shortened list: the candidate
you want is often not in the first slice, and truncating before scoring throws it away
with no way to get it back.

---

## 10. One real session, all the way through

`public_0001` — `scenario_type: buying`, hidden target `B09PYB7B6Z`. Every value below
is a real trace, not an illustration.

**The target product**

```
title:      QIAN0813 Celttic Knot Triple Moon Pentagram Pentacle Star Wicca Pendant Necklace ...
categories: ["Clothing, Shoes & Jewelry", "Boys", "Jewelry", "Necklaces"]
price: 9.99   rating_number: 490
```

**Evaluator → intent card** (from the product's own features)

```
hard_constraints: ["Material:alloy", "Triple Moon Pentagram Symbol"]
soft_preferences: ["The Triple Moon represents the Phases of the Moon ...", "♥ a special gift to ..."]
coarse_category:  "Jewelry Necklaces"
```

**Turn 1 message**

```
"I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy."
```

**Stage A — `extract.extract()`** (rule-based path, no LLM configured)

```
intent=browsing  override=False  no_preference=None
  text='jewelry'      attr=feature  hard=False  w=0.35  key='jewelry'     ← resolved
  text='necklaces'    attr=feature  hard=False  w=0.35  key=''
  text='requirement'  attr=feature  hard=False  w=0.35  key=''
  text='material'     attr=feature  hard=False  w=0.35  key=''
  text='alloy'        attr=feature  hard=False  w=0.35  key='alloy'       ← resolved
```

Three things are visible here, and all three are real limitations of the default path:
`intent` is `browsing` even though the scenario is *buying* (the rule path only says
buying when it finds a budget); every constraint is `feature` and soft; and
`"requirement"` — a word from the evaluator's sentence template — leaked in as a
constraint.

**Stage B — `state.update()`**

```
scenario='browsing'   category='Jewelry Necklaces & Pendants'   constraints=5
```

The category was *inferred* by token vote — note it isn't the evaluator's
`"Jewelry Necklaces"`, it's the nearest real bucket.

**Stage C — `retrieve.retrieve()`**

```
Slots(category='Jewelry Necklaces & Pendants',
      free_text=['jewelry','necklaces','requirement','material','alloy'])
  hard  = {}          ← nothing to filter on: 'feature' maps to no hard slot
  soft  = {}
  phrases = ['jewelry','necklaces','requirement','material','alloy']

DualResult: pool_size=200  filtered_size=50000  items=200
            route_sizes={'keyword': 500, 'category': 500}
            filters_applied=[]  filters_skipped=[]

  1. B01DS32E9U  score=1.0215  routes={keyword: 15.003, category: 1.0}
  2. B06XBRZC6T  score=0.9902  routes={keyword: 14.495, category: 1.0}
  3. B06VTSYTS8  score=0.9687  routes={keyword: 14.516, category: 1.0}
  4. B01IGYIPFI  score=0.8853  routes={keyword: 22.373, category: 0.398}
  5. B079KSQDR3  score=0.7966  routes={keyword: 18.939}

pool = 200 asins    TARGET IS AT POSITION 95
```

`filtered_size=50000` confirms the hard filter did nothing. Both routes hit their 500
cap, fusion merged them, and the target survived the cut to 200 — at position 95, which
would be a miss if retrieval order were the final answer.

**Stage E — `ask.decide()`**

```
ask_attribute='feature'   list_width=10   message='Is there a specific feature you need?'
```

Had this turn missed, the evaluator would have replied with the next undisclosed
`feature`-classified item from the intent card — i.e. *"Triple Moon Pentagram Symbol"*,
which is exactly the rare string that would pin the target.

**Stage D — `rank.rank()`**

```
['B07NND729D','B096LVYWCP','B08N5SJZ1H','B08YYHDJD1','B077SY7DSS',
 'B07TSM177T','B07RPV75CN','B07RZQDJPX','B09PYB7B6Z','B07JFL14ZD']
                                          ^^^^^^^^^^ target, rank 9
```

**Hit on turn 1 at rank 9.** Retrieval had it at 95; re-scoring the full pool against
the session's constraints moved it to 9. That single number is the entire argument for
why `rank.py` re-scores instead of trusting retrieval's order.

Session result: `hit=True`, `first_hit_turn=1`, `reciprocal_rank=1/9`.

---

## 11. Quick reference

### The call chain

| # | Owner | File | Function | In | Out |
|---|---|---|---|---|---|
| — | — | `starter/preprocessing.py` | `build(path)` | catalog path | `Preprocessing` (once) |
| — | — | `starter/agent.py` | `respond(sid, msg, turn, top_k)` | one turn | response dict |
| A | Owner A | `starter/extract.py` | `extract(message, turn, state)` | `str` | `Extraction` |
| B | Owner B | `starter/state.py` | `update(state, extraction)` | `Extraction` | `SessionState` |
| C | Owner C | `starter/retrieve.py` | `retrieve(prep, state)` | `SessionState` | `list[str]`, ≤200 |
| C | — | `multi_retrieval/router.py` | `DualTrackRetriever.retrieve(query)` | `DualQuery` | `DualResult` |
| E | Owner E | `starter/ask.py` | `decide(prep, state, pool, turn)` | pool | `TurnPolicy` |
| D | Owner D | `starter/rank.py` | `rank(prep, state, pool, top_k)` | pool | `list[str]`, ≤10 |

### Every constant you might want to sweep

| Constant | Value | Where |
|---|---|---|
| `_POOL_SIZE` | 200 | `starter/retrieve.py:52` |
| `POOL_LIMIT` | 200 | `multi_retrieval/router.py:35` |
| `ROUTE_LIMIT` | 500 | `multi_retrieval/routes/keyword.py:22`, `category.py:18` |
| `MAX_TERMS` | 40 | `multi_retrieval/routes/keyword.py:23` |
| `TEXT_FILTER_CAP` | 8000 | `multi_retrieval/filters.py:34` |
| `BM25_WEIGHTS` | `(0,6,4,2.5,2.5,1.5,1)` | `multi_retrieval/index.py:30` |
| `RRF_K` | 60 | `multi_retrieval/fuse.py:18` |
| `DEFAULT_TRACKS` | buying `1.0/0.2/0.1`, browsing `1.0/0.6/0.1` | `multi_retrieval/types.py:166` |
| `DEFAULT_WEIGHT` (prior) | 0.2 | `multi_retrieval/routes/prior.py:30` |
| `POP_W` | 0.35 | `starter/rank.py:27` |
| `CONTRADICTION_W` | 2.0 | `starter/rank.py:29` |
| `_LIST_WIDTH` | 10 | `starter/ask.py:29` |
| `_MIN_SPLIT` | 0.05 | `starter/ask.py:33` |
| `_MAX_POOL_SAMPLE` | 400 | `starter/ask.py:34` |
| `_SOFT_WEIGHT` | 0.35 | `starter/extract.py:44` |
| `MIN_PREFIX_LEN` / `PREFIX_FANOUT_CAP` | 12 / 64 | `starter/preprocessing.py:38-39` |
| `USE_MULTI_RETRIEVAL` | `True` | `starter/retrieve.py:43` |

### Five things that surprise people

1. **`top_k` is ignored.** `respond()` takes it; `ask.py:_LIST_WIDTH` decides.
2. **The candidate pool carries no scores.** Retrieval's ordering is thrown away and
   `rank.py` starts over.
3. **The vector route is off.** No embedder is passed, so two routes run, not three.
4. **The LLM is off by default.** `_NullClient` always fails, so the rule-based
   extractor is what you're actually measuring unless you set `TECHJAM_LLM_PROVIDER`.
5. **`ask.decide()` runs before `rank.rank()`** and sees all 200 candidates.

### Running it

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl \
                                    --dataset data/public_set.jsonl \
                                    --output results.json
```

---

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — the design rationale and the honest list of gaps
- [RETRIEVAL.md](RETRIEVAL.md) — retrieval's measurement history, the vector route, and
  things tried and cut
- [OWNERSHIP.md](OWNERSHIP.md) — who owns which file
- [agent_api_contract.json](agent_api_contract.json) — the fixed request/response schema
