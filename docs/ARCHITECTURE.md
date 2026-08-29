# Architecture

A conversational product search agent over a 50,000-item Amazon clothing
catalog. The customer types plain English — *"something waterproof for hiking,
nothing over fifty bucks"* — and each turn we return up to ten products and
one follow-up question.

Who owns which file: [OWNERSHIP.md](OWNERSHIP.md).

---

## 1. The central problem

A shopper and a product listing do not use the same words. They say
**"waterproof"**; the catalog says **"Water Resistant"**. They say
**"stretchy"**; the catalog says **"Spandex"**. Nothing matches literally.

So the pipeline is built around one job: **turn customer words into catalog
words, then use the catalog's own structure to narrow down.**

That job is split in two.

- **`extract.py` proposes.** An LLM reads the sentence and, for every
  requirement, emits `variants` — phrasings a real product page might use.
- **`preprocessing.py` disposes.** Each variant is checked against the actual
  inverted index. The first one that genuinely exists becomes the constraint's
  `key`. Anything that resolves to nothing keeps an empty key and is carried
  anyway, in case another route can use it later.

Everything downstream searches on resolved keys, so retrieval never has to
guess about language.

---

## 2. One turn

`agent.py` is wiring and nothing else — no branching, no fallbacks:

```
respond(session_id, user_message, turn, top_k)
        |
        v
  extract.extract(message, turn, state)      A   English  --> Extraction
        |                                            (LLM, then rules)
        v
  state.update(state, extraction)            B   remember, replace, retract
        |
        v
  retrieve.retrieve(prep, state)             C   --> ~200 candidate asins
        |                                            (3 routes, fused)
        v
  ask.decide(prep, state, pool, turn)        E   --> question + list width
        |
        v
  rank.rank(prep, state, pool, list_width)   D   --> ordered asins
        |
        v
  {"message", "ask_attribute", "recommendations", "usage"}
```

Every decision lives in the module that owns it. If the wiring ever needs an
`if`, that logic is in the wrong file.

---

## 3. Layout

```
starter/
  schema.py         shared dataclasses             no owner
  preprocessing.py  catalog loaded + indexed once  no owner
  extract.py        English -> requirements        OWNER A
  state.py          conversation memory            OWNER B
  retrieve.py       candidate pool                 OWNER C
  rank.py           ordering                       OWNER D
  ask.py            next question + reply text     OWNER E
  agent.py          wiring only
  baseline.py       original starter, reference only - do not develop here
```

Pure standard library. No third-party packages, no vector database, no network
calls except the optional LLM endpoint.

`data/catalog.jsonl` is **not in git** — it is a 50k-row GitHub Release
download. Fetch it yourself and drop it at that path.

There is no unit-test suite. `tests/test_evaluator.py` is the organizer's
original file and does not touch our code.

---

## 4. What each stage does

### `preprocessing.py` — built once in `Agent.__init__`

Read-only afterwards. Nothing is ever rebuilt per turn.

- **Inverted index**: normalised catalog string -> set of `parent_asin`. Two
  tables: a narrow one over each product's most characteristic strings, and a
  broad one over everything. Retrieval starts narrow and widens.
- **`lookup(key)` is prefix-tolerant**, via bisect over the sorted key list, in
  both directions. Long feature strings get truncated, so exact matching alone
  would miss.
- **`idf(key)`** — essential, not optional. `"Imported"` sits on roughly 13,900
  products and `"Machine Wash"` on 8,900. Matching without weighting by rarity
  is close to meaningless.
- **`safe_price(product)` never raises.** 79% of rows have a null price and 117
  hold a string — 112 of those a single mojibake character. **No other file may
  call `float(product["price"])`.**
- **`coarse_category`** — deliberately lossy. 1,136 products land on
  `'Shoes & Jewelry Westlake'` because a shop name leaked into the category
  path. Category is therefore a **soft gate**, never a hard filter.
- **`popularity`** = `log1p(rating_number)`, precomputed.

### `extract.py` — the LLM stage

Sends the message and asks for strict schema-validated JSON: constraints (each
with `text`, `attribute`, `hard`, `variants`), `intent`, `override`,
`no_preference`. Anything malformed is rejected outright rather than
half-trusted.

A slow turn can cost the whole session, so the LLM is on a short leash:
**2-second timeout, one retry, then a rule-based fallback** that reads colours,
materials, budgets, sizes and use-cases directly from the sentence.
`extract()` **never raises** and never returns empty-handed for a non-empty
message.

With no provider configured it uses `NullClient`, which always fails — so the
fallback is what runs by default. Configure a real model with:

```bash
TECHJAM_LLM_PROVIDER=ollama          # or: openai
TECHJAM_LLM_MODEL=llama3.1
TECHJAM_LLM_BASE_URL=http://localhost:11434
TECHJAM_LLM_API_KEY=...              # openai-compatible only
TECHJAM_LLM_TIMEOUT=2.0
```

No keys in the repo, ever. Results are cached per (message, turn, known-so-far).

### `state.py` — conversation memory

Three operations. **Wipe** when the customer retracts. **Replace** when they
restate a single-valued attribute — "black" then "white" swaps, because a
person has one colour in mind at a time. **Append** for anything new. Declined
attributes are marked dead so we never ask twice.

`user_profile` is **soft signal only**: never a constraint, never a filter.

### `retrieve.py` — candidates

Three routes, each returning `{asin: score}`, fused by reciprocal rank so a
fourth could be added without touching anything else.

- **`_keyword`** (primary) intersects hard constraints most-informative-first.
  On an empty intersection it backs off in a deliberate order: drop the least
  informative constraint, then widen to the broad index, and **give up the
  category gate last**. A constraint we cannot find is **skipped, not
  intersected** — an unfindable string is our indexing gap, not evidence the
  product is absent, and intersecting on nothing would wipe the pool.
- **`_category`** weights the category bucket by how specific it is. Soft: it
  only adds weight, never removes a candidate another route found.
- **`_vector`** is a deliberate stub returning `{}`. It exists for wording that
  resolves to no catalog string at all — but if Owner A's variants work well it
  may never fire, so it must earn its dependency with a measurement.

**It never returns an empty list.** An empty list is a guaranteed miss; a
mediocre list costs only a turn.

### `rank.py` — ordering

```
score = sum(weight x idf) over matched constraints
      + POP_W x popularity
      - budget contradiction penalty
```

Rare matches count far more than common ones. The penalty applies only to
products with a **known** price above a stated budget — unknown is not a
violation, or the 79% of the catalog without prices would be buried the moment
anyone mentions money.

**Score the full pool, then cut.** Never rank an already-shortened list.

### `ask.py` — the follow-up

Roughly two solid constraints is what collapses the candidate set, so the aim
is reaching that fast. It picks the attribute that best **splits** the current
pool (`coverage x impurity`), so an attribute every candidate shares scores
zero — its answer would tell us nothing. If nothing splits usefully it asks
nothing and lets the customer lead.

Never asks "brand" or "category". `message` must always be a plain string: the
harness discards the entire turn if it is not.

---

## 5. The contract

Fixed. Do not change it.

```python
Agent(catalog_path)                                  # positional
reset(session_id, user_profile) -> None
respond(session_id, user_message, turn, top_k) -> {
    "message":         str,          # non-str discards the whole turn
    "ask_attribute":   str | None,
    "recommendations": [{"parent_asin": str}, ...],
    "usage":           {"prompt_tokens": int, "completion_tokens": int},
}
```

---

## 6. Known gaps

Honest list, all reproducible today.

1. **No negation handling.** *"nothing formal"* produces a `formal`
   constraint — the opposite of what was meant.
2. **Retraction cue words leak into constraints.** *"actually forget that"*
   currently yields `actually` and `forget` as feature constraints. Small fix
   in the fallback's leftover-token pass.
3. **A broad retraction only clears matching attributes.** *"forget that"* with
   no new attribute leaves earlier requirements standing, so a stale answer can
   survive a turn it should not.
4. **`list_width` is fixed at 10** inside `ask.py`, which cannot see the
   harness's `top_k`.
5. **`_vector` is unbuilt**, so wording that resolves to nothing has no
   semantic path.
6. **Route weights and `POP_W` are guesses** and should be swept, not tuned by
   feel.

Gaps 1-3 all live in Owner A's fallback path and are the highest-value fixes.
