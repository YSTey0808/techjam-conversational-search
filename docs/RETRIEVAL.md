# The Retrieval Stage — `multi_retrieval/`

How Owner C's candidate generation works, end to end.

Read [OWNERSHIP.md](OWNERSHIP.md) first if you haven't. This document covers one
box in that diagram: given what the customer has told us so far, produce the
~200 products `rank.py` should score.

---

## 1. Where it sits

```
message ──► extract.py ──► state.py ──► retrieve.py ──► ask.py ──► rank.py ──► 10 items
            Owner A        Owner B      Owner C         Owner E     Owner D
                                          │
                                   multi_retrieval/
```

`starter/retrieve.py` is a thin adapter. It converts `SessionState` into the
package's own input type and hands back a list of `parent_asin`. Its signature
is unchanged, so `agent.py` needed no edit.

**It does not rank.** `agent.py` already calls `rank.rank()` afterwards, and
letting it score the pool measured 0.03 better than `multi_retrieval` ordering
results itself. This stage decides *which* candidates `rank.py` sees; nothing
more.

To switch back to the original implementation, set `USE_MULTI_RETRIEVAL = False`
at the top of `starter/retrieve.py`. The original code is preserved there as
`_original_retrieve` and nothing else needs changing.

---

## 2. What it reads from the catalog

`data/catalog.jsonl`, 50,000 products, read once at startup and never written.

Different parts of the system use different fields, on purpose:

| Field | Used for |
|---|---|
| `title`, `features`, `details`, `description`, `store`, `categories` | full-text search (the keyword route) |
| `categories` | the category route, and the bucket key below |
| `price`, `average_rating`, `rating_number` | numeric filters |
| `rating_number` | the popularity prior |
| `title` + `features` + `description` | the text that gets embedded, if the vector route is on |

So yes — `details` and `description` are both used. `description` feeds the
full-text index but is weighted lowest, because it is mostly marketing copy.

---

## 3. What it expects as input

```python
DualQuery(
    slots = Slots(category="Jewelry Necklaces", material="alloy",
                  free_text=["Triple Moon Pentagram Symbol"]),
    intent = "buying",          # or "browsing"
    top_k  = 200,
)
```

**Deciding which part of a sentence is the category and which is a fact is not
this package's job.** It receives that already separated. In the pipeline
`state.py` does it, and `starter/retrieve.py` maps `Constraint.attribute` onto
slot names — `material` → `material`, `use_case` → `occasion`, and anything
unmapped into `free_text` so it is never silently dropped.

That boundary matters. The quality of what arrives in `Slots` limits everything
downstream; see [section 10](#10-the-honest-limit).

---

## 4. What gets built at startup

One pass over the file, about 1.3 seconds, building five things. Terms are
explained as they appear.

**An FTS5 table.** FTS5 is SQLite's built-in full-text search. You declare a
virtual table whose columns are indexed word by word, then query it with `MATCH`
and let SQLite rank the results. It's why this package needs no search library.
Each product is inserted with its integer position as the row id, so a search
result points straight back into every other index with no translation.

**An inverted index over category paths.** A normal index maps *document →
words*. Invert it and you get *word → documents*: `"necklaces"` → `[12, 847,
3301, …]`. Those lists are called posting lists, and they make "find everything
tagged with this word" instant instead of a 50,000-row scan. 803 distinct
category words in this catalog.

**Numeric columns** — three plain Python lists (`price`, `rating`, `reviews`),
indexed by product number. Not numpy; they're only ever read one element at a
time. Note that just 10,410 of 50,000 products have a price at all.

**Product text for embedding** — one truncated string per product, only used if
the vector route is switched on.

**Nothing else.** No embeddings by default. See [section 8](#8-the-vector-route).

### The category bucket key

Worth its own explanation, because the key does not look like a category name.

`coarse_category()` splits every breadcrumb on commas, throws away the generic
`Clothing` variants, keeps the **last two** survivors, and joins them:

```
["Clothing, Shoes & Jewelry", "Boys", "Jewelry", "Necklaces"]

  "Clothing, Shoes & Jewelry"  ->  "Clothing"          dropped
                               ->  "Shoes & Jewelry"   kept
  "Boys"                                               kept
  "Jewelry"                                            kept
  "Necklaces"                                          kept

  last two, joined:  "Jewelry Necklaces"
```

Some keys come out looking like nonsense — `"Shoes & Jewelry Westlake"` exists
because a shop name leaked into a category path. That is fine and even useful: a
weird key is *more* distinctive, because fewer products share it. We never
interpret the string, only match it.

---

## 5. What happens on each call

Three stages, in this order.

```
1. FILTER   compute which products are eligible
2. ROUTES   three independent searches over the WHOLE catalog
3. MERGE    intersect with the eligible set, fuse, order, truncate
```

**The routes run in parallel, in the sense that matters:** each searches the
entire index independently, knows nothing about the others, and returns
`{product_index: score}`. None of them filters another's input.

**The filter is applied *after* the routes, not before.** It is computed first,
but it narrows the routes' *results* by intersection. This has a consequence
worth knowing: each route returns at most its top 500 out of 50,000, so if the
target isn't in a route's top 500 for the whole catalog, filtering afterwards
cannot rescue it — a filter can only remove, never add.

If filtering silences every route, the constraints were wrong about this catalog
rather than about the customer, so the routes re-run unfiltered and the
abandoned filters are recorded in `filters_skipped`.

### The eligible set, and what "intersect" means

The **eligible set** is a set of product numbers — the products allowed to
appear in the answer at all. It is built by `filters.py` before the routes run,
from two kinds of slot:

| Slot | Becomes eligible if |
|---|---|
| `item`, `brand`, `department` | the product's indexed text matches the value |
| `price_max` | its price is at or under the limit, **or it has no price** |
| `min_rating`, `min_reviews` | it meets the threshold |

`category` is **not** in that list by default. Section 9 explains why at length.

Each of those produces a set, and they are combined by **intersection** —
keeping only products present in *every* set. Set intersection is written `∩`:

```
item="boot"        -> {12, 84, 91, 205, 330, …}   1,847 products
min_reviews=100    -> {12, 91, 402, …}           9,120 products
                                    ∩
eligible set       -> {12, 91, …}                  612 products
```

Then each route's results are narrowed the same way. A route returns its top 500
from the whole catalog; anything not in the eligible set is dropped from that
route's scores before fusion:

```
keyword route returned  500 products
                                    ∩  eligible set
survivors                31 products  ->  these go into fusion
```

Three rules govern how the eligible set is built, and each exists because the
obvious alternative loses sessions.

**Most selective first.** Constraints are sorted by how few products they match,
smallest set first, so the pool shrinks fastest on the strongest evidence and
anything dropped by backoff is the weakest constraint.

**A filter that would empty the set is skipped, not enforced.** If intersecting
one more constraint leaves nothing, that constraint is abandoned and recorded in
`filters_skipped`. Returning nothing is a guaranteed miss; returning a loose
list costs at most one turn.

**Unknown is not a violation.** Only 10,410 of 50,000 products carry a price, so
treating a missing price as "over budget" would discard four fifths of the
catalog over a constraint those products were never checked against. A product
with no price survives every budget.

A filter matching more than 8,000 products is also dropped — something matching
a sixth of the catalog is not narrowing anything, and applying it only costs
time.

Pass `layered=False` to `DualTrackRetriever` to skip this stage entirely and let
every slot simply feed the routes. Measured difference: **+0.0003**, so the
layer is nearly redundant with what the routes already do.

---

## 6. The three routes

### keyword — BM25 over the full-text index

The primary route. It builds one FTS5 expression from the slot values: every
word `OR`'d together, plus a quoted phrase for each multi-word slot.

`OR` is an FTS5 operator meaning *match documents containing any of these
terms*, as opposed to `AND`, which would require all of them. So this route
scores loosely rather than narrowing — a product matching four of five words
still appears, just lower.

BM25 is the standard relevance ranking: rarer words count for more, and matches
in short fields count for more than matches in long ones. Column weights are
title 6.0, categories 4.0, features 2.5, details 2.5, store 1.5, description
1.0. SQLite's `bm25()` returns lower-is-better, so scores are negated on the way
out and every route in this package agrees that higher means better.

Returns the top 500.

### category — IDF-weighted overlap

Tokenises the `category`, `item` and `department` slots, looks each word up in
the inverted index, and scores by **IDF-weighted** overlap.

IDF (inverse document frequency) is `log(total / number_containing)`. A word in
every product scores 0; a word in one product scores about 10.8. It makes rare
words automatically outweigh common ones with no hand-tuning.

Concretely: every product in this catalog sits under "Clothing, Shoes &
Jewelry", so `"jewelry"` appears in all 50,000 category paths and its IDF is
exactly **0.0** — it contributes nothing, correctly, without anyone special-casing
it.

This route **never removes a candidate.** See [section 9](#9-the-most-expensive-bug).

### vector — dense similarity

Off by default. Covered in [section 8](#8-the-vector-route).

---

## 7. Merging, ordering, and how many come back

**Fusion.** Reciprocal Rank Fusion by default: each route contributes
`weight / (60 + rank)`. It combines *positions*, not scores, which matters
because BM25 scores, IDF fractions and cosine similarities are on completely
different scales — a route scoring in the thousands must not outvote one scoring
in decimals just because its numbers are bigger.

Weighted-additive fusion is available as `fusion="additive"` and keeps magnitude
instead. Both ship so the choice stays measurable.

**Intent selects the weights**, and nothing else:

| | keyword | category | vector |
|---|---|---|---|
| buying | 1.0 | 0.2 | 0.1 |
| browsing | 1.0 | 0.6 | 0.1 |

A zero weight skips that route entirely.

Be honest about what this buys: separating the two tracks measured **+0.0012**
against using identical weights everywhere, which is inside run-to-run noise.
The mechanism works and the rankings genuinely differ, but on this evaluator it
is not currently earning its keep. Don't cite it as evidence that dual-track
routing pays.

**The popularity prior.** After fusion, the top 200 are reordered by fused score
plus a small popularity term. The hidden targets are real purchase records, so
they skew heavily towards products people actually buy. It is applied *after*
fusion, on scores normalised to [0, 1], and only to the shortlist — all three
details matter, and getting any of them wrong makes popularity overrule
evidence rather than break its ties.

**Ordering and size.** `fuse.order()` sorts and truncates to `min(top_k, 200)`.

A question that comes up: if `rank.py` re-scores everything afterwards, why sort
at all? Because **selection requires ordering.** Truncating to 200 means knowing
which 200 scored highest, and you cannot know that without ranking them. What
`rank.py` discards is only the relative order *among* the survivors. The
tie-break on `parent_asin` is not cosmetic either — under RRF many products tie
exactly, so at the 200-item boundary it decides *membership*.

The same logic explains why the popularity prior stops mattering once `rank.py`
runs: anything that only affects *ordering* is free but pointless there, while
anything affecting *membership* still counts. The fusion weights still matter,
for exactly that reason.

**Output** is a `DualResult`: the ranked `items`, plus `pool_size`,
`filtered_size`, `route_sizes`, `filters_applied` and `filters_skipped`. Those
last fields exist to explain a turn when something goes wrong — and would have
exposed the bug in section 9 far earlier if anyone had looked at them.

It never returns an empty list. An empty list is a guaranteed miss; a mediocre
one costs at most a turn.

---

## 8. The vector route

Built, measured three ways, and **switched off**.

"Embedding" means turning text into a fixed-length list of numbers — 384 of
them, whatever the input length. Similar *meanings* produce vectors pointing in
similar directions even with no shared words:

```
"waterproof hiking boot"  vs  "leather ankle boot for trails"   ->  0.686
"waterproof hiking boot"  vs  "silver moon necklace"            -> -0.013
```

50,000 products × 384 numbers is a 50,000 × 384 grid of floats, held in a
**numpy array** — 76.8 MB. numpy is the container and the arithmetic engine, not
the vector itself; the 384 numbers are the vector. Searching is one matrix
multiply: encode the query, multiply, get 50,000 similarity scores at once. The
matrix is cached to `.cache/multi_retrieval/` so the 105-second encode happens
once.

Measured in the shipping configuration:

| Backend | score |
|---|---|
| **none** | **0.8602** |
| `HashingEmbedder` (dependency-free) | 0.8601 |
| `sentence-transformers` (real encoder, 2 GB of torch) | 0.8579 |

The encoder works correctly — it does rank the right product first on our test
query. But semantic similarity is the wrong question here. The target is not the
product most *like* the query; it is the one product whose text contains the
phrase the customer quoted. Embeddings deliberately blur exactly the
distinctions that identify it.

To try it anyway: `pip install -r requirements-multi-retrieval.txt`, then pass
`embedder=SentenceTransformerEmbedder()`.

---

## 9. The most expensive bug

Worth reading even if you never touch this code, because the failure mode
generalises.

`category` was originally a **hard filter** — products had to carry every word
of the category slot. It looked reasonable and it destroyed recall. In the
sessions this package failed, a route had already surfaced the target **99.2%**
of the time, and it survived the filter **8%** of the time.

The cause: `state.py` *infers* the category from scattered words, producing
things like `'Active Shirts & Tees T-Shirts'`. Requiring a product's path to
carry every one of those words excludes the right answer. And the backoff didn't
catch it, because **backoff protects against an empty pool, not a wrong one** —
the pool looked perfectly healthy, it just didn't contain the target.

Fixing it: **0.7478 → 0.8601.**

The rule now: a filter ships only if you can say *why* it cannot remove the
target. `category` earns weight through the category route and removes nothing,
unless the caller sets `Slots.category_trusted` to vouch that the customer said
it word-for-word. A verbatim category is safe to filter on and worth about the
same in the other direction — the distinction is trust, not the field.

`tests/test_mr_filters.py::test_category_never_filters` pins this so it cannot
come back.

---

## 10. The honest limit

Everything above is worth less than the stage before it.

| Change | Effect on the pipeline |
|---|---|
| Swapping in `multi_retrieval` | 0.5517 → **0.6887** |
| Fixing extraction (measured with a stand-in) | 0.5560 → **0.8389** |
| Both | → **0.8601** |

Retrieval can only search for what extraction hands it. Today
`extract.py`'s rule-based fallback turns `"Triple Moon Pentagram Symbol"` into
`triple`, `moon`, `symbol` — and no retrieval system can recover a phrase
destroyed upstream. That is Owner A's territory, and their own notes already
identify connecting a real model as the biggest available upgrade.

Two smaller upstream items with measured value: `state.py` passing the
**verbatim** category rather than an inferred one was worth about +0.12, and
`ask.py`'s hardcoded `list_width = 10` leaves score on the table, because the
evaluator freezes your rank the moment the target appears.

---

## 11. Running it

```bash
# the whole agent, the normal way
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl \
                                     --dataset data/public_set.jsonl

# retrieval alone, with a scaffolding slot filler (isolates retrieval quality)
python3 scripts/score_multi_retrieval.py
python3 scripts/score_multi_retrieval.py --fusion additive --no-layered --vector hashing

# each retrieval stage inside the real pipeline, with and without fixed extraction
python3 scripts/score_integrated.py

python3 -m unittest discover tests
```

The slot filler inside `scripts/score_multi_retrieval.py` reads the simulator's
sentence templates. It is scaffolding for measuring retrieval on its own, is
labelled as such in the file, and is not part of the agent.

---

## 12. Things tried and cut

Each was implemented, measured against the real evaluator, and rejected. They
are listed because knowing what *doesn't* work is most of what the measurements
bought.

| Idea | Result |
|---|---|
| `category` as a hard filter | 0.7478 vs 0.8601 — see section 9 |
| Vector route, any backend | ≤ 0.8601 vs 0.8602 without |
| Seeded progressive intersection (ported from `_original_retrieve`) | 0.8220 vs 0.8601 |
| `multi_retrieval` doing its own ranking | 0.8102 vs 0.8601 with `rank.py` |
| Per-intent route weights | +0.0012, inside noise |
| Hard-then-soft constraint layering | +0.0003 |
| Passing `Constraint.key` instead of `.text` | −0.007 |

`keyword_mode="seeded"` and the vector route both remain in the code, off by
default, with their measurements in their docstrings — so they can be re-tested
in one flag if extraction changes.
