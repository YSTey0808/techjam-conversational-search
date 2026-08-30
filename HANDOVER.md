# Handover — pipeline state, broken parts, and what's next

Findings as of 2026-08-31, `integration` @ `25f1879`. Everything below was measured
against the working tree, not inferred.

**Bottom line:** the pipeline does not run end to end. The `origin/state-extraction`
merge migrated `schema.py` + `extract.py` and deleted `state.py`, but left three
consumers on the old `SessionState` API. Every stage after extraction raises
`AttributeError` on turn 1, and the evaluator silently swallows it.

---

## Part 1 — Broken by the schema migration (do this first)

The merge (a843d83, 25f1879) rewrote `SessionState` from a flat constraint list to a slot
map. Verified by driving each stage with a mocked extractor:

```
BROKEN  retrieve.retrieve   retrieve.py:254   for constraint in state.constraints:
BROKEN  ask.decide          ask.py:94         clarification = state.get("clarification", {})
BROKEN  adapter.rerank      adapter.py:53     for i, c in enumerate(state.constraints)
```

### Component map

| component | state | sites |
| --- | --- | --- |
| [starter/schema.py](starter/schema.py) | **current** — source of truth | — |
| [starter/extract.py](starter/extract.py) | **current** — degrades cleanly with no API key | — |
| [starter/agent.py](starter/agent.py) | **current** — wiring correct as written | — |
| [starter/retrieve.py](starter/retrieve.py) | stale — imports updated, body not | 4 |
| [starter/ask.py](starter/ask.py) | stale — untouched by the merge | 2 |
| [reranker/adapter.py](reranker/adapter.py) | stale — predates the merge | 3 |
| [scripts/score_integrated.py](scripts/score_integrated.py) | stale — imports deleted `starter.state` | 8 |

`multi_retrieval/`, `reranker/reranker.py`, `evaluator/` are unaffected — none touch
`SessionState`.

### The translation table

Nearly every fix is this, applied mechanically:

| old | new |
| --- | --- |
| `state.constraints` → `list[Constraint]` | `state.filled_slots` → `dict[str, Slot]` |
| `state.hard_constraints` | `state.hard_slots` (confidence ≥ `HARD_CONFIDENCE`) |
| `c.attribute` | the dict key |
| `c.text` | `slot.val` |
| `c.key` | `slot.key` (all bound keys: `slot.keys`) |
| `c.hard` | `slot.hard` (property) |
| `state.profile` (dict) | `state.user_profile` (**`UserProfile` dataclass**) |
| `state.get("slots", {})` | `state.slots` |
| `slot.get("val")` | `slot.val`, or `slot.filled` |
| `state.get("clarification",{}).get("asked_attributes",[])` | `state.asked` |
| `state.category`, `state.scenario` | unchanged — survived as properties |

### Checklist

- [ ] **`starter/retrieve.py`** — 4 sites
  - line 90 — type hint `constraints: list[Constraint]` → `list[Slot]` (`Slot` already imported, line 24)
  - line 114 `_score_matches` — iterate `state.filled_slots.items()`; `slot.key` for the lookup, dict key for the attribute
  - line 137 `_keyword` — `[c for c in state.constraints if c.hard and c.key]` → `[s for s in state.hard_slots.values() if s.key]`
  - line 254 `_slots_from` — iterate `state.filled_slots.items()`; dict key replaces `constraint.attribute`, `slot.val` replaces `constraint.text`
  - `_ATTRIBUTE_TO_SLOT` (line 59) still correct; its `size` entry is now dead (no `size` in `SLOT_ATTRIBUTES`) but harmless

- [ ] **`starter/ask.py`** — 2 sites
  - line 78 `_known_attributes` — `state.get("slots", {})` → `state.slots`; `slot.get("val")` → `slot.filled` (the property does exactly the hand-rolled test)
  - line 94 `_asked_attributes` — → `set(state.asked) | state.dead_attributes`
  - leave `_STATE_SLOT_NAME = {"other": "others"}` (line 53) alone — already correct per [schema.py:12-13](starter/schema.py#L12-L13)

- [ ] **`reranker/adapter.py`** — 3 sites
  - lines 52-53 `_constraint_rows` — iterate `state.filled_slots.items()`; **keep** the `f"{attribute}:{i}"` suffixing, `constraint_status` keys must stay unique for `_is_guaranteed`
  - line 61 `_query` — → `[str(s.val) for s in state.filled_slots.values() if s.val]`
  - line 71 `build_request` — **not a rename.** `state.profile or None` → `dataclasses.asdict(state.user_profile)`. `reranker.py` reads the profile as a dict ([reranker.py:98](reranker/reranker.py#L98), [reranker.py:106](reranker/reranker.py#L106)) but `user_profile` is now a dataclass

- [ ] **`scripts/score_integrated.py`** — 8 sites, lowest priority (measurement harness, not the pipeline). `oracle_extract` must write slots via `state.bind()` instead of returning an `Extraction`

### Verify

```powershell
# every stage, mocked extractor, no API key
python - <<'PY'
from starter import ask, preprocessing, retrieve
from starter.schema import SessionState, UserProfile
from reranker import adapter
prep = preprocessing.build('data/catalog.jsonl')
st = SessionState(session_id='s', user_profile=UserProfile.from_dict({}))
st.begin_turn(1); st.record_customer(1, 'I want a black cotton shirt.')
st.bind('category','shirts',confidence=0.9,turn=1)
st.bind('material','cotton',key='cotton',confidence=1.0,turn=1)
st.bind('color','black',key='color: black',confidence=0.8,turn=1)
pool = retrieve.retrieve(prep, st);               print('retrieve ->', len(pool))
policy = ask.decide(prep, st, pool[:200], 1);     print('ask      ->', policy.ask_attribute)
items = adapter.rerank(prep, st, pool[:200], 10); print('rerank   ->', len(items))
PY

python -c "from starter.agent import Agent; a=Agent('data/catalog.jsonl'); a.reset('s',{}); print(a.respond('s','I want a black cotton shirt.',1,10))"
python -m evaluator.local_evaluator
```

**Pass:** three lines, no traceback; 10 recommendations on **turn 1** (not just turn 10);
MTTC well below 10.

### Why the breakage is invisible today

[local_evaluator.py:241-242](evaluator/local_evaluator.py#L241-L242) catches every
exception and substitutes `{"message":"", "ask_attribute":None, "recommendations":[]}`.
Only turn 10 survives, because the `turn >= 10` early return at
[ask.py:207](starter/ask.py#L207) exits before the broken code is reached. So the agent
returns **zero recommendations on nine turns out of ten**, and `MTTC 10.58` in
`results.json` is the fingerprint. Measured on 20 sessions with the crash removed:
**0.2514 → 0.3613**, MTTC 10.6 → 6.7.

---

## Part 2 — `ask_attribute` is always `None` (independent of Part 1)

Fixing Part 1 will **not** make the agent ask questions.
`_slot_values` ([ask.py:133](starter/ask.py#L133)) reads `prep.product_slots`;
`Preprocessing` never defines it ([preprocessing.py:191-207](starter/preprocessing.py#L191-L207)).
So `_split_quality` returns `0.0` for every attribute and nothing beats `_MIN_SPLIT`.

`data/catalog_normalised.jsonl` is the missing input — its schema matches the vocabulary
[ask.py:112-131](starter/ask.py#L112-L131) documents.

- [ ] Load it in `preprocessing.build()`, key by `parent_asin`, expose as
      `prep.product_slots: dict[str, dict[str, list[str]]]`
- [ ] Path from `TECHJAM_SLOTS`, defaulting to `data/catalog_normalised.jsonl` — mirror
      [retrieve.py:49](starter/retrieve.py#L49)'s `TECHJAM_CATALOG` handling
- [ ] **Missing file must degrade, not raise** — leave `product_slots` empty
- [ ] Drop `audience` and `region` ([ask.py:50-51](starter/ask.py#L50-L51): not API ask attributes)
- [ ] `_slot_values` already handles list / str / comma-separated and filters `"unknown"` — no new parsing needed
- [ ] `state.asked` is written by `note_asked()` but **nothing calls it** — without that, the agent re-asks the same attribute every turn

---

## Part 3 — The inverted index (design done, not built)

### The original question, answered

`multi_retrieval/index.py` **stores nothing**.
[index.py:83](multi_retrieval/index.py#L83) is `sqlite3.connect(":memory:")` — FTS5 lives
in process RAM and dies on exit. No `main()`, no CLI, no write path. There is no build
command and no artifact; it rebuilds every run.

Profiled, 50k products:

| phase | time | share |
| --- | --- | --- |
| `json.loads` | 0.47s | 24% |
| `flatten()` text fields | 0.17s | 9% |
| category postings | 0.16s | 8% |
| **FTS5 create + insert** | **1.14s** | **59%** |
| total | 1.94s | |

The only on-disk artifact in the package is the embedding cache
(`.cache/multi_retrieval/<hash>.npy`, [embed.py:153-165](multi_retrieval/embed.py#L153-L165)),
written only when an embedder is passed — the shipping pipeline passes none, so a normal
run creates nothing.

### `catalog_normalised.jsonl` is not the keyword corpus

Zero field overlap with `TEXT_FIELDS` beyond `parent_asin`/`price` — `feature`/`category`
singular there, `features`/`categories` plural in the index. Indexing it yields a
**silently empty** index:

```
data/catalog.jsonl             category_tokens: 803   bm25('cotton OR shirt'): 50 hits
data/catalog_normalised.jsonl  category_tokens: 0     bm25('cotton OR shirt'):  0 hits
```

### Slot index design — build it as a *second* index, not a replacement

Field coverage in `catalog_normalised.jsonl` (50,000 rows):

| field | coverage | distinct | note |
| --- | --- | --- | --- |
| `category` | 100% | 8 | very coarse |
| `audience` | 100% | 7 | **unused anywhere today — best free win** |
| `brand` | 99.2% | 19747 | high precision |
| `material` | 74.3% | 47 | |
| `use_case` | 55.3% | 14 | |
| `feature` | 52.4% | 25 | |
| `style` | 43.8% | 12 | |
| `color` | 37.1% | 38 | |
| `region` | 33.3% | 99 | not an ask attribute |
| `budget` | 20.8% | 4 | |

**No `size` field** — `Slots` and `starter` both have a `size` attribute with no source.

Filtering power (intersecting the target's own slots, 200 public sessions):

| fields | median pool | ≤10 |
| --- | --- | --- |
| all 9 | **1** | 200/200 |
| without `brand` | **1** | 170/200 |
| `material+color+budget` | 100 | 50/200 |
| `audience` alone | 28117 | 0/200 |

**But the customer does not speak in slot values.**
[local_evaluator.py:52-71](evaluator/local_evaluator.py#L52-L71) builds utterances from
`catalog.jsonl`'s raw `features`/`details` strings *verbatim*. Of 800 phrases the
simulator says, only **221 (27.6%)** appear as a normalised slot value, and the misses are
the discriminative ones:

```
says: 'triple moon pentagram symbol'   slots: [alloy, budget, gift, holiday, jewelry, kids]
```

The slot table holds *what type* a product is; the phrase identifying *which* product
lives only in raw feature text — which is what BM25 indexes. So fuse the two; replacing
BM25 discards the specific 72% of customer vocabulary.

### Planned artifact — `data/inverted_index.jsonl`

19,903 terms, 367,595 entries, **5.8 MB**. Line 1 a manifest (source size+mtime for
staleness detection), then one line per term:

```json
{"field":"material","value":"cotton","df":9126,"ids":["B07K34RX5J","B095PZG4SR"]}
```

- Store `parent_asin`, not row integers — self-contained, immune to catalog reordering
- Store `df`, not `idf` — derivable, one source of truth
- Skip `region` and `price`; index `budget`'s 4 buckets
- Command: `python scripts/build_inverted_index.py` (defaults `--source data/catalog_normalised.jsonl --out data/inverted_index.jsonl`)

- [ ] `scripts/build_inverted_index.py` — new, stdlib only
- [ ] `multi_retrieval/slot_index.py` — loader, verifies manifest against source
- [ ] `multi_retrieval/routes/slot.py` — scores by summed IDF, mirroring [category.py:39](multi_retrieval/routes/category.py#L39)
- [ ] `HardFilter` gains `audience`/`budget` eligibility
- [ ] `types.py` — add a `slot` weight to `TrackConfig`/`DEFAULT_TRACKS`
- [ ] A/B in `score_integrated.py`: BM25 only / slot only / fused

**Only `catalog_normalised.jsonl` feeds this build.** Schema changes are free as long as
`parent_asin` stays the key, one row per product, JSONL. A renamed field is one line in
the build script.

---

## Suggested order

1. **Part 1** — nothing can be measured until the pipeline runs; every `score_integrated.py`
   variant reports the same broken baseline regardless of retrieval quality
2. **Part 2** — cheap, and turns the ask policy on
3. **Part 3** — largest scope, and its A/B is only meaningful after 1 and 2

## Repo hygiene

- `data/catalog_normalised.jsonl` (13.4 MB) is **untracked and not ignored** — decide
  whether to commit it or ignore it; an untracked copy will block a `git pull` that adds
  the same path
- `NOTES.md` untracked
- `integration` has **no upstream**; `origin/integration` does not exist, so a bare
  `git pull` fails with "no tracking information"
