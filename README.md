# Conversational E-Commerce Search Agent

A shopping agent for the TechJam Conversational E-Commerce Search Challenge. The
customer types plain English — *"I'm looking for Jewelry Necklaces. A key
requirement is: Material:alloy."* — and each turn the agent returns up to ten
products from a frozen 50,000-item Amazon Clothing, Shoes & Jewelry catalog plus
one follow-up question. The session ends when the hidden target appears in the
top ten, or after ten turns.

**Result on the 200 public sessions: TechnicalScore 0.8532**, against the weak
BM25 starter's 0.1067.

| | Hit Rate@10 | MRR | MTTC | Efficiency | **TechnicalScore** |
|---|---|---|---|---|---|
| Weak BM25 starter | 0.125 | 0.0680 | 9.81 | 0.119 | 0.1067 |
| **This agent** | **0.985** | **0.6418** | **2.595** | **0.8405** | **0.8532** |

Per scenario: browsing 1.000 hit@10, boundary 1.000, buying 0.975,
intent_override 0.967. The full run takes **13.8 seconds** and reports **zero
tokens** — the measured configuration makes no model calls at all (see
[Model choice, cost and latency](#model-choice-cost-and-latency)).

---

## 1. Project overview

### The central problem

A shopper and a product listing do not use the same words. The shopper says
*"waterproof"*; the catalog says *"Water Resistant"*. They say *"stretchy"*; the
catalog says *"Spandex"*. So the whole pipeline is built around one job: **turn
customer words into catalog words, then use the catalog's own structure to
narrow down.**

Two things follow from that, and they are the two ideas the system is really
built on:

1. **The raw catalog is not a searchable structure, so we built one.** 50,000
   listings of free-text bullets were reshaped into a faceted table — one row
   per product, ten controlled-vocabulary columns (`category`, `audience`,
   `brand`, `material`, `color`, `feature`, `style`, `use_case`, `region`,
   `price`). Eight columns come from deterministic rules; `style` and `use_case`
   come from a single batched LLM pass. That table is what lets the agent ask
   *"how many distinct colours are in this candidate pool?"* — a question raw
   feature bullets cannot answer.
2. **A good question is how you obtain the next clue.** The evaluator's customer
   only discloses a new requirement when the agent asks about the right
   attribute. So the follow-up question is chosen by information gain, not by
   what sounds natural.

### One turn

`starter/agent.py` is wiring and nothing else — no branching, no fallbacks.
Anything that needs a decision belongs to the module that owns it.

```
respond(session_id, user_message, turn, top_k)
        |
        v
  extract.extract(message, turn, state)      A   English  ->  SessionState
        |                                            (regex frames, then LLM)
        v
  retrieve.retrieve(prep, state)             C   ->  ~200 candidates
        |                                            (BM25 + category, fused)
        v
  ask.decide(prep, state, pool, turn)        E   ->  question + list width
        |                                            (entropy over the pool)
        v
  adapter.rerank(prep, state, pool, width)   D   ->  the final ten
        |
        v
  {"message", "ask_attribute", "recommendations", "usage"}
```

**Stage A — `starter/extract.py`.** Reads one message into the session state.
`SessionState` is the only memory: a reading that never reaches a slot never
happened. A deterministic router matches the message against known shapes and
writes typed slots with a confidence score; a *degradation gate* decides whether
that parse can be trusted, and anything it rejects falls through to an LLM that
returns strict schema-validated JSON. `extract()` never raises — an exception
would lose the whole turn, recommendations included.

**Stage C — `starter/retrieve.py` + `multi_retrieval/`.** Builds a candidate
pool of ~200. A hard filter decides who is eligible (only clean, dense facets:
`brand`, `audience`, verbatim `category`, and numeric limits), then independent
routes search the whole catalog and are fused by reciprocal rank:

- **keyword** — BM25 over a SQLite FTS5 index, title weighted 6.0 down to
  description 1.0. The primary route.
- **category** — IDF-weighted overlap on the category path. Adds weight, never
  removes a candidate.
- **vector** — cosine similarity against precomputed `nomic-embed-text-v1.5`
  vectors. Loads only when `sentence-transformers` is installed and the vectors
  match the catalog; otherwise the two lexical routes carry the query.

Three rules govern the filter, and each exists because the obvious alternative
loses sessions: **a filter that would empty the set is skipped, not enforced**;
**unknown is not a violation** (only 10,415 of 50,000 products carry a price, so
a missing price survives every budget); and **a filter matching most of the
catalog is not a filter**. It never returns an empty list — an empty list is a
guaranteed miss, a mediocre one costs at most a turn.

**Stage E — `starter/ask.py`.** Picks the attribute whose answer would tell us
most, scored as normalised Shannon entropy over *known* values × the fraction of
the pool where the attribute is known — a C4.5-style missing-value discount, so
a sparse facet is useful but discounted rather than silently rewarded. `brand`
and `category` are deliberately never asked: the evaluator's classifier can
never return either label, so the question is a spent turn. When nothing splits
the pool it asks `other`, which is answerable, instead of asking nothing.

**Stage D — `reranker/`.** Stage B is pure stdlib: it fuses retrieval rank,
Bayesian-smoothed store rating and user-profile preference match by RRF, and
promotes candidates that match every stated constraint. Stage C is an optional
LLM rerank over the shrunk pool, which falls back to the Stage B order on any
failure.

### Data preparation

Two offline pipelines produce inputs the agent reads at startup.

- **`dataset_normalisation/`** builds `data/catalog_normalised.jsonl` — the
  faceted table described above. Both the retrieval filter and `ask.py`'s
  entropy calculation read it.
- **`embedder/`** builds `data/embeddings/v2_nomic.npy` — 50,000 × 768
  L2-normalised float32 vectors, row-aligned with the catalog, for the vector
  route.

Full detail: [docs/REFORMATTER_README.md](docs/REFORMATTER_README.md) and
[embedder/EMBEDDER_README.md](embedder/EMBEDDER_README.md).

### Layout

```
starter/       extract.py  retrieve.py  ask.py  agent.py  preprocessing.py  schema.py
multi_retrieval/   the retrieval engine: index, filters, routes, fusion, prior
reranker/      Stage B shrink + Stage C LLM rerank
llm_client.py  provider-agnostic LLM interface (anthropic | groq | mock | none)
dataset_normalisation/   catalog -> faceted table
embedder/      catalog -> nomic vectors
evaluator/     the organizer's local evaluator (unmodified scoring)
scripts/       evaluation and debugging harnesses
tests/         119 unit tests, stdlib unittest
```

---

## 2. Setup and installation

Python 3.10 or later (developed on 3.14).

### Install

```bash
pip install -r requirements.txt
```

That is `pydantic` (schema validation of LLM output), `python-dotenv` (loads
`.env`) and `anthropic`. All three degrade: with none of them installed the
agent still runs, it just cannot make model calls.

Two optional extras, neither needed to reproduce the headline score:

```bash
pip install -r requirements-multi-retrieval.txt   # numpy, sentence-transformers, einops
pip install -r requirements-embed.txt             # torch + sentence-transformers, to rebuild vectors
```

### Get the catalog

`data/catalog.jsonl` is a 58 MB GitHub Release download and is deliberately not
in git.

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify it against the published `SHA256SUMS`. Expected row count: 50,000.

### Build the derived data

`data/catalog_normalised.jsonl` is required — `ask.py` scores 0.0 on every
attribute without it, and the retrieval filter loses its facets.

```bash
# rules only: no API key, no cost, ~16s, leaves style/use_case empty
python3 -m dataset_normalisation.pipeline run --no-llm

# or the full table, including the batched LLM pass for style/use_case
export ANTHROPIC_API_KEY=...
python3 -m dataset_normalisation.pipeline run

cp data/normalised/catalog_normalised.jsonl data/catalog_normalised.jsonl
```

The embeddings are optional. Rebuilding them takes about 42 minutes on an Apple
M4 and needs no API key:

```bash
python3 -m embedder.build_embedding_sample --rate 1
python3 -m embedder.embed build --variant v2
```

### Configure a model (optional)

```bash
cp .env.example .env    # then fill in a key
```

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | selects the Anthropic provider; extraction and Stage C use `claude-sonnet-5` |
| `GROQ_API_KEY` | selects Groq (free tier); uses `openai/gpt-oss-120b` |
| `TECHJAM_LLM_PROVIDER` | forces `anthropic`, `groq`, `mock` or `none` |
| `TECHJAM_LLM_MODEL` | model override; ignored when it does not match the chosen provider |
| `RERANKER_USE_LLM=0` | forces Stage B order even with a key present |
| `TECHJAM_CATALOG` / `TECHJAM_CATALOG_NORMALISED` / `TECHJAM_SLOTS` / `TECHJAM_EMBEDDINGS` | override data paths |

**With no key configured the agent runs its full deterministic path** — no
network, no per-turn latency, no cost. That is the configuration the score above
was measured in. Never commit a real key; `.env` is gitignored.

---

## 3. Steps to reproduce our results

From a clean checkout with `data/catalog.jsonl` and
`data/catalog_normalised.jsonl` in place, and **no API key set**:

```bash
python3 -m evaluator.local_evaluator
```

This writes per-session results and aggregate metrics to `results.json` and
prints the summary. It reproduces exactly the table at the top of this file:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.985,
  "mrr": 0.641835,
  "mttc": 2.595,
  "efficiency": 0.8405,
  "recommended_technical_score": 0.853151,
  "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

The run is fully deterministic — same catalog, same numbers, every time. It took
13.8 s wall clock on an Apple M4, of which 5.2 s is one-off indexing at startup:
`preprocessing.build()` takes 3.6 s and `multi_retrieval`'s FTS5 index over the
same 50,000 products takes 1.6 s. Neither is ever rebuilt per turn.

Useful variations:

```bash
# explicit paths, when running against a different catalog or dataset
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl \
                                     --dataset data/public_set.jsonl \
                                     --output results.json

python3 -m scripts.run_local_evaluator --config myrun  # same run + a compact report under evaluation_reports/
python3 -m unittest discover tests                     # 119 tests, ~0.1s
```

To watch conversations against an LLM-driven customer instead of the templated
one, set `GROQ_API_KEY` and run `python3 -m scripts.local_chat_simulator`. It
plays five real catalog products through the agent, prints every turn and saves
a transcript under `chat_transcripts/`.

Retrieval alone, isolated from extraction:

```bash
python3 scripts/score_multi_retrieval.py
python3 scripts/score_multi_retrieval.py --fusion additive --no-layered --vector hashing
```

### Model choice, cost and latency

| | |
|---|---|
| Models used in the scored run | **none** |
| Reported token usage | 0 prompt, 0 completion |
| Cost of the scored run | $0.00 |
| Latency | 13.8 s for 200 sessions, including a one-off 5.2 s index build ≈ **43 ms per session** thereafter, no network |
| Models available when configured | `claude-sonnet-5` (Anthropic) or `openai/gpt-oss-120b` (Groq) for extraction fallback and Stage C rerank |
| One-off LLM cost in data prep | a single Anthropic Batch API call labelling `style` and `use_case` for 50,000 products |

The agent is designed so every model call is a *fallback*, not a dependency: the
LLM extraction path runs only when the deterministic parse is not trustworthy,
and the Stage C rerank runs only when explicitly enabled.

---

## 4. Limitations, and what we would improve

We would rather state these plainly than have a judge find them.

**1. The headline score is measured against a simulator whose sentences we parse
exactly — and that is the number's biggest weakness.** `extract._route()`
matches nine fixed message shapes produced by `evaluator/local_evaluator.py`
(*"I'm looking for X. A key requirement is: Y."*, *"For that, what matters is:
…"*, and so on). Against those, extraction is essentially perfect and free. A
real shopper writing *"need something warm for hiking, nothing over fifty
bucks"* matches no template, so the turn falls through to the LLM path — which
needs an API key that the scored run does not have. **We believe 0.8532 would
not survive contact with paraphrased input**, and we deliberately built the
degradation gate (`_frame_is_trustworthy`) so that a paraphrase is routed to the
LLM rather than silently mis-parsed. What is missing is the measurement: a
held-out paraphrase set, and the same score reported on it. That is the first
thing we would build with more time.

**2. The Stage C LLM reranker ships but never runs.** `starter/agent.py:45`
calls `adapter.rerank(..., False)`, hardwiring the LLM step off. The code, the
prompt, the fallback and its tests are all in place; it has simply never been
turned on inside the agent and measured. It is a one-line change plus a scored
run, and it is the cheapest untested upside in the repo.

**3. The vector route is off in practice.** `data/embeddings/v2_nomic.npy` is
built (146 MB, 50,000 × 768) and `multi_retrieval` will adopt it, but only if
`sentence-transformers` is importable in the interpreter running the agent. In
our environment it is not, so `diagnostics["vector_route"]` reads `null` and the
scored run used two lexical routes. Separately, the embedder's own measurements
say semantic search is not obviously the right tool here: hit@100 is 0.755 while
hit@10 is 0.295, i.e. the right product is usually *found* but not ranked near
the top — which argues for a reranker over the top 100, not a bigger encoder.

**4. `scripts/score_integrated.py` is broken.** It still imports `starter.state`
and the `Constraint`/`Extraction` types, all removed in the state-schema
migration, so it raises `ImportError` on launch. It was the harness that
compared retrieval variants inside the real pipeline; losing it means retrieval
changes are currently only measurable end-to-end.

**5. The design docs have drifted.** `docs/ARCHITECTURE.md`,
`docs/OWNERSHIP.md` and `docs/PIPELINE.md` describe a five-stage pipeline with
`starter/state.py` and `starter/rank.py`. Both files were deleted — state
handling moved into `extract.py` and ranking into `reranker/`. The reasoning in
those documents is still sound; the file names and call chains are not.
`docs/RETRIEVAL.md` is current.

**6. Constants are hand-set, not swept.** Route weights, the RRF constant, the
popularity prior weight (0.2), `_MIN_SPLIT` (0.05) and the reranker's Stage B
weights were chosen by judgement and spot checks. `docs/PIPELINE.md` lists every
one with its location precisely so they can be swept properly; nobody has done
the sweep.

**7. Known dead ends in the attribute vocabulary.** There is no `size` column in
the normalised catalog — `parent_asin` is a variant group, so neither size nor
colour has a single correct value at that grain — so `size` scores 0.0 and can
never be picked as a question. `brand` is excluded for the opposite reason: it
would win on entropy every turn (19,747 distinct values) but the evaluator's
customer can never answer it.

**8. The remaining points are in a ranking tail, not in retrieval.** Of 200
sessions we hit 197, and the median hit is at **rank 1** — 102 sessions land the
target first, and 138 land it in the top three. The MRR shortfall is a tail: 13
sessions scrape in at rank 10 and 26 more sit between ranks 6 and 9, each
contributing almost nothing. So the question is not *can we find it* but *why
does a correct candidate sometimes sit at rank 10*, which is Stage D's problem —
and Stage D's LLM step is exactly the part that never runs (limitation 2).

**9. The 200 public sessions are a small, self-referential sample.** The
customer quotes the target product's own catalog strings back at us, which is
precisely what the index is built to match. Combined with limitation 1, the
honest reading is that this score measures the pipeline against a cooperative,
templated counterpart — not against a shopper.

**With more time, in priority order:** measure on paraphrased input and fix
whatever that exposes; turn Stage C on and score it; get the vector route
actually loading in the agent's interpreter; repair `score_integrated.py`; sweep
the constants; bring the docs back in line with the code.

---

## 5. Team member contributions

| Member | Contribution |
|---|---|
| **YSTey0808** | Pipeline integration and wiring (`starter/agent.py`), `preprocessing.py` and the normalised slot table it loads, the inverted-index builder, and the end-to-end documentation (`docs/PIPELINE.md`, `HANDOVER.md`). Merged the four feature branches into a working pipeline. |
| **Zi Yong** | The `multi_retrieval/` retrieval engine end to end — FTS5 index, facet hard filter, keyword/category/vector routes, RRF fusion, popularity prior — plus `llm_client.py`, the `reranker/` Stage B and Stage C implementation, and `docs/RETRIEVAL.md`. |
| **Tan Jay** | Stage A: `starter/extract.py` and the `SessionState` schema (`starter/schema.py`) — the frame router, confidence model, dependency-cascade logic, LLM extraction path, and the regex-first degradation gate. |
| **yijiechong13** | Stage E: the entropy-based ask-attribute policy in `starter/ask.py`, plus the local evaluation tooling — `scripts/local_chat_simulator.py` and `scripts/run_local_evaluator.py`. |
| **fabian** | The data layer: the `dataset_normalisation/` pipeline producing `catalog_normalised.jsonl` (rule columns, LLM batch labelling, attribute-value extraction) and the `embedder/` package producing the nomic catalog vectors, including the V1-vs-V2 and nomic-vs-arctic model comparisons. |

`evaluator/`, `docs/competition_specification.md`, `docs/agent_api_contract.json`
and `starter/baseline.py` are the organizer's, unmodified except for optional
per-turn logging.

---

## Further reading

- [docs/PIPELINE.md](docs/PIPELINE.md) — one customer message traced end to end, file by file
- [docs/RETRIEVAL.md](docs/RETRIEVAL.md) — the retrieval stage, its measurement history, and everything tried and cut
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the design rationale (see limitation 5 on drift)
- [docs/REFORMATTER_README.md](docs/REFORMATTER_README.md) — the catalog normalisation pipeline
- [embedder/EMBEDDER_README.md](embedder/EMBEDDER_README.md) — the embedding build
- [reranker/SCHEMA.md](reranker/SCHEMA.md) — the reranker request/response schema
- [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) — data source and redistribution terms

The catalog and sessions derive from Amazon Reviews 2023 by McAuley Lab, UCSD.
