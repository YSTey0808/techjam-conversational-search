# Embedder

Turns the 50,000 catalog products into vectors, so you can search by meaning
instead of by keyword.

Built with `nomic-embed-text-v1.5`, running locally on your machine. No API key,
no cost.

---

## 1. What you have

Three files in `data/embeddings/`. They only work together.

| file | size | what it is |
|---|---|---|
| `v2_nomic.npy` | 146 MB | the vectors: 50,000 rows x 768 numbers |
| `v2_nomic_ids.json` | 0.67 MB | which product each row belongs to |
| `v2_nomic_meta.json` | tiny | which model and settings made it |

**The product ID is not inside the `.npy` file.** The link is the row position:

```
row 0  ->  ids[0]  ->  "B07K34RX5J"
row 1  ->  ids[1]  ->  "B07KCFS4VC"
```

Row order is the same as `data/catalog.jsonl`, line for line.

> These files are gitignored. They live on your machine only. See section 5 to
> rebuild them.

---

## 2. How to use it

### Load

```python
import numpy as np, json

vectors = np.load("data/embeddings/v2_nomic.npy")            # (50000, 768)
ids     = json.load(open("data/embeddings/v2_nomic_ids.json"))
```

### Search

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                            trust_remote_code=True, device="mps")

query = model.encode(["search_query: warm waterproof winter boots"],
                     normalize_embeddings=True)[0]

scores = vectors @ query                  # one number per product
top10  = np.argsort(-scores)[:10]

for i in top10:
    print(round(float(scores[i]), 3), ids[i])
```

That is the whole search. The vectors are already normalised, so a dot product
**is** the cosine similarity. No database needed — 50,000 products scores in
about 10-20 ms.

Or use the built-in command:

```bash
python3 -m embedder.embed search --variant v2 "warm waterproof winter boots"
```

### The one thing you must not get wrong

`nomic` is trained with two different prefixes:

| what you are encoding | prefix |
|---|---|
| a product | `search_document: ` |
| a user's query | `search_query: ` |

The products already have theirs baked in. **You must add `search_query: ` to
every query.** If you forget it, or use the wrong one, nothing errors — the
results just quietly get worse. This is the easiest mistake to make here.

---

## 3. What text goes into a product vector

Defined in `variants.py`, joined with ` | `:

```
title
categories path (minus the constant "Clothing, Shoes & Jewelry" root)
the 10 normalised facets (category, audience, brand, material, color,
                          feature, style, use_case, region, price)
details: Color, Material, Department, Style
all feature bullets
description
```

Real example:

```
Hanes Men's Underwear Briefs Pack, Mid-Rise, Moisture-Wicking, 6-Pack |
Men > Clothing > Underwear > Briefs |
Clothing men hanes cotton polyester stretch lightweight quick-dry imported |
Department: Mens |
Solids: 100% Cotton; heathers: 75% cotton, 25% polyester ...
```

Average 208 tokens per product. Nothing is cut off — the window is 2048 tokens
and the longest product is 1731.

Deliberately left out: `average_rating`, `rating_number`, and the noisy
`details` keys (model numbers, package dimensions, dates). Ratings are useful
for ranking results, but they are numbers, not meaning — use them after the
search, not inside the vector.

---

## 4. Why this setup

Two things were tested and rejected. The code for both has been removed; the
measurements are recorded here.

### V2 (full text) beat V1 (facets only)

Scored on 200 eval sessions, turn-1 message only:

| | hit@1 | hit@10 | MRR | hit@100 | median rank |
|---|---|---|---|---|---|
| V1 — facets only | 0.010 | 0.060 | 0.023 | 0.220 | 507 |
| **V2 — full text** | **0.105** | **0.295** | **0.157** | **0.755** | **23** |

V1 scored **zero** on all 30 intent_override sessions. Stripped to enum tokens
like `Clothing women cotton casual`, products lose the wording a user actually
searches with.

V1 is still useful — as a metadata filter (`category`, `audience`, `brand`), not
as the search index.

### nomic beat a 4x bigger model

| | hit@10 | MRR | hit@100 | median rank |
|---|---|---|---|---|
| **nomic** (137M) | **0.295** | **0.157** | **0.755** | **23** |
| arctic-embed-l-v2.0 (568M) | 0.290 | 0.144 | 0.720 | 37 |

A paired test says these are a **tie** (p = 1.000). We kept nomic because at
equal accuracy the smaller model is cheaper: 4x fewer parameters, 768-dim
vectors instead of 1024, faster to build.

`Alibaba-NLP/gte-large-en-v1.5` was also tried and does not work — its custom
model code crashes during encoding, on both GPU and CPU.

### Where the remaining gain is

hit@100 is 0.755 but hit@10 is only 0.295. **The right product is usually found,
it is just not ranked near the top.** A bigger embedding model does not fix
that — a reranker over the top 100 does. That is the next thing to build, not
another encoder.

> Those numbers came from a 12,648-product test corpus. On the full 50,000 they
> will be lower, because a 4x bigger haystack is a harder problem.

---

## 5. Rebuilding

### Setup (once)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-embed.txt
```

Downloads ~2 GB (torch), plus ~550 MB of model weights on first run.

### Build

```bash
python3 -m embedder.build_embedding_sample --rate 1   # list all 50,000 products
python3 -m embedder.embed build --variant v2          # encode them
```

**About 42 minutes** on an Apple M4.

`--rate 1` means every product. `--rate 4` gives you a 25% sample instead, which
builds in about 10 minutes.

> A sample is only useful for build-speed testing, not for scoring. At `--rate 4`
> just 52 of the 200 eval answers land in the sample, so `compare` numbers become
> meaningless. Score on the full catalog.

### If it stops partway

Run the same `build` command again. It saves progress every 2,048 rows and picks
up where it left off.

> The resume counter lives in `v2_nomic.progress`. If you delete that file, a
> rerun starts from zero and re-encodes everything.

### Run it in the background

A build dies when its terminal closes. Detach it:

```bash
nohup python3 -m embedder.embed build --variant v2 > build.log 2>&1 &
```

Check on it with `tail -2 build.log`.

### Rebuild when

- the catalog changes
- `catalog_normalised.jsonl` changes (facets are inside the vectors)
- you edit `variants.py`

---

## 6. Scoring a change

```bash
python3 -m embedder.compare --variants v2
```

Replays the evaluator's own turn-1 message for all 200 eval sessions and reports
hit@1, hit@10, MRR, hit@100 and median rank, broken down by scenario. No LLM, no
cost, fully deterministic — so any difference you see is your change, not noise.

Ignore the `difficulty_bucket` breakdown. It is the same split as
`scenario_type` under different labels: easy = buying, hard = intent_override,
medium = browsing + boundary.

---

## 7. Files

```
embedder/
  build_embedding_sample.py   pick which products to encode
  variants.py                 the text a product turns into
  embed.py                    build / search, and the model registry
  compare.py                  turn-1 retrieval scores
```

To add a model, add an entry to `MODELS` in `embed.py`. **Read its prefixes off
the model's own `config_sentence_transformers.json`** rather than guessing —
they differ per model (arctic uses `query: ` and no document prefix), and a wrong
prefix fails silently.
