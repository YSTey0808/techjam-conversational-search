# Catalog Reformatter

Reshapes `data/catalog.jsonl` (50,000 Amazon Clothing, Shoes & Jewelry listings)
into a normalised, faceted table.

Eight columns are deterministic rules (~16s, stdlib only). Two — `style` and
`use_case` — come from a single Anthropic Batch API call.

---

## 1. How to run

### Install

```bash
pip install -r requirements.txt      # anthropic, pandas
export ANTHROPIC_API_KEY=...         # only needed for style/use_case
```

The rules stages need nothing installed — `requirements.txt` is only for the
LLM stage and `to_dataframe()`.

### Rules only (no API key, no cost, ~16s)

```bash
python3 -m dataset_normalisation.pipeline run --no-llm
```

Produces all 8 rule columns. `style` and `use_case` stay `[]`.

### Full run, two stages

The batch is the long pole, so `start` submits it **first** and runs the rules
while it cooks. Batches usually finish within the hour; the SLA is 24h.

```bash
python3 -m dataset_normalisation.pipeline start      # submit batch + run rules  (~20s)
python3 -m dataset_normalisation.pipeline status     # check progress (--wait to poll)
python3 -m dataset_normalisation.pipeline finish     # collect labels + merge
```

`start` caches the batch id in `data/normalised/.pipeline_batch`, so `finish` works from a
new shell days later.

### Full run, one blocking command

```bash
python3 -m dataset_normalisation.pipeline run        # submits, polls every 60s, merges
```

### Pilot first

Sanity-check the labels on a small sample before committing the full run.

```bash
python3 -m dataset_normalisation.pipeline start --limit 200
```

### Flags

| flag | effect |
|---|---|
| `--limit N` | process only the first N rows |
| `--out PATH` | output path (default `data/normalised/catalog_normalised.jsonl`) |
| `--no-llm` | skip the LLM stage; `style`/`use_case` stay empty |
| `--with-source` | also emit `*_source` columns (see Output Schema) |
| `--interval N` | poll interval in seconds for `run` / `status --wait` (default 60) |
| `--batch-id ID` | for `status` / `finish`, when not using the cached id |
| `--wait` | for `status`, poll until the batch ends |

`--limit` takes the **first** N rows, not a random sample. The head of the file
is not representative (the first 3,000 rows are ~80% priced vs 21% overall), so
use a pilot to sanity-check labels, not to measure fill rates.

A `--limit` run will not overwrite the full table: if `--out` is left at the
default it writes to `data/normalised/catalog_normalised.limit<N>.jsonl` instead.

### Files written

| path | what | tracked? |
|---|---|---|
| `data/normalised/catalog_normalised.jsonl` | the output table | gitignored |
| `data/normalised/labels.jsonl` | cached LLM labels | gitignored |
| `data/normalised/.pipeline_batch` | batch id for `finish` | gitignored |

Re-runs read `data/normalised/labels.jsonl` and **submit only rows it does not already
cover**. So a 200-row pilot followed by a full run labels the remaining 49,800
and never re-submits the same `parent_asin`. That cache is also what makes the
output reproducible — the rules are deterministic, the LLM is not.

**Keep `data/normalised/labels.jsonl`.** It is the only artifact here that cannot be
regenerated for free; with it, a full rebuild is ~16s.

### In Python

```python
from dataset_normalisation.pipeline import load_catalog, run_rules, to_dataframe
df = to_dataframe(run_rules(load_catalog()))     # needs pandas
```

### Degradation

Without `anthropic` or `ANTHROPIC_API_KEY`, the pipeline warns, leaves
`style`/`use_case` empty, and still produces all 8 rule columns. Any labels
already in the cache are still merged.

---

## 2. Input Schema

`data/catalog.jsonl` — one JSON object per line, 50,000 rows. Every field is
always present; the percentage is how often it is **non-empty**.

| field | type | non-empty | notes |
|---|---|---|---|
| `parent_asin` | `str` | 100.0% | primary key. A **variant group**, not a single SKU |
| `title` | `str` | 100.0% | richest single signal; drives `color`, `category` |
| `features` | `list[str]` | 89.6% | bullet points; drives `material`, `feature` |
| `description` | `list[str]` | 52.2% | unused — adds ~2-3pp over title+features |
| `price` | `float \| str \| None` | 21.1% | mostly null. Junk forms: `"—"`, `"from 12.99"` |
| `categories` | `list[str]` | 100.0% | hierarchical path, 2-8 levels; level 0 is the root |
| `details` | `dict[str, str]` | 96.7% | free-form k/v; keys vary wildly |
| `average_rating` | `float` | 100.0% | unused |
| `rating_number` | `int` | 100.0% | unused |
| `store` | `str \| None` | 99.4% | the usable brand field |

Notable `details` keys: `Department` (87%), `Color` (4.9%), `Material` (4.1%),
`Country of Origin` (1.7%), `Size` (1.8%).

> **`parent_asin` is a variant group.** The title describes the group while
> `details.Color` names one child SKU. This caps `color` fill near 35% and is
> why `size` is not an output column — neither has a single correct value at
> this grain.

---

## 3. Output Schema

`data/normalised/catalog_normalised.jsonl` — one JSON object per input row, same order.
Fill rates measured on the full 50,000.

| column | type | fill | source |
|---|---|---|---|
| `parent_asin` | `str` | 100.0% | direct copy |
| `category` | `str` (enum) | 100.0% | `categories` + `title` |
| `audience` | `str` (enum) | 100.0% | `details.Department` + `categories` |
| `brand` | `str` | 99.2% | `store`, lowercased |
| `material` | `list[str]` (enum) | 74.3% | `features` |
| `use_case` | `list[str]` (enum) | 55.3% | `title` + `features` (LLM) |
| `feature` | `list[str]` (enum) | 52.4% | `features` |
| `style` | `list[str]` (enum) | 43.8% | `title` + `features` (LLM) |
| `color` | `list[str]` (enum) | 37.1% | `details.Color` + `title` |
| `region` | `str` | 33.3% | `details` + `features` |
| `price` | `float \| None` | 20.8% | `price`, parsed |

### Empty conventions

| type | means "not stated" |
|---|---|
| `list[str]` | `[]` |
| `region` | `"unknown"` |
| `price` | `null` |

**Empty is a correct answer, not a gap.** `color` is ~63% empty because most
listings never state one. If `style`/`use_case` come back near-100% filled, the
model is inventing labels.

### `--with-source` columns

Adds four columns recording which rule fired, so you can filter to the
high-precision tier (e.g. `color` from `details` is ~95% right, from `title`
~88%):

| column | values |
|---|---|
| `category_source` | `path` · `title` · `none` |
| `color_source` | `details` · `title` · `none` |
| `audience_source` | `department` · `categories` · `default` |
| `region_source` | `details` · `made_in` · `usa+imported` · `imported` · `usa` · `none` |

`category.py` also defines a `prefix` branch, but it never fires: every key in
its `PREFIX` map is also in `NODE`, so step 1 always resolves first. Dead code,
harmless, not worth a migration.

### Example record

```json
{
  "parent_asin": "B097NZPX2Y",
  "category": "Clothing",
  "audience": "women",
  "brand": "bcbgeneration",
  "material": ["rayon", "spandex"],
  "color": [],
  "feature": ["machine-washable", "stretch"],
  "style": ["casual"],
  "use_case": [],
  "price": null,
  "region": "imported"
}
```

> Output is JSONL, not CSV. Five columns are `list[str]`, which CSV cannot
> round-trip without stringifying to `"['black', 'silver']"`.

---

## 4. Enums

### `category` — 8
```
Clothing · Shoes · Jewelry · Watches · Bags & Luggage · Accessories · Costumes · Other
```

### `audience` — 7
```
women · men · girls · boys · baby · kids · unisex
```
`unisex` is both a real value and the fallback for audience-less products
(luggage tags, shoe trees). `audience_source == "default"` distinguishes them.

### `region` — semi-open
```
local · imported · unknown · <country name>
```
`local` is anchored to the US. Amazon's boilerplate *"Made in USA and Imported"*
fires both signals on ~2% of rows; those resolve to `imported` with
`region_source == "usa+imported"`. 42 distinct values seen — mostly
`China`, `Italy`, `India`, `Thailand`, `Vietnam`.

### `color` — 38
```
black white gray ivory cream red blue green yellow orange purple pink brown
navy burgundy teal olive khaki beige tan coral mint peach mustard salmon rust
indigo champagne nude clear multicolor
rose gold · gold · silver · bronze · copper · platinum · pewter
```
Metallics are kept as colours deliberately — a sterling silver bracelet **is**
silver-coloured, and `details.Color` agrees. This is the inverse of
`material.py`, which rejects bare `gold`/`silver` *because* they read as colour.

### `feature` — 25
```
machine-washable stretch lightweight pockets breathable adjustable padded
quick-dry non-slip waterproof water-resistant hypoallergenic handmade
uv-protection memory-foam recycled arch-support insulated windproof packable
wrinkle-free rfid-blocking anti-odor reversible tarnish-resistant
```
Capabilities only. `material` owns composition, `style` owns look, `use_case`
owns occasion.

### `material` — 47
Fibres, leathers, fabrics, plastics, metals, trim. See
`dataset_normalisation/columns/material.py` → `VOCAB`.

### `style` — 12
```
casual formal business elegant vintage athletic
boho minimalist classic streetwear romantic western
```

### `use_case` — 14
```
everyday work formal-event party wedding sport outdoor
beach travel gift holiday school lounge costume
```

> `style` and `use_case` enums are enforced in the JSON schema sent to the API
> (`"enum": [...]` on the array items), not merely requested in the prompt.
> Without that, open-vocabulary extraction reproduces `details.Style`, which
> holds 283 distinct values across 466 rows — half of them not styles.

---

---

## 5. Data quality

### Empty is the honest answer, not a gap

Most columns are far from 100% filled because most listings simply do not state
the attribute. The caps are properties of the source data, not the rules.

| column | empty | why it caps there |
|---|---|---|
| `price` | 79.2% | `price` is null on ~79% of rows. Absent, not hidden — no method recovers it. |
| `region` | 66.7% | most listings never state origin |
| `color` | 62.9% | `parent_asin` is a **variant group**: the title describes the group, `details.Color` names one child SKU. Neither has a single correct answer. |
| `style` | 56.2% | most listings state no aesthetic |
| `feature` | 47.6% | stated capabilities only |
| `use_case` | 44.7% | most listings state no occasion |
| `material` | 25.7% | composition is usually stated in `features[]` |

A near-100% fill on `style` or `use_case` would mean the model was inventing
labels. These rates are the target, not a shortfall.

### Multi-value shape

| column | mean per filled row | 2+ values |
|---|---|---|
| `feature` | 1.91 | — |
| `material` | 1.76 | — |
| `use_case` | 1.61 | 44% |
| `color` | 1.26 | 21% |
| `style` | 1.11 | 11% |

### What was validated, and what was not

`style` and `use_case` were checked on a 200-row sample against the source text:

- Every row labelled `gift` mentions a gift in the listing.
- Every row labelled `everyday` uses everyday/daily/casual wording. An earlier
  prompt failed this on 25% of such rows and was rewritten.
- Caps of 2 (`style`) and 3 (`use_case`) are enforced in code after collection,
  because the API rejects `maxItems` in `output_config` schemas.

Not validated: the rule-based columns have no independent ground truth beyond
`details.Color` (which covers 4.9% of rows) and spot checks. `category` is
resolved by category path on ~96% of rows with no way to verify that branch.

### Known quirks

- **`region` mixes three value spaces** — `local` / `imported` / `unknown` and
  bare country names. Split it if you need a clean facet.
- **`category.py` has a dead `prefix` branch.** Every key in its `PREFIX` map is
  also in `NODE`, so step 1 always wins. Harmless.
- **115 rows (0.23%)** had their batch request fail and were labelled by hand
  against the same rules. They need no special handling.
- **Metallics are colours here.** `gold`/`silver` in `color` are deliberate — a
  sterling silver bracelet is silver-coloured, and `details.Color` agrees. This
  is the inverse of `material.py`, which rejects bare `gold`/`silver`.

---

## Module layout

```
dataset_normalisation/
  pipeline.py            entry point — orchestration only
  columns/
    audience.py          audience()
    price.py             parse_price()       -> float | None
    category.py          product_family()
    color.py             extract_colors()
    feature.py           extract_features()
    material.py          extract_materials()
    region.py            region()
    style_use_case.py    build_request()     -> Batch API request
```
