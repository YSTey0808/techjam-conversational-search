# Catalog normalisation notes

Working notes for reshaping `data/catalog.jsonl` (50,000 rows) into a normalised table.

## Sampling caveat
Every number below comes from a **12,002-row sample (~24%)**, drawn as 4 contiguous
blocks (lines 1-3000, 12000-15000, 28000-31000, last 3000) — not a random sample.
Re-verify on the full file before trusting anything to 0.1pp.

## What the raw data is
Single domain: Amazon **Clothing, Shoes & Jewelry**. No electronics/home/etc.

Fields per row: `parent_asin`, `title`, `features[]`, `description[]`, `price` (often null),
`categories[]` (hierarchical path, 2-8 levels), `details{}` (free-form k/v),
`average_rating`, `rating_number`, `store`.

The `categories` taxonomy is dirty: 561 distinct paths per 1,800 rows, inconsistent depth,
and store/campaign names leak into level 2 (`Westlake`, `Boot Shop`, `Expert Beauty Trends`,
`Deal of the Day: ...`).

## Decisions made

| target column | type | source | how |
|---|---|---|---|
| `product_family` | enum (8) | `categories` + `title` | rules — `starter/product_family.py` |
| `audience` | enum (7) | `categories` + `details.Department` | rules (not yet written) |
| `brand` | str | `store` | direct copy |
| `material` | `list[str]` | `features[]` | rules — `starter/material.py` |

Rules over LLM throughout. It's a one-time offline job, rules already reach 99%+ coverage,
and they're deterministic and free. LLM reserved for the residual tail only.

### product_family — `starter/product_family.py`
`product_family(row) -> (family, source)`. Enum:
`Clothing · Shoes · Jewelry · Watches · Bags & Luggage · Accessories · Costumes · Other`

Resolution order: (1) category path walked **deepest→shallowest** so a specific leaf beats a
vague ancestor; (2) level-2 bucket fallbacks (`Boot Shop`→Shoes, `Sport Specific Clothing`→
Clothing, `Luggage & Travel Gear`→Bags, `Costumes & Accessories`→Costumes); (3) title keywords,
**last match wins** with trailing `with …` stripped, so "A-Line Dress with Belt" is a dress;
(4) `Other`.

Measured: 95.6% resolved by path, 3.9% by title, **0.5% Other**.
Distribution: Clothing 45.4 / Shoes 25.9 / Jewelry 10.3 / Accessories 8.4 /
Bags 5.3 / Watches 2.4 / Costumes 1.7 / Other 0.5.

Title-rule accuracy **95.4%**, measured against path-labelled rows as ground truth
(last-match beat first-match-wins, 94.3%). Since title decides only 3.9% of rows that's
~0.2% corpus-wide error. The path branch is **unverified** — no independent ground truth.

### material — `starter/material.py`
`extract_materials(row) -> list[str]`. Reads `features[]` only (a `use_title` option was
built then removed on request). Empty list when nothing is stated — decision: park every
mentioned material in one flat list, no primary/secondary split, no percentages stored.

Closed vocabulary of 47 canonical materials, mined from the actual `N% X` strings.
**Closed on purpose** — an open percentage capture picks up `100% satisfaction guaranteed`,
`30% off`, `100% brand new`.

Key rules: longest-match-first (`faux leather` > `leather`, `sterling silver` > `silver`);
`synthetic` dropped when anything specific is found; **bare `gold`/`silver` rejected** as
colour — they need a qualifier (`14k gold`, `gold plated`, `925 silver`). Ordering is by
prominence: percentage-bearing materials first (desc), then source order — so `material[0]`
is the dominant one without a second column.

Measured: **75.3% filled**, 24.7% empty. Mean 1.8 materials/row. All 47 terms used.
Spot-checked 15 random rows traced to source text — 15/15 correct.

Known + accepted: component materials land in the same list, so a shoe with "Rubber sole"
and "Leather Lining" returns `['rubber', 'leather']`.

### audience (decided, not yet implemented)
7 groups: `women · men · girls · boys · baby · kids · unisex`.
Sources: `details.Department` (89% coverage, 78 raw values that collapse to ~10 by
lowercasing and stripping `-adult`/`'s`/punctuation) + `categories` level 2 (93%).
**Either one: 96%.** ~95% of rows are just women/men/girls/boys.

Multi-valued Department is **0.09%** (11 rows) — ignore it, take the first token.
Junk values to route to the categories fallback: `Belts`, `Watches`, `Luggage`, `Bike`,
`Apparel`, `Industrial and Scientific`. One Chinese value (`女士`).
The ~4% with neither are genuinely audience-less (luggage tags, shoe trees, aprons) → `unisex`.

## Column coverage findings (don't re-measure these)
- **Wearable size is effectively absent.** 66% have a size-ish key but it's shipping
  dimensions: Package Dimensions 44%, Product Dimensions 21%, actual `Size` **1.8%**.
- **Material lives in `features[]`, not `details.Material`** (4.5%). `features[0]` alone is
  often the bare composition string ("Spandex", "67% Polyester, 33% Cotton") = 55%;
  all of `features[]` = 71% by keyword. Title adds only **+6.2pp** on top (77% of its
  matches are redundant); description adds +2.1pp. Ceiling ~80%.
- **`store` is the usable brand field** — 99.3% coverage, matches `details.Brand`/
  `Manufacturer` 76% of the time. 7,010 distinct values; `Generic`, `Amazon Collection`,
  `Amazon Essentials`, `Disney` are not really brands.

## LLM cost estimate (if ever needed)
50k products, measured 314 tok/product (161 slimmed by dropping `description[]`),
~700-tok schema prompt, ~130 tok out → 50.7M in / 6.5M out.
Haiku 4.5: $83 sync, **$42 via Batch API**, ~$22 with caching. Sonnet 5 doubles it,
Opus 5 is 5x. Batch API is the right call (24h SLA, 50% off, latency irrelevant).
Caveat: 5-min cache TTL makes the cached figure optimistic in a long batch.
