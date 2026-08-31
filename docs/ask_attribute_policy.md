# Ask Attribute Policy

## Purpose

`starter/ask.py` decides which `ask_attribute` to return for the next
clarification question. It does not extract user state, retrieve products, or
rerank candidates.

## Inputs

- session state slots from query/reply extraction
- `clarification.asked_attributes` from session state
- candidate pool from retrieval: `list[parent_asin]`
- normalized product slots from preprocessing:
  `prep.product_slots[parent_asin][attribute]`

## Output

- `ask_attribute`: the best eligible attribute, or `null`
- `message`: fixed template text for the selected attribute
- `list_width`: fixed at 10 to match the Top-10 recommendation contract

## Eligibility

An attribute is scored only if:

- it is not already filled in session state
- it has not already been asked in the current intent context
- the current turn is below the final turn

Session-state known slots are read from:

```text
state["slots"][attribute]["val"]
```

Clarification history is read from:

```text
state["clarification"]["asked_attributes"]
```

## Scoring

For each eligible attribute:

```text
score(attribute) =
normalized_entropy(attribute) x known_ratio(attribute)
```

Let `C` be the current candidate pool and `a` be one eligible attribute.

```text
N = number of products in C
K_a = products in C where attribute a has at least one known value
known_ratio(a) = |K_a| / N
```

For each known value `v` of attribute `a`, count the product mass in that
bucket:

```text
count_a(v) = number of candidate products whose slot a contains value v
known_total(a) = sum_v count_a(v)
```

If a product has multiple values for the same attribute, its weight is split
across those values so one product still contributes total mass 1.

Convert counts into probabilities:

```text
p_a(v) = count_a(v) / known_total(a)
```

Compute Shannon entropy over known values only:

```text
H(a) = - sum_v p_a(v) log2 p_a(v)
```

Normalize by the maximum possible entropy for the number of known value
buckets:

```text
m_a = number of distinct known values for attribute a
normalized_entropy(a) = H(a) / log2(m_a), if m_a > 1
normalized_entropy(a) = 0, otherwise
```

`known_ratio` is:

```text
number of candidate products with a known value for this attribute
/
total number of candidate products
```

Final score:

```text
score(a) = normalized_entropy(a) x known_ratio(a)
```

This follows a C4.5-style missing-value discount: compute split quality from
known values, then scale by the fraction of candidates where the attribute is
known.

### Example

Current candidate pool:

```text
100 products
```

For `color`, the extracted product slots are:

```text
black: 30
white: 20
blue: 10
unknown/null: 40
```

Unknown/null is not a branch. It is handled by `known_ratio`.

```text
known_total(color) = 30 + 20 + 10 = 60
known_ratio(color) = 60 / 100 = 0.60
```

Probabilities over known values:

```text
p(black) = 30 / 60 = 0.50
p(white) = 20 / 60 = 0.333
p(blue) = 10 / 60 = 0.167
```

Shannon entropy:

```text
H(color) =
- [
  0.50 log2(0.50)
  + 0.333 log2(0.333)
  + 0.167 log2(0.167)
]
≈ 1.459
```

There are 3 known value buckets, so the maximum possible entropy is:

```text
log2(3) ≈ 1.585
```

Normalized entropy:

```text
normalized_entropy(color) =
1.459 / 1.585
≈ 0.921
```

Final score:

```text
score(color) =
0.921 x 0.60
≈ 0.553
```

## Missing Values

Null, empty, and `"unknown"` product slot values mean missing catalog evidence.
They are not treated as real entropy branches.

Products with missing slot values remain in the candidate pool. Missing values
only reduce the attribute score through `known_ratio`.

## Product Slot Schema

Expected preprocessing output:

```text
prep.product_slots[parent_asin][attribute] -> list[str] | str | None
```

Askable attributes:

```text
category, material, color, size, brand, budget, style, feature, use_case, other
```

`catalog_normalised` also contains `audience` and `region`, but these are not
API `ask_attribute` values and are ignored by `ask.py`.

The API field is `other`; the normalized/session slot may be named `others`.
`ask.py` maps `other -> others`.

Comma-separated extracted values are treated as multiple slot values.

## Budget

Budget may need separate handling because it is derived from price and may use
category-aware price buckets. If every product receives a derived budget bucket,
budget can become over-selected unless the price spread is meaningful.

For now, budget is scored like the other attributes when present in
`product_slots`.

## Limitations

- depends on preprocessing quality and slot fill rates
- assumes `prep.product_slots` is built before `ask.decide(...)`
- assumes session state records `asked_attributes`
- does not use ranking confidence to decide whether asking is needed
- does not apply category-specific precedence yet
