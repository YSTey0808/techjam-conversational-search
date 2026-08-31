"""Catalog loading and indexing. NO OWNER. Built ONCE in Agent.__init__.

Everything here is read-only after construction. If you need a new lookup
structure, add it here rather than rebuilding anything per turn.

Public surface used by the five modules:

    build(catalog_path)          -> Preprocessing
    prep.lookup(key, broad)      -> set[parent_asin]   PREFIX-TOLERANT
    prep.idf(key)                -> float
    prep.category_pool(category) -> set[parent_asin]
    prep.popularity[asin]        -> float              log1p(rating_number)
    prep.price[asin]             -> float | None       never raises
    prep.coarse[asin]            -> str
    prep.product_keys[asin]      -> tuple[str, ...]    every indexed string
    prep.attribute_of(key)       -> str
    prep.title[asin]             -> str
    prep.text[asin]              -> str                searchable text, truncated
    prep.average_rating[asin]    -> float | None       stars, not the count
    prep.product_slots[asin]     -> dict[str, list[str]]   normalised slot table
"""

from __future__ import annotations

import json
import math
import os
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
CONSTRAINT_LIMIT = 180
_EXCLUDED_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
_WS_RE = re.compile(r"\s+")

# Prefix matching guards. A long feature string can arrive truncated, so exact
# lookup misses and we fall back to prefix matching. Short strings are
# excluded because "leather" would prefix-match a large slice of the catalog.
MIN_PREFIX_LEN = 12
PREFIX_FANOUT_CAP = 64

# Per-product searchable text kept for the reranker. Truncated because the full
# corpus over 50k products is far more than any downstream consumer reads.
TEXT_LIMIT = 800        


def clean_constraint(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    return _WS_RE.sub(" ", str(value)).strip(" -;,.\t\n")[:limit].rstrip()


def normalize(text: str) -> str:
    """The matching key: whitespace-collapsed, trimmed, case-folded."""
    return clean_constraint(text).lower()


def flatten_values(value: object) -> list[str]:
    """Render a features list or details dict as individual strings."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def coarse_category(values: list[str]) -> str:
    """Last two meaningful category parts.

    LOSSY BY CONSTRUCTION: 1,136 products land on 'Shoes & Jewelry Westlake'
    because a store name sits in the category path. Treat the result as a SOFT
    gate -- never filter a candidate out on it alone.
    """
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def safe_price(product: dict) -> float | None:
    """NEVER raises.

    79% of catalog rows have a null price, and 117 hold a string -- 112 of
    those a single mojibake character, the rest of the form "from 12.99".
    No other file may call float(product["price"]).
    """
    value = product.get("price")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().lower().removeprefix("from").strip().lstrip("$").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def safe_rating(product: dict) -> float | None:
    """Average star rating. NEVER raises; None when the row has no usable value."""
    value = product.get("average_rating")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def constraint_candidates(product: dict) -> list[str]:
    """Every catalog string a customer could plausibly be describing.

    Features and details entries, plus three synthesised strings: a bare
    material word, "color: x", and "budget around $p". Order is stable so the
    first four can be treated as the product's most characteristic strings.
    """
    candidates = [*flatten_values(product.get("features")), *flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        value = clean_constraint(item)
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    if not cleaned:
        cleaned = [clean_constraint(str(product.get("title") or "product"))]
    return cleaned


# --- the normalised slot table ---------------------------------------------
# A second, pre-extracted view of the same 50,000 products:
# data/catalog_normalised.jsonl, one row per parent_asin, every attribute
# already reduced to a small controlled vocabulary. ask.py needs it to measure
# how well an attribute splits a candidate pool -- raw features/details cannot
# answer "how many distinct colours are in this pool?".
#
# scripts/build_inverted_index.py reads the same file through the same
# functions, so the ask policy and the inverted index can never end up on
# different vocabularies.

SLOT_PATH = "data/catalog_normalised.jsonl"

# Every attribute column in the file. `price` is deliberately absent: it is a
# number, not a slot value, and nothing here buckets it.
SLOT_FIELDS = ("category", "audience", "brand", "material", "color",
               "feature", "style", "use_case", "region")

# "unknown" is a real string in the file, not an absence marker we invented --
# it is the value on 33,346 of the 50,000 `region` cells. Treating it as a
# value would make "region" look like a well-populated, high-entropy attribute.
MISSING_SLOT_VALUES = {"", "unknown", "null", "none", "n/a", "na"}


def normalize_slot_value(value: object) -> str:
    """One slot cell -> its comparison form, or "" if it says nothing."""
    if value is None or isinstance(value, bool):
        return ""
    text = _WS_RE.sub(" ", str(value)).strip().lower()
    return "" if text in MISSING_SLOT_VALUES else text


def row_slots(row: dict) -> dict[str, list[str]]:
    """One normalised catalog row -> {field: [values]}, empty fields dropped.

    Scalar and list columns are handled the same way, so a schema change that
    turns `brand` into a list costs nothing here.
    """
    slots: dict[str, list[str]] = {}
    for field in SLOT_FIELDS:
        raw = row.get(field)
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        values: list[str] = []
        for item in items:
            text = normalize_slot_value(item)
            if text and text not in values:
                values.append(text)
        if values:
            slots[field] = values
    return slots


def load_product_slots(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """The slot table keyed by parent_asin. NEVER raises.

    A missing file returns {}, and ask._slot_values already treats that as "no
    data": every split score is 0.0 and the agent falls back to asking "other".
    Degrading is the whole point -- this file is not in the release bundle, and
    an agent that cannot be constructed scores zero on every session.
    """
    table: dict[str, dict[str, list[str]]] = {}
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError:
        return table
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                asin = str(row["parent_asin"])
            except (ValueError, KeyError, TypeError):
                continue        # one malformed row must not cost the other 49,999
            table[asin] = row_slots(row)
    return table


# --- attribute classification ---------------------------------------------
# Used by ask.py to work out which attribute best splits a pool. Ours, not
# borrowed: it only has to be self-consistent with our own index.
_MATERIAL_WORDS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex",
                   "silk", "rayon", "fabric", "denim", "suede", "mesh")
_COLOR_WORDS = ("color", "black", "white", "blue", "red", "pink", "green",
                "brown", "gray", "grey", "purple", "yellow", "orange")
_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow", "length", "fit ", "inseam")
_STYLE_WORDS = ("department", "style", "sleeve", "neck", "collar", "cut", "closure")
_USE_WORDS = ("hiking", "running", "gym", "winter", "summer", "outdoor", "work",
              "travel", "casual", "formal", "athletic", "waterproof", "water resistant")


def _classify(text: str) -> str:
    lowered = text.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in _MATERIAL_WORDS):
        return "material"
    if any(word in lowered for word in _COLOR_WORDS):
        return "color"
    if any(word in lowered for word in _SIZE_WORDS):
        return "size"
    if any(word in lowered for word in _USE_WORDS):
        return "use_case"
    if any(word in lowered for word in _STYLE_WORDS):
        return "style"
    return "feature"


class Preprocessing:
    """Read-only catalog index."""

    def __init__(self) -> None:
        self.asins: list[str] = []
        self.popularity: dict[str, float] = {}
        self.rating_number: dict[str, int] = {}
        self.average_rating: dict[str, float | None] = {}
        self.price: dict[str, float | None] = {}
        self.title: dict[str, str] = {}
        self.text: dict[str, str] = {}
        self.coarse: dict[str, str] = {}
        self.product_keys: dict[str, tuple[str, ...]] = {}
        self.product_slots: dict[str, dict[str, list[str]]] = {}
        self.postings: dict[str, set[str]] = {}
        self.canon_postings: dict[str, set[str]] = {}
        self.cat_index: dict[str, set[str]] = {}
        self.sorted_keys: list[str] = []
        self.n_docs = 0
        self._idf_cache: dict[str, float] = {}
        self._attr_cache: dict[str, str] = {}

    # -- lookup -------------------------------------------------------------

    def _prefix_keys(self, key: str) -> list[str]:
        """Indexed keys related to `key` by truncation, in either direction."""
        if len(key) < MIN_PREFIX_LEN:
            return []
        keys = self.sorted_keys
        out: list[str] = []
        i = bisect_left(keys, key)
        j = i
        while j < len(keys) and keys[j].startswith(key):
            out.append(keys[j])
            j += 1
            if len(out) >= PREFIX_FANOUT_CAP:
                break
        if i > 0 and len(keys[i - 1]) >= MIN_PREFIX_LEN and key.startswith(keys[i - 1]):
            out.append(keys[i - 1])
        return out

    def lookup(self, key: str, broad: bool = False) -> set[str]:
        """Products carrying this string. Prefix-tolerant on an exact miss.

        `broad=False` restricts to the product's most characteristic strings,
        which is more precise; `broad=True` searches every indexed string.
        Retrieval widens to broad as a backoff step.
        """
        if not key:
            return set()
        table = self.postings if broad else self.canon_postings
        hit = table.get(key)
        if hit:
            return hit
        matched: set[str] = set()
        for candidate in self._prefix_keys(key):
            found = table.get(candidate)
            if found:
                matched |= found
        return matched

    def document_frequency(self, key: str) -> int:
        hit = self.postings.get(key)
        if hit is not None:
            return len(hit)
        return sum(len(self.postings.get(c, ())) for c in self._prefix_keys(key))

    def idf(self, key: str) -> float:
        """Smoothed inverse document frequency.

        Mandatory for any matching: "Imported" sits on ~13.9k products and
        "Machine Wash" on ~8.9k, so unweighted matching is meaningless.
        """
        cached = self._idf_cache.get(key)
        if cached is not None:
            return cached
        value = math.log((self.n_docs + 1.0) / (self.document_frequency(key) + 1.0))
        self._idf_cache[key] = value
        return value

    def category_pool(self, category: str | None) -> set[str]:
        if not category:
            return set()
        return self.cat_index.get(category, set())

    def attribute_of(self, key: str) -> str:
        cached = self._attr_cache.get(key)
        if cached is None:
            cached = self._attr_cache[key] = _classify(key)
        return cached


_ACTIVE: Preprocessing | None = None


def active() -> Preprocessing | None:
    """The most recently built index, or None.

    Exists because extract() resolves catalog vocabulary but its signature
    takes no `prep` argument. Read-only; treat it as a singleton handle, not
    as mutable global state.
    """
    return _ACTIVE


def build(catalog_path: str | Path, slots_path: str | Path | None = None) -> Preprocessing:
    """Load and index the catalog. Call once.

    `slots_path` defaults to TECHJAM_SLOTS, then to data/catalog_normalised.jsonl
    -- the same override shape retrieve.py uses for TECHJAM_CATALOG. A missing
    slot file is not an error; see load_product_slots.
    """
    global _ACTIVE
    prep = Preprocessing()
    if slots_path is None:
        slots_path = os.environ.get("TECHJAM_SLOTS", SLOT_PATH)
    prep.product_slots = load_product_slots(slots_path)
    postings: dict[str, set[str]] = defaultdict(set)
    canon: dict[str, set[str]] = defaultdict(set)
    cat_index: dict[str, set[str]] = defaultdict(set)

    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product["parent_asin"])
            prep.asins.append(asin)

            rating = product.get("rating_number")
            rating = rating if isinstance(rating, int) else 0
            prep.rating_number[asin] = rating
            prep.popularity[asin] = math.log1p(max(rating, 0))
            prep.average_rating[asin] = safe_rating(product)
            prep.price[asin] = safe_price(product)
            prep.title[asin] = clean_constraint(str(product.get("title") or ""), TEXT_LIMIT)
            prep.text[asin] = searchable_text(product)[:TEXT_LIMIT]

            category = coarse_category([str(v) for v in (product.get("categories") or [])])
            prep.coarse[asin] = category
            cat_index[category].add(asin)

            keys = [normalize(v) for v in constraint_candidates(product)]
            keys = [k for k in keys if k]
            prep.product_keys[asin] = tuple(keys)
            for position, key in enumerate(keys):
                postings[key].add(asin)
                if position < 4:
                    canon[key].add(asin)

    prep.postings = dict(postings)
    prep.canon_postings = dict(canon)
    prep.cat_index = dict(cat_index)
    prep.sorted_keys = sorted(prep.postings)
    prep.n_docs = len(prep.asins)
    _ACTIVE = prep
    return prep
