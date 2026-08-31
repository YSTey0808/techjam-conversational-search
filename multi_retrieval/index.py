"""One pass over the catalog, building everything the routes and the hard
filter need.

Two input shapes are accepted and auto-detected per file:

* **normalised** -- the faceted table from ``dataset_normalisation`` (columns
  ``category``, ``audience``, ``brand``, ``price`` ...). Its clean enums drive
  the hard filter: ``category_enum_postings`` / ``audience_postings`` /
  ``brand_postings``.

  It carries no product title, free text, or category *path*, so the lexical
  routes (keyword, vector) and the category route -- which needs fine-grained
  path tokens like "necklaces" or "hiking boots", not an 8-value enum -- cannot
  work off it. When a row-aligned raw ``catalog.jsonl`` is supplied as
  ``raw_catalog_path`` the FTS columns, the embedding text, and
  ``category_postings`` are filled from it. The two files are zipped row for
  row and the alignment is verified. Without the sidecar these fall back to the
  enum values (degraded, but the package still imports and runs).

* **raw** -- the original Amazon export (``title``, ``categories`` path,
  ``features`` ...). Parsed unchanged; ``category_enum_postings`` stays empty
  and ``raw_catalog_path`` is ignored.

Products are addressed by integer index throughout. The FTS5 table is inserted
with an explicit rowid equal to that index, so a BM25 query hands back the same
numbering the other routes use, with no translation layer.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from array import array
from itertools import zip_longest
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# FTS5 columns, in order. Kept identical for both input shapes: a normalised row
# maps its facets onto these columns (see ``_row_normalised``) so ``search_bm25``
# and the keyword route need no branch.
TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")

# BM25 column weights: parent_asin is unindexed, then title..description.
# Title matters most, boilerplate description least.
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# How much product text to embed. MiniLM truncates around 256 word-pieces, so
# more than this is encoded and then discarded.
EMBED_CHARS = 600

# Facet values that mean "not stated" and must not become searchable tokens or
# postings -- treating "unknown" as a brand would gate every unbranded product.
_EMPTY_FACET = {"", "unknown", "none", "n/a"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def flatten(value: object) -> str:
    """Turn any catalog field into one flat string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def parse_number(value: object) -> float | None:
    """Catalog prices are sometimes the string '-', so this has to fail softly."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _facet_list(value: object) -> list[str]:
    """A normalised list facet, lowercased, with the empty markers dropped."""
    if isinstance(value, list):
        items = [str(v).strip().lower() for v in value]
    elif value is None:
        items = []
    else:
        items = [str(value).strip().lower()]
    return [item for item in items if item and item not in _EMPTY_FACET]


def _facet_scalar(value: object) -> str:
    """A normalised scalar facet, lowercased, empty markers collapsed to ''."""
    text = "" if value is None else str(value).strip().lower()
    return "" if text in _EMPTY_FACET else text


class CatalogIndex:
    def __init__(
        self,
        catalog_path: str | Path,
        *,
        raw_catalog_path: str | Path | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.raw_catalog_path = Path(raw_catalog_path) if raw_catalog_path else None
        self.format = "raw"                    # set for real in _build()

        self.ids: list[str] = []
        self.price: list[float | None] = []
        self.rating: list[float | None] = []
        self.reviews: list[float] = []
        self.embed_text: list[str] = []

        # token -> product indexes, ascending.
        #
        # category_postings is tokens of the category *path* -- fine-grained
        # ("necklaces", "hiking", "loafers") -- and is what CategoryRoute scores
        # on. For a raw file it comes from that file; for a normalised file it
        # comes from the raw sidecar (the faceted table has no path).
        self.category_postings: dict[str, array] = {}

        # The hard filter's category gate uses this instead: tokens of the clean
        # 8-value `category` enum from the normalised table. Empty for a raw file.
        self.category_enum_postings: dict[str, array] = {}

        self.audience_postings: dict[str, array] = {}   # normalised only
        self.brand_postings: dict[str, array] = {}      # normalised only

        self.connection = sqlite3.connect(":memory:", check_same_thread=False)

        started = time.monotonic()
        self._build()
        self.build_seconds = time.monotonic() - started
        self.size = len(self.ids)
        self.has_ratings = any(value is not None for value in self.rating)

    # ------------------------------------------------------------------ build

    def _detect_format(self, product: dict) -> str:
        # A normalised row carries a scalar `category` enum and no product title.
        if isinstance(product.get("category"), str) and "title" not in product:
            return "normalised"
        return "raw"

    def _append_posting(self, table: dict[str, array], key: str, index: int) -> None:
        bucket = table.get(key)
        if bucket is None:
            table[key] = bucket = array("i")
        bucket.append(index)

    def _peek_format(self) -> str:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return self._detect_format(json.loads(line))
        return "raw"

    def _nonblank(self, path: Path):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        insert = (
            "INSERT INTO products(rowid, parent_asin, title, categories, "
            "features, details, store, description) VALUES (?,?,?,?,?,?,?,?)"
        )

        self.format = self._peek_format()
        use_sidecar = (
            self.format == "normalised"
            and self.raw_catalog_path is not None
            and self.raw_catalog_path.exists()
        )

        batch: list[tuple] = []

        def flush() -> None:
            if batch:
                cursor.executemany(insert, batch)
                batch.clear()

        if use_sidecar:
            # Zip the faceted table and the raw export row for row: the facets
            # drive the structured layer, the raw text drives FTS / embeddings.
            paired = zip_longest(
                self._nonblank(self.catalog_path),
                self._nonblank(self.raw_catalog_path),
            )
            for index, (norm_line, raw_line) in enumerate(paired):
                if norm_line is None or raw_line is None:
                    raise RuntimeError(
                        f"{self.catalog_path.name} and its sidecar "
                        f"{self.raw_catalog_path.name} have different row counts"
                    )
                norm = json.loads(norm_line)
                raw = json.loads(raw_line)
                if str(norm["parent_asin"]) != str(raw.get("parent_asin")):
                    raise RuntimeError(
                        f"row {index}: {self.catalog_path.name} has "
                        f"{norm['parent_asin']!r} but the sidecar has "
                        f"{raw.get('parent_asin')!r}; the two files are not row-aligned"
                    )
                batch.append(self._row_normalised(index, norm, raw))
                if len(batch) >= 1000:
                    flush()
        else:
            for index, line in enumerate(self._nonblank(self.catalog_path)):
                product = json.loads(line)
                batch.append(
                    self._row_normalised(index, product, None)
                    if self.format == "normalised"
                    else self._row_raw(index, product)
                )
                if len(batch) >= 1000:
                    flush()

        flush()
        self.connection.commit()

    # --- raw Amazon export ------------------------------------------------

    def _row_raw(self, index: int, product: dict) -> tuple:
        asin = str(product["parent_asin"])
        self.ids.append(asin)

        fields = {name: flatten(product.get(name)) for name in TEXT_FIELDS}

        for token in set(tokens(fields["categories"])):
            self._append_posting(self.category_postings, token, index)

        self.price.append(parse_number(product.get("price")))
        self.rating.append(parse_number(product.get("average_rating")))
        self.reviews.append(parse_number(product.get("rating_number")) or 0.0)

        self.embed_text.append(
            " ".join([fields["title"], fields["features"], fields["description"]])[:EMBED_CHARS]
        )

        return (
            index, asin, fields["title"], fields["categories"], fields["features"],
            fields["details"], fields["store"], fields["description"],
        )

    # --- normalised faceted table --------------------------------------

    def _row_normalised(self, index: int, product: dict, raw: dict | None) -> tuple:
        asin = str(product["parent_asin"])
        self.ids.append(asin)

        category = _facet_scalar(product.get("category"))
        audience = _facet_scalar(product.get("audience"))
        brand = _facet_scalar(product.get("brand"))

        # Clean-enum postings the hard filter's category / audience / brand
        # gates use. category is tokenised ("bags & luggage" -> "bags",
        # "luggage"); audience and brand are matched whole.
        for token in set(tokens(category)):
            self._append_posting(self.category_enum_postings, token, index)
        if audience:
            self._append_posting(self.audience_postings, audience, index)
        if brand:
            self._append_posting(self.brand_postings, brand, index)

        self.price.append(parse_number(product.get("price")))

        if raw is not None:
            # Lexical layer: real title / feature bullets / description so the
            # keyword and vector routes match what a customer actually quotes,
            # and the raw category *path* so CategoryRoute keeps its fine-grained
            # tokens. `store` stays the clean normalised brand.
            for token in set(tokens(flatten(raw.get("categories")))):
                self._append_posting(self.category_postings, token, index)
            text = {name: flatten(raw.get(name)) for name in TEXT_FIELDS}
            text["store"] = brand or text["store"]
            self.rating.append(parse_number(raw.get("average_rating")))
            self.reviews.append(parse_number(raw.get("rating_number")) or 0.0)
            if self.price[index] is None:
                self.price[index] = parse_number(raw.get("price"))
            self.embed_text.append(
                " ".join([text["title"], text["features"], text["description"]])[:EMBED_CHARS]
            )
            return (
                index, asin, text["title"], text["categories"], text["features"],
                text["details"], text["store"], text["description"],
            )

        # No sidecar: a degraded fallback so the package still imports and runs.
        # FTS text and category_postings come from the enum values -- the lexical
        # and category routes work, they just cannot match free-form phrasing or
        # fine-grained category paths.
        material = _facet_list(product.get("material"))
        color = _facet_list(product.get("color"))
        feature = _facet_list(product.get("feature"))
        style = _facet_list(product.get("style"))
        use_case = _facet_list(product.get("use_case"))
        region = _facet_scalar(product.get("region"))

        for token in set(tokens(category)):
            self._append_posting(self.category_postings, token, index)
        self.rating.append(None)
        self.reviews.append(0.0)
        self.embed_text.append(
            " ".join([category, audience, brand, *material, *color, *feature,
                      *style, *use_case])[:EMBED_CHARS]
        )
        descriptive = [*color, *style, *use_case]
        if region:
            descriptive.append(region)
        return (
            index, asin, "",
            " ".join(t for t in (category, audience) if t),
            " ".join(material + feature),
            " ".join(descriptive),
            brand, "",
        )

    # ----------------------------------------------------------------- lookup

    def category_document_frequency(self, token: str) -> int:
        bucket = self.category_postings.get(token)
        return len(bucket) if bucket is not None else 0

    def idf(self, count: int) -> float:
        return math.log(self.size / max(count, 1))

    def search_bm25(self, expression: str, limit: int) -> list[tuple[int, float]]:
        """Run one FTS5 MATCH. Returns (product index, score), best first.

        SQLite's bm25() is lower-is-better, so scores are negated on the way out
        and every route in this package agrees that higher means better.
        A malformed expression returns nothing rather than raising — query text
        comes from customer input and must never crash a turn.
        """
        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        try:
            # Order by the SAME weighted bm25 expression that is selected.
            # "ORDER BY rank" would sort by FTS5's default all-ones weighting
            # instead, so the returned order would not match the returned
            # scores -- and the column weights above would silently do nothing.
            rows = self.connection.execute(
                f"SELECT rowid, bm25(products, {weights}) FROM products "
                f"WHERE products MATCH ? ORDER BY bm25(products, {weights}) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [(int(rowid), -float(score)) for rowid, score in rows]

    def diagnostics(self) -> dict:
        return {
            "catalog_size": self.size,
            "format": self.format,
            "build_seconds": round(self.build_seconds, 2),
            "category_path_tokens": len(self.category_postings),
            "category_enum_tokens": len(self.category_enum_postings),
            "audience_values": len(self.audience_postings),
            "brands": len(self.brand_postings),
            "priced_products": sum(1 for p in self.price if p is not None),
            "has_ratings": self.has_ratings,
        }


__all__ = ["CatalogIndex", "flatten", "tokens", "parse_number"]
