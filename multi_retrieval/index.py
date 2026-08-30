"""One pass over the catalog, building everything the three routes need.

Independent of ``retrieval/`` on purpose: its own parsing, its own indexes. The
two packages are competing implementations, so sharing code between them would
make the comparison meaningless.

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
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Fields fed to the FTS5 index, in column order. Mirrors what the evaluator
# treats as searchable, so BM25 sees the same text a shopper could quote.
TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")

# BM25 column weights: parent_asin is unindexed, then title..description.
# Title matters most, boilerplate description least.
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# How much product text to embed. MiniLM truncates around 256 word-pieces, so
# more than this is encoded and then discarded.
EMBED_CHARS = 600

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


class CatalogIndex:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)

        self.ids: list[str] = []
        self.price: list[float | None] = []
        self.rating: list[float | None] = []
        self.reviews: list[float] = []
        self.embed_text: list[str] = []

        # category token -> product indexes, ascending (so bisect works)
        self.category_postings: dict[str, array] = {}

        self.connection = sqlite3.connect(":memory:", check_same_thread=False)

        started = time.monotonic()
        self._build()
        self.build_seconds = time.monotonic() - started
        self.size = len(self.ids)

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(self._row(index, product))
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products(rowid, parent_asin, title, categories, "
                        "features, details, store, description) VALUES (?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products(rowid, parent_asin, title, categories, "
                "features, details, store, description) VALUES (?,?,?,?,?,?,?,?)",
                batch,
            )
        self.connection.commit()

    def _row(self, index: int, product: dict) -> tuple:
        asin = str(product["parent_asin"])
        self.ids.append(asin)

        fields = {name: flatten(product.get(name)) for name in TEXT_FIELDS}

        for token in set(tokens(fields["categories"])):
            bucket = self.category_postings.get(token)
            if bucket is None:
                self.category_postings[token] = bucket = array("i")
            bucket.append(index)

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
            "build_seconds": round(self.build_seconds, 2),
            "category_tokens": len(self.category_postings),
            "priced_products": sum(1 for p in self.price if p is not None),
        }


__all__ = ["CatalogIndex", "flatten", "tokens", "parse_number"]
