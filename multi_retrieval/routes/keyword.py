"""Keyword route: BM25 over the FTS5 index.

The primary route for a Buying turn, where the customer has named something
concrete and we want the products whose text actually contains those words.

Two passes per query, because they fail in different ways:

* an OR of every term keeps recall up when only part of the phrasing matches
* a quoted phrase per multi-word slot rewards adjacency, so "hiking boot" beats
  a product that mentions hiking in one place and boots in another

FTS5 query syntax is fragile — a stray quote or bracket in customer text is a
syntax error, not a poor result. Every expression here is rebuilt from
re-tokenized words, so raw punctuation can never reach the parser.
"""

from __future__ import annotations

from ..index import CatalogIndex, tokens
from ..types import Slots

ROUTE_LIMIT = 500
MAX_TERMS = 40


def build_expression(phrases: list[str]) -> str:
    """Turn slot values into one FTS5 MATCH expression.

    Returns "" when there is nothing searchable, which the caller treats as
    "this route has no opinion" rather than as an error.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for phrase in phrases:
        words = tokens(phrase)
        if not words:
            continue
        if len(words) > 1:
            quoted = '"' + " ".join(words) + '"'
            if quoted not in seen:
                seen.add(quoted)
                terms.append(quoted)
        for word in words:
            quoted = f'"{word}"'
            if quoted not in seen:
                seen.add(quoted)
                terms.append(quoted)

    return " OR ".join(terms[:MAX_TERMS])


class KeywordRoute:
    name = "keyword"

    def __init__(self, index: CatalogIndex, *, limit: int = ROUTE_LIMIT) -> None:
        self.index = index
        self.limit = limit

    def search(self, slots: Slots) -> dict[int, float]:
        expression = build_expression(slots.phrases)
        if not expression:
            return {}
        return dict(self.index.search_bm25(expression, self.limit))


__all__ = ["KeywordRoute", "build_expression"]
