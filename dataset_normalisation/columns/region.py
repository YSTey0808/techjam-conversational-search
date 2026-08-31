"""Derive a sourcing region from a catalog row.

Resolution order (first hit wins):
  1. details.Country of Origin -- an explicit country, ~1.7% of rows
  2. "Made in <country>" in the text
  3. Amazon's boilerplate: "Imported" and/or "Made in USA"
  4. UNKNOWN

The awkward case is Amazon's stock phrase "Made in USA and Imported", which
fires both signals on 2.1% of rows. It means the listing covers both, so the
row cannot be called LOCAL -- it resolves to IMPORTED, with the source marked
'usa+imported' so the ambiguity stays recoverable.

LOCAL is anchored to the United States: this is the Amazon US catalog, so
"local" means domestically made from the buyer's point of view.
"""
import re

LOCAL    = "local"
IMPORTED = "imported"
UNKNOWN  = "unknown"

# Country spellings -> canonical name. Anything else is passed through titled.
_COUNTRY_ALIAS = {
    "usa": "USA", "us": "USA", "u.s.a": "USA", "u.s": "USA",
    "america": "USA", "united states": "USA",
    "uk": "UK", "united kingdom": "UK", "england": "UK",
    "prc": "China", "p.r.c": "China",
}

_COO_KEYS = ("country of origin", "country/region of origin")

_MADE_IN_USA = re.compile(
    r"made\s+in\s+(?:the\s+)?(?:u\.?\s?s\.?\s?a\.?|united\s+states|america)\b", re.I)
_MADE_IN_X = re.compile(r"made\s+in\s+(?:the\s+)?([a-z][a-z .]{2,20}?)\b(?=[,.;)|]|\s+and\b|$)", re.I)
_IMPORTED = re.compile(r"\bimported\b", re.I)

# "made in our own factory" / "made in a way that..." -- not countries
_NOT_A_COUNTRY = {"our", "a", "the", "an", "this", "small", "house", "limited", "such"}


def _canon(name):
    key = name.strip().rstrip(".").lower()
    if key in _COUNTRY_ALIAS:
        return _COUNTRY_ALIAS[key]
    return name.strip().rstrip(".").title()


def region(row):
    """row -> (value, source).

    value is 'local', 'imported', a country name, or 'unknown'.
    source is 'details' | 'made_in' | 'usa+imported' | 'imported' | 'usa' | 'none'.
    """
    details = row.get("details") or {}

    # 1. explicit country field
    for key, value in details.items():
        if key.strip().lower() in _COO_KEYS and str(value).strip():
            country = _canon(str(value))
            return (LOCAL if country == "USA" else country), "details"

    text = " | ".join(
        [row.get("title") or ""] + list(row.get("features") or [])
    )

    usa = bool(_MADE_IN_USA.search(text))
    imported = bool(_IMPORTED.search(text))

    # 2. a named country other than the USA
    if not usa:
        for raw in _MADE_IN_X.findall(text):
            if raw.strip().lower() not in _NOT_A_COUNTRY:
                return _canon(raw), "made_in"

    # 3. boilerplate. "Made in USA and Imported" covers both, so it cannot
    #    be called local -- fall to imported but record the collision.
    if usa and imported:
        return IMPORTED, "usa+imported"
    if usa:
        return LOCAL, "usa"
    if imported:
        return IMPORTED, "imported"

    return UNKNOWN, "none"
