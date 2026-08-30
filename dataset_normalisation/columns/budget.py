"""Bucket a catalog row into a price band.

Arithmetic, not extraction -- the only source is the `price` field, and when
that is null no amount of text mining recovers it (the number is absent from
the listing, not hidden in it). Expect ~75% of rows to come back UNKNOWN.

Thresholds are the measured quartiles of the 12k sample, rounded to round
numbers: p25 $15.59, median $22.99, p75 $38.12, p95 $129.95.
"""
import re

BUDGET  = "budget"    # < $15    -- roughly the cheapest quartile
MID     = "mid"       # $15-40   -- the middle half
PREMIUM = "premium"   # $40-100  -- top quartile below the long tail
LUXURY  = "luxury"    # >= $100  -- top ~5%
UNKNOWN = "unknown"   # price is null or unparseable

BANDS = [BUDGET, MID, PREMIUM, LUXURY, UNKNOWN]

# (exclusive upper bound, label) -- walked in order, first match wins
_EDGES = [(15.0, BUDGET), (40.0, MID), (100.0, PREMIUM)]

# "from 12.99" appears alongside plain floats; "—" is this catalog's null.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_price(value):
    """Coerce a raw price cell to float, or None when it carries no number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    m = _NUMBER.search(str(value).replace(",", ""))
    if not m:
        return None
    price = float(m.group())
    return price if price > 0 else None


def budget(row):
    """row -> (band, price). price is the parsed float, or None."""
    price = parse_price(row.get("price"))
    if price is None:
        return UNKNOWN, None
    for upper, label in _EDGES:
        if price < upper:
            return label, price
    return LUXURY, price
