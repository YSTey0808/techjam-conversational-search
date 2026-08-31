"""Parse the raw `price` cell to a float.

Arithmetic, not extraction -- the only source is the `price` field, and when
that is null no amount of text mining recovers it (the number is absent from
the listing, not hidden in it). ~79% of rows come back None.

This module used to also bucket price into a `budget` band (budget/mid/premium/
luxury/unknown). That column was dropped: it is a pure function of `price`, which
is still emitted, so any consumer can band it however it likes rather than being
stuck with thresholds baked in here.
"""
import re

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
