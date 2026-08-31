"""Extract functional product attributes from a catalog row.

Reads features[] only -- these bullets are where Amazon sellers state
capabilities ("Machine washable", "RFID blocking", "Moisture wicking").
Closed vocabulary of 25 attributes, mined from the actual bullet text.

Scope note: this column is capabilities, not composition, occasion or look.
`material` owns what it is made of, `style` owns how it looks, `use_case` owns
what it is for. "stretch" is the one deliberate overlap with material -- a
buyer filtering for stretch cares about the property, not the spandex.

Empty list when nothing is stated, consistent with material and color.
"""
import re

# canonical attribute -> alias pattern (matched case-insensitively)
VOCAB = {
    "machine-washable":  r"machine[\s-]?wash(?:able|ing)?",
    "stretch":           r"stretch(?:y|able)?|elastic(?:ated)?|four[\s-]way\s*stretch",
    "lightweight":       r"light[\s-]?weight",
    "pockets":           r"pockets?",
    "breathable":        r"breathable|air[\s-]?flow|ventilat(?:ed|ion)",
    "adjustable":        r"adjustable",
    "padded":            r"padded|cushion(?:ed|ing)",
    "quick-dry":         r"quick[\s-]?dry(?:ing)?|fast[\s-]?dry(?:ing)?|moisture[\s-]?wicking|wicking",
    "non-slip":          r"non[\s-]?slip|anti[\s-]?slip|slip[\s-]?resistant|non[\s-]?skid",
    "waterproof":        r"water\s?proof",
    "water-resistant":   r"water[\s-]?resistant|water[\s-]?repellent",
    "hypoallergenic":    r"hypo[\s-]?allergenic|nickel[\s-]?free|lead[\s-]?free",
    "handmade":          r"handmade|hand[\s-]?crafted|hand[\s-]?made",
    "uv-protection":     r"\buv\b|\bupf\b|sun\s*protection",
    "memory-foam":       r"memory\s?foam",
    "recycled":          r"recycled|eco[\s-]?friendly|sustainabl[ey]",
    "arch-support":      r"arch\s?support",
    "insulated":         r"insulated|thermal\b",
    "windproof":         r"wind\s?proof",
    "packable":          r"packable|foldable|collapsible|fold[\s-]?away",
    "wrinkle-free":      r"wrinkle[\s-]?(?:free|resistant)",
    "rfid-blocking":     r"\brfid\b",
    "anti-odor":         r"anti[\s-]?odou?r|odou?r[\s-]?resistant|anti[\s-]?microbial|antimicrobial",
    "reversible":        r"reversible",
    "tarnish-resistant": r"tarnish[\s-]?(?:free|resistant|proof)",
}

_PATTERNS = [(name, re.compile(r"\b(?:" + pat + r")\b", re.I)) for name, pat in VOCAB.items()]

# "no pockets", "without adjustable strap" -- the bullet denies the attribute.
_NEGATED = re.compile(r"(?:\bno\b|\bnot\b|\bwithout\b|\bnon\b)\W{0,3}$", re.I)


def extract_features(row):
    """row -> list[str] of canonical attributes, in order of appearance.

    Empty list when features[] states none. Roughly half the catalog carries
    at least one.
    """
    text = " | ".join(row.get("features") or [])
    if not text.strip():
        return []

    found = {}
    for name, pat in _PATTERNS:
        for m in pat.finditer(text):
            if _NEGATED.search(text[max(0, m.start() - 12):m.start()]):
                continue           # denied here; a later bullet may still assert it
            found.setdefault(name, m.start())
            break

    return [n for n, _ in sorted(found.items(), key=lambda kv: kv[1])]
