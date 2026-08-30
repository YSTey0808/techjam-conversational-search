"""Extract a normalised material list from a catalog row.

Reads features[], matches a closed vocabulary, and returns canonical names
ordered by prominence: materials carrying a composition percentage come first,
highest first, then the rest in the order they appear.

The vocabulary is closed on purpose. Percentages in this catalog are just as
often marketing ("100% satisfaction guaranteed", "30% off", "100% brand new")
as composition, so an open "N% <word>" capture picks up junk.
"""
import re

# canonical name -> alias pattern (matched case-insensitively, longest first)
VOCAB = {
    # --- fibres ---
    "cotton":        r"(?:organic|combed|supima|pima|preshrunk|brushed)?\s*cotton",
    "polyester":     r"polyester(?:\s*fiber)?|\bpoly\b",
    "spandex":       r"spandex|elastane|lycra",
    "nylon":         r"nylon|polyamide",
    "rayon":         r"rayon|viscose",
    "modal":         r"modal",
    "wool":          r"(?:merino|lambs?|shetland)?\s*wool",
    "cashmere":      r"cashmere",
    "silk":          r"silk",
    "linen":         r"linen",
    "acrylic":       r"acrylic",
    "bamboo":        r"bamboo",
    "microfiber":    r"micro\s?fib(?:er|re)",

    # --- leathers (faux before real: longest-match-wins handles it) ---
    "faux leather":  r"(?:faux|synthetic|vegan|pu|p\.u\.|artificial|imitation)[\s-]*leather|pleather|polyurethane\s*leather",
    "leather":       r"(?:genuine|real|full[\s-]grain|top[\s-]grain|nappa|patent)?\s*leather",
    "suede":         r"(?:faux\s*)?suede|nubuck",

    # --- constructed fabrics ---
    "denim":         r"denim",
    "canvas":        r"canvas",
    "mesh":          r"mesh",
    "fleece":        r"fleece|sherpa",
    "velvet":        r"velvet|velour",
    "satin":         r"satin",
    "chiffon":       r"chiffon",
    "lace":          r"lace",
    "jersey":        r"jersey",
    "corduroy":      r"corduroy",
    "tweed":         r"tweed",
    "neoprene":      r"neoprene",
    "flannel":       r"flannel",
    "terry":         r"terry\s*cloth|french\s*terry",

    # --- plastics / rubber ---
    "rubber":        r"rubber|\bEVA\b|thermoplastic",
    "polyurethane":  r"polyurethane|\bTPU\b",
    "pvc":           r"pvc|polyvinyl\s*chloride",

    # --- metals: bare "gold"/"silver" is usually a COLOUR, so require a
    #     qualifier that only appears when the metal is the actual material ---
    "sterling silver": r"sterling\s*silver|\b925\s*silver",
    "silver":        r"silver[\s-]*(?:plated|filled|tone\b(?!\s*(?:color|colour)))|(?:solid|pure|fine)\s*silver",
    "gold":          r"(?:\d{1,2}\s*k(?:t|arat)?|solid|pure|white|yellow|rose)\s*gold|gold[\s-]*(?:plated|filled)",
    "stainless steel": r"stainless\s*steel",
    "titanium":      r"titanium",
    "platinum":      r"platinum",
    "brass":         r"brass",
    "alloy":         r"(?:zinc|metal)?\s*alloy",

    # --- jewellery / trim ---
    "pearl":         r"(?:fresh\s?water\s*|faux\s*)?pearls?",
    "crystal":       r"crystal|rhinestone|cubic\s*zirconia|\bcz\b",
    "resin":         r"resin",
    "wood":          r"wood(?:en)?",
    "glass":         r"glass",

    # --- generic fallbacks, only useful when nothing specific is stated ---
    "synthetic":     r"synthetic|man[\s-]?made|manmade",
}

# Longest alternation first so "faux leather" beats "leather" and
# "sterling silver" beats "silver".
_ORDER = sorted(VOCAB, key=lambda k: -len(VOCAB[k]))
_PATTERNS = [(name, re.compile(r"\b(?:" + VOCAB[name] + r")\b", re.I)) for name in _ORDER]

# "63.9% Polyester" -- used only to rank, never to discover new materials.
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*$")


def extract_materials(row):
    """row -> list[str] of canonical materials, most prominent first.

    Empty list when nothing is stated. Percentages, where present, decide the
    order; otherwise order of appearance is kept (this catalog lists
    composition in descending order, so the first entry is the dominant one).
    """
    text = " | ".join(row.get("features") or [])
    if not text.strip():
        return []

    found = {}   # canonical -> (percentage or None, first position)
    for name, pat in _PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        # a percentage immediately before the match makes this a composition
        pct_match = _PCT.search(text[max(0, m.start() - 12):m.start()])
        pct = float(pct_match.group(1)) if pct_match else None
        found[name] = (pct, m.start())

    # overlapping matches: drop the generic when a specific one covers it
    if "faux leather" in found:
        found.pop("leather", None)
    if "sterling silver" in found:
        found.pop("silver", None)
    if len(found) > 1:
        found.pop("synthetic", None)

    # percentages first (descending), then the rest in order of appearance
    with_pct = [(n, v) for n, v in found.items() if v[0] is not None]
    without  = [(n, v) for n, v in found.items() if v[0] is None]
    with_pct.sort(key=lambda kv: -kv[1][0])
    without.sort(key=lambda kv: kv[1][1])
    return [n for n, _ in with_pct] + [n for n, _ in without]
