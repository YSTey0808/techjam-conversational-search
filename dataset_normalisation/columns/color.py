"""Extract a normalised colour list from a catalog row.

Resolution order (first hit wins):
  1. details.Color, when it parses to the closed vocabulary
  2. title, after stripping the brand name
  3. empty

features[] is deliberately NOT read. Measured against details.Color as ground
truth, a single colour drawn from features[] is right only 30% of the time --
that text describes gift boxes, linings, care labels and *other* variants
("also available in black"), not the product. Title is 88-95% correct.

Expect ~35% of rows to come back empty and treat that as correct. parent_asin
is a variant *group*: the title describes the group while details.Color names
one child SKU, so many listings genuinely never commit to a colour.

No metal or gemstone context guard. Unlike material.py -- which rejects bare
"gold"/"silver" precisely because they read as colours -- a sterling silver
bracelet IS silver-coloured, and details.Color agrees. Suppressing it measured
-1.2pp precision and -4pp recall, so the rule is not ported across.
"""
import re

# canonical name -> alias pattern (matched case-insensitively, longest first)
VOCAB = {
    # --- achromatic ---
    "black":      r"black|jet\s*black",
    "white":      r"white|off[\s-]white",
    "gray":       r"gr[ea]y|charcoal|gunmetal|graphite|slate|smoke",
    "ivory":      r"ivory",
    "cream":      r"cream",

    # --- primaries / secondaries ---
    "red":        r"red|crimson|scarlet",
    "blue":       r"blue|cobalt|sky\s*blue|royal\s*blue",
    "green":      r"green|forest\s*green|lime",
    "yellow":     r"yellow",
    "orange":     r"orange",
    "purple":     r"purple|violet|lilac|lavender|plum",
    "pink":       r"pink|fuchsia|fuschia|magenta|blush",
    "brown":      r"brown|chocolate|espresso|mocha|cognac|camel",

    # --- named shades ---
    "navy":       r"navy",
    "burgundy":   r"burgundy|maroon|wine",
    "teal":       r"teal|aqua|turquoise",
    "olive":      r"olive",
    "khaki":      r"khaki",
    "beige":      r"beige|taupe|sand",
    "tan":        r"tan",
    "coral":      r"coral",
    "mint":       r"mint",
    "peach":      r"peach|apricot",
    "mustard":    r"mustard",
    "salmon":     r"salmon",
    "rust":       r"rust",
    "indigo":     r"indigo",
    "champagne":  r"champagne",
    "nude":       r"nude",

    # --- metallics: kept as colours on purpose (see module docstring) ---
    "rose gold":  r"rose\s*gold",
    "gold":       r"gold",
    "silver":     r"silver",
    "bronze":     r"bronze",
    "copper":     r"copper",
    "platinum":   r"platinum",
    "pewter":     r"pewter",

    # --- other ---
    "clear":      r"clear|transparent",
    "multicolor": r"multi[\s-]?colou?red?|multi[\s-]?colou?r|rainbow|assorted\s*colou?rs",
}

# Deliberately excluded, though they appear often in titles:
#   denim, pearl, stone, natural, jade, sapphire, emerald, ruby, amber, onyx
# These are materials or gemstones, not colours -- material.py owns them.
# "rose" alone is excluded too: it is nearly always "rose gold" or a flower motif.

# Longest alternation first so "rose gold" beats "gold".
_ORDER = sorted(VOCAB, key=lambda k: -len(VOCAB[k]))
_PATTERNS = [(name, re.compile(r"\b(?:" + VOCAB[name] + r")\b", re.I)) for name in _ORDER]

_COLOR_KEYS = ("color", "colour", "color name")
# 3+ digits means a variant SKU string, not a colour:
# "04-diamond-Apr-18k rose gold", "Varsity Red (650) / Black".
_SKU_ISH = re.compile(r"(?:\D*\d){3,}")


def _scan(text):
    """Return {canonical: first position} for every colour found in text."""
    found = {}
    for name, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            found[name] = m.start()
    # a "rose gold" match also matches "gold"; keep only the specific one
    if "rose gold" in found:
        found.pop("gold", None)
    return found


def _ordered(found):
    return [n for n, _ in sorted(found.items(), key=lambda kv: kv[1])]


def extract_colors(row):
    """row -> (list[str], source). source is 'details' | 'title' | 'none'.

    Colours are returned in order of appearance. Empty list when nothing is
    stated -- which is the correct answer for roughly a third of the catalog.
    """
    details = row.get("details") or {}

    # 1. the only pre-labelled colour field in the data (~4% of rows)
    for key, value in details.items():
        if key.strip().lower() in _COLOR_KEYS:
            raw = str(value)
            if not _SKU_ISH.search(raw):
                found = _scan(raw)
                if found:
                    return _ordered(found), "details"
            break

    # 2. title, with the brand stripped first -- "Sabrina Silver", "Pink Queen"
    #    and "Yellow Box" are stores, not colours.
    title = row.get("title") or ""
    store = (row.get("store") or "").strip()
    if len(store) > 3 and store.lower() in title.lower():
        title = re.sub(re.escape(store), " ", title, flags=re.I)

    found = _scan(title)
    if found:
        return _ordered(found), "title"

    return [], "none"
