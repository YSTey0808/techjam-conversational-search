"""Map a catalog row to one of 7 audience groups.

Resolution order (first hit wins):
  1. details.Department  -- present on 87% of rows, 78 raw spellings
  2. categories level 2  -- Women / Men / Girls / Boys / Baby
  3. UNISEX

`unisex` is both a real value and the fallback. Rows reaching step 3 are
genuinely audience-less (luggage tags, shoe trees, aprons), so the default is
a correct answer rather than a guess -- but `source` still distinguishes an
asserted unisex from a defaulted one.
"""
import re

WOMEN  = "women"
MEN    = "men"
GIRLS  = "girls"
BOYS   = "boys"
BABY   = "baby"
KIDS   = "kids"
UNISEX = "unisex"

AUDIENCES = [WOMEN, MEN, GIRLS, BOYS, BABY, KIDS, UNISEX]

# Order matters and is load-bearing:
#   baby before girls/boys  -- "Baby-girls" is baby, not girls
#   boys/girls before kids  -- "teen-boys" is boys, not kids
#   women/men before unisex -- "Adult-Male" is men, not the bare "adult" rule
RULES = [
    (BABY,   r"baby|infant|newborn"),
    (GIRLS,  r"\bgirls?\b"),
    (BOYS,   r"\bboys?\b"),
    (KIDS,   r"\bchilds?\b|\bchildren\b|\bkids?\b|\bteens?\b|toddler|junior|youth"),
    (WOMEN,  r"\bwom[ae]n\b|\bwomens\b|\bladies\b|\blady\b|female|女士"),
    (MEN,    r"\bm[ae]n\b|\bmens\b|male|\bgents?\b|男士"),
    (UNISEX, r"unisex|\badults?\b|\ball\b"),
]
RULES = [(group, re.compile(pat, re.I)) for group, pat in RULES]

# level-2 category node -> audience
LEVEL2 = {
    "women": WOMEN,
    "men": MEN,
    "girls": GIRLS,
    "boys": BOYS,
    "baby": BABY,
}

# "Unisex-adult (luggage only)" -- the parenthetical is never the audience
_PAREN = re.compile(r"\([^)]*\)")
# multi-valued Department is 0.09% of rows; take the first listed audience
_SPLIT = re.compile(r"[,;/&]|\band\b")


def _normalise(raw):
    """Lowercase, drop parentheticals, and keep only the first listed value."""
    text = _PAREN.sub(" ", str(raw))
    text = text.replace("’", "'").replace("_", "-")
    text = re.sub(r"['\"]", "", text)
    text = _SPLIT.split(text, maxsplit=1)[0]
    return re.sub(r"[-]+", " ", text).strip().lower()


def _match(text):
    for group, pat in RULES:
        if pat.search(text):
            return group
    return None


def audience(row):
    """row -> (group, source). source is 'department' | 'categories' | 'default'."""
    department = (row.get("details") or {}).get("Department")
    if department:
        group = _match(_normalise(department))
        if group:
            return group, "department"
        # junk values (Belts, Watches, Luggage, Bike, Apparel, ...) fall through

    path = row.get("categories") or []
    if len(path) > 1:
        group = LEVEL2.get(path[1].strip().lower())
        if group:
            return group, "categories"

    return UNISEX, "default"
