"""Map a catalog row to one of 8 product families.

Resolution order (first hit wins):
  1. category path, walked deepest -> shallowest  (most specific node wins)
  2. path prefix fallbacks for buckets that never name the family
  3. title keywords
  4. OTHER
"""
import re

CLOTHING   = "Clothing"
SHOES      = "Shoes"
JEWELRY    = "Jewelry"
WATCHES    = "Watches"
BAGS       = "Bags & Luggage"
ACCESSORIES= "Accessories"
COSTUMES   = "Costumes"
OTHER      = "Other"

FAMILIES = [CLOTHING, SHOES, JEWELRY, WATCHES, BAGS, ACCESSORIES, COSTUMES, OTHER]

# --- 1. category node name -> family -------------------------------------
# Checked deepest-first, so a specific leaf beats a vague ancestor.
NODE = {
    # jewelry-adjacent things that are NOT jewelry (must win over "Jewelry")
    "jewelry boxes & organizers": ACCESSORIES,
    "jewelry accessories": ACCESSORIES,
    "jewelry towers": ACCESSORIES,
    "shoe care & accessories": ACCESSORIES,
    "shoe & boot trees": ACCESSORIES,
    "shoelaces": ACCESSORIES,
    "ice & snow grips": ACCESSORIES,
    "watch accessories": ACCESSORIES,
    "watchbands": ACCESSORIES,
    "shoe, jewelry & watch accessories": ACCESSORIES,

    # bags
    "handbags & wallets": BAGS,
    "luggage & travel gear": BAGS,
    "luggage": BAGS,
    "backpacks": BAGS,
    "travel duffels": BAGS,
    "suitcases": BAGS,
    "carry-ons": BAGS,
    "wallets": BAGS,
    "crossbody bags": BAGS,
    "shoulder bags": BAGS,
    "totes": BAGS,
    "clutches & evening bags": BAGS,
    "diaper bags": BAGS,
    "gym bags": BAGS,
    "travel accessories": BAGS,
    "packing organizers": BAGS,
    "luggage tags": BAGS,

    # costumes
    "costumes & accessories": COSTUMES,
    "costumes & cosplay apparel": COSTUMES,
    "costumes": COSTUMES,
    "wigs": COSTUMES,
    "masks": COSTUMES,

    # core families
    "watches": WATCHES,
    "wrist watches": WATCHES,
    "jewelry": JEWELRY,
    "shoes": SHOES,
    "boot shop": SHOES,
    "clothing": CLOTHING,
    "sport specific clothing": CLOTHING,
    "uniforms, work & safety": CLOTHING,
    "accessories": ACCESSORIES,
    "novelty & more": CLOTHING,
}

# --- 2. path prefix fallbacks --------------------------------------------
# For paths whose level-2 bucket implies a family but never spells it out.
PREFIX = {
    "boot shop": SHOES,
    "sport specific clothing": CLOTHING,
    "costumes & accessories": COSTUMES,
    "luggage & travel gear": BAGS,
    "shoe, jewelry & watch accessories": ACCESSORIES,
}

# --- 3. title keywords ----------------------------------------------------
# Order no longer decides the winner (last match in the title does), but keep
# the specific families first so ties resolve sensibly.
TITLE_RULES = [
    (COSTUMES, r"\b(costumes?|cosplay|wig|halloween)\b"),
    (WATCHES,  r"\b(wrist ?watch(es)?|smartwatch(es)?|watch(es)?)\b"),
    (JEWELRY,  r"\b(necklace|earrings?|bracelet|anklet|brooch|pendant|charms?|"
               r"cufflinks?|(engagement|wedding|signet) ring|rings?)\b"),
    (BAGS,     r"\b(backpack|handbag|purse|tote|satchel|clutch|duffel|duffle|"
               r"suitcase|luggage|wallet|crossbody|messenger bag|bag|briefcase)\b"),
    (SHOES,    r"\b(shoes?|sneakers?|boots?|booties?|sandals?|loafers?|slippers?|"
               r"clogs?|mules?|espadrilles?|moccasins?|oxfords?|pumps?|heels?|"
               r"flats?|flip[- ]?flops?|cleats?|trainers?)\b"),
    (ACCESSORIES, r"\b(sunglasses|eyeglasses|belts?|scarf|scarves|gloves?|mittens?|"
               r"hats?|caps?|beanie|headband|neckties?|bow ?tie|suspenders?|"
               r"umbrella|keychain|shoelaces?|insoles?)\b"),
    (CLOTHING, r"\b(shirts?|tees?|t-shirts?|tank ?tops?|blouses?|dress(es)?|"
               r"pants?|trousers?|jeans?|shorts?|skirts?|leggings?|jackets?|"
               r"coats?|hoodies?|sweatshirts?|sweaters?|cardigans?|pullovers?|"
               r"vests?|socks?|bras?|underwear|panties|briefs|boxers?|lingerie|"
               r"pajamas?|pyjamas?|sleepwear|robes?|swimsuits?|swimwear|bikinis?|"
               r"jumpsuits?|rompers?|overalls?|uniforms?|aprons?|kilts?|tunics?|"
               r"onesies?|bodysuits?|apparel|tights|stockings)\b"),
]
TITLE_RULES = [(fam, re.compile(pat, re.I)) for fam, pat in TITLE_RULES]

# Everything after these reads as an add-on, not the product itself.
MODIFIER = re.compile(r"\s+(?:with|w/)\s+", re.I)


def product_family(row):
    """row -> (family, source). source is 'path' | 'prefix' | 'title' | 'none'."""
    path = [c.strip().lower() for c in (row.get("categories") or [])]

    # 1. deepest -> shallowest, skipping the useless root
    for node in reversed(path[1:]):
        if node in NODE:
            return NODE[node], "path"

    # 2. level-2 bucket fallback
    for node in path[1:]:
        if node in PREFIX:
            return PREFIX[node], "prefix"

    # 3. title -- last match wins (the head noun in an English product title
    #    is usually last), after dropping trailing "with <accessory>" modifiers
    #    so "A-Line Dress with Belt" reads as a dress, not a belt.
    title = MODIFIER.split(row.get("title") or "", maxsplit=1)[0]
    best = None
    for fam, pat in TITLE_RULES:
        for m in pat.finditer(title):
            if best is None or m.start() > best[0]:
                best = (m.start(), fam)
    if best:
        return best[1], "title"

    return OTHER, "none"
