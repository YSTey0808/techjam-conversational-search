"""Column definitions for `style` and `use_case`.

Both columns come from a single Batch API call -- one request returns both
lists -- so they share one module, one prompt and one schema. Splitting them
into two files would imply they can be generated independently; they cannot,
and doing so would double the cost.

Both are closed enums, enforced in the JSON schema itself (`"enum": [...]` on
the array items), not merely requested in the prompt. That is what stops this
becoming `details.Style`, which holds 283 distinct values across 466 rows,
half of them not styles.

Empty is a first-class answer. Models fill by default, so the system prompt
makes "not stated" the expected output and the schema sets no `minItems`.
Colour and material are ~64% / ~25% empty; these should look similar. A
near-100% fill rate means the model is inventing labels.

Model: claude-haiku-4-5, the tier costed in CATALOG_NOTES.md. Swap MODEL for
`claude-opus-5` if label quality matters more than the ~5x cost difference.

Submitting, polling and collecting live in dataset_normalisation/pipeline.py -- that is the
only entry point.
"""
import json

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MODEL = "claude-haiku-4-5"

# --- enums, mined from title/features token frequencies on the 12k sample ---
STYLE = [
    "casual", "formal", "business", "elegant", "vintage", "athletic",
    "boho", "minimalist", "classic", "streetwear", "romantic", "western",
]
USE_CASE = [
    "everyday", "work", "formal-event", "party", "wedding", "sport",
    "outdoor", "beach", "travel", "gift", "holiday", "school",
    "lounge", "costume",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "style": {
            "type": "array",
            "items": {"type": "string", "enum": STYLE},
            "description": "Aesthetic of the product. Empty if the listing does not say. At most 2.",
        },
        "use_case": {
            "type": "array",
            "items": {"type": "string", "enum": USE_CASE},
            "description": "Occasions or activities the listing explicitly markets it for. "
                           "Empty if it does not say. At most 3.",
        },
    },
    "required": ["style", "use_case"],
    "additionalProperties": False,
}

SYSTEM = f"""You label Amazon Clothing, Shoes & Jewelry listings for a product search index.

Return two lists, drawn ONLY from these closed vocabularies:

style:    {', '.join(STYLE)}
use_case: {', '.join(USE_CASE)}

Rules:
1. Label only what the listing actually states or unambiguously implies from
   the product itself. Do not infer from the brand, the price, or your own
   sense of what the product is probably for.
2. An empty list is the correct answer when the listing does not say. Most
   listings state neither a style nor an occasion -- returning [] for both is
   normal and expected, not a failure.
3. Prefer fewer, well-supported labels. At most 2 style and 3 use_case, and
   usually fewer. If more than three occasions seem plausible, that is a sign
   the listing states none of them -- return [] instead of listing them all.
4. "gift" only when the listing markets it as a gift (gift box, "gift for
   her", holiday gifting). Do not add it because an item could be gifted.
5. "everyday" ONLY when the listing literally says everyday / every day /
   daily / casual wear. Do not label "everyday" because a product is the kind
   of thing a person happens to wear often -- a clog, a pant, a slipper, a
   t-shirt are NOT "everyday" unless the listing says so. This is the most
   common mistake: when unsure, leave it out.

Examples:

"Women's Vintage 1950s Rockabilly Swing Dress for Cocktail Party"
-> style: ["vintage"], use_case: ["party", "formal-event"]

"Mens Moisture Wicking Athletic Crew Socks 6-Pack, Cushioned"
-> style: ["athletic"], use_case: ["sport"]

"ZJ Clothes Women Plus Size Camisole Strappy Swing TOP Cami Vest"
-> style: [], use_case: []

"Sterling Silver Cross Pendant Necklace, 18 inch chain"
-> style: [], use_case: []

"Crocs Unisex-Adult Men's and Women's Baya Graphic Clog"
-> style: [], use_case: []
   (a clog is worn often, but the listing never says so -- do not infer)

"Koinshha Women's Knitted Turtleneck Sweater Long Sleeve Solid Color Pullover"
-> style: [], use_case: []
   (a plain garment with no stated occasion -- do not list every plausible one)
"""


# The API rejects `maxItems` in output_config schemas, so the cap the prompt
# asks for is enforced here instead -- kept because "prefer fewer" as advice
# alone produced rows with 6-7 use_case labels.
MAX_STYLE = 2
MAX_USE_CASE = 3


def normalise_labels(data):
    """Clamp a model response to the enums and the per-column caps."""
    style = [x for x in (data.get("style") or []) if x in STYLE][:MAX_STYLE]
    use_case = [x for x in (data.get("use_case") or []) if x in USE_CASE][:MAX_USE_CASE]
    return {"style": style, "use_case": use_case}


def slim(row):
    """Trim a catalog row to the fields that carry style/occasion signal.

    description[] is dropped: CATALOG_NOTES measured 314 tok/product with it
    and 161 without, for signal that title + features already carry.
    """
    return {
        "title": row.get("title") or "",
        "features": (row.get("features") or [])[:8],
        "categories": (row.get("categories") or [])[1:],
        "store": row.get("store") or "",
    }


def build_request(row):
    return Request(
        custom_id=row["parent_asin"],
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": SYSTEM,
                # identical across every request in the batch
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": json.dumps(slim(row), ensure_ascii=False),
            }],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        ),
    )
