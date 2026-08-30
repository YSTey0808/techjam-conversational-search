"""OWNER E -- choose what to ask next, and what to say.

    decide(prep, state, pool, turn) -> TurnPolicy       <- the only public name

Split quality = normalized entropy over KNOWN values x known_ratio.

  entropy       Shannon entropy of the value distribution, computed only over
                products where the attribute is actually known. A product
                with no value for the attribute is missing data, not a value
                in its own right, so it never becomes its own bucket and
                never dilutes the entropy of the ones we do know about.
  known_ratio   known / pool size. This is where "we don't have data on this"
                gets penalized -- separately from entropy, not mixed into it.
                This follows a C4.5-style missing-value discount: compute
                split quality from known values, then scale it by the fraction
                of the pool where the attribute is known.

_ASK_ATTRIBUTES is temporary wiring. The final allowed ask list should come
from the shared API/team config, and product values should come from the
catalog slot table produced by preprocessing.
"""

from __future__ import annotations

import math

from starter.preprocessing import Preprocessing
from starter.schema import TurnPolicy

# ask.py owns list width. The harness convention is a top-10 list; if that ever
# changes, change it here -- it is the only place that decides how many
# recommendations we return.
_LIST_WIDTH = 10

# Below this, no attribute splits the pool usefully, so we say nothing and let
# the customer lead rather than burning a turn on a useless question.
_MIN_SPLIT = 0.05
_ASK_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "brand",
    "budget",
    "style",
    "feature",
    "use_case",
    "other",
)
# catalog_normalised.xlsx also has audience and region; ignore them for ask
# selection because they are not API ask_attributes.
# API ask_attribute is "other"; the extracted catalog/session slot is "others".
_STATE_SLOT_NAME = {"other": "others"}
_MISSING_SLOT_VALUES = {"", "unknown", "null", "none", "n/a", "na"}

_QUESTIONS = {
    "material": "What material are you after?",
    "color": "Any particular colour in mind?",
    "size": "What size do you need?",
    "style": "Any particular style you prefer?",
    "use_case": "What will you mainly be using it for?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Is there a specific feature you need?",
    "brand": "Do you have a brand preference?",
    "category": "What type of item are you looking for?",
    "other": "Any other requirement I should consider?",
}
_DEFAULT_MESSAGE = "Here are the closest matches I have so far."


def _slot_name(attribute: str) -> str:
    return _STATE_SLOT_NAME.get(attribute, attribute)


def _known_attributes(state: dict) -> set[str]:
    """Attributes already filled by query/reply extraction."""
    known = set()
    slots = state.get("slots", {})
    if not isinstance(slots, dict):
        return known

    for attribute in _ASK_ATTRIBUTES:
        slot = slots.get(_slot_name(attribute), {})
        if not isinstance(slot, dict):
            continue
        value = slot.get("val")
        if value not in (None, "", []):
            known.add(attribute)
    return known


def _asked_attributes(state: dict) -> set[str]:
    """Attributes already asked in this intent context."""
    clarification = state.get("clarification", {})
    if not isinstance(clarification, dict):
        return set()

    asked = clarification.get("asked_attributes", [])
    return {str(value) for value in asked} if isinstance(asked, list) else set()


def _candidates(state: dict) -> list[str]:
    """Attributes still worth asking about."""
    asked = _asked_attributes(state)
    known = _known_attributes(state)
    return [
        attribute for attribute in _ASK_ATTRIBUTES
        if attribute not in known and attribute not in asked
    ]


def _slot_values(prep: Preprocessing, asin: str, attribute: str) -> tuple[str, ...]:
    """Read normalized slot values for one product and ask attribute.

        prep.product_slots[asin][attribute] -> list[str] | str | None

    ask.py should not infer attributes from raw title/features/details, regexes,
    or product_keys. It should only consume the final slot table:

        category, material, color, size, brand, budget, style, feature,
        use_case, other

    Current extracted catalog columns not used for asking: audience, region.
    API attributes not directly present as an Excel column: other/others.
    Comma-separated Excel cells are treated as multiple slot values.

    Empty/null means "unknown from catalog extraction", not "does not match".
    The literal string "unknown" from extracted slots, especially budget, is
    also treated as missing instead of a real entropy bucket.
    Unknown values are excluded from entropy and counted only through the
    known-ratio denominator in _split_quality().
    """
    product_slots = getattr(prep, "product_slots", None)
    if not isinstance(product_slots, dict):
        return ()

    slots = product_slots.get(asin, {})
    if not isinstance(slots, dict):
        return ()

    raw = slots.get(_slot_name(attribute))
    if raw in (None, "", []):
        return ()

    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = []
        for value in raw:
            if value in (None, ""):
                continue
            values.extend(str(value).split(","))
    else:
        values = [str(raw)]

    normalized = []
    seen = set()
    for value in values:
        text = " ".join(value.strip().lower().split())
        if text not in _MISSING_SLOT_VALUES and text not in seen:
            seen.add(text)
            normalized.append(text)
    return tuple(normalized)


def _split_quality(prep: Preprocessing, pool: list[str], attribute: str) -> float:
    """How well would asking about this attribute divide the pool? 0 = uselessly."""
    counts: dict[str, float] = {}
    known = 0
    for asin in pool:
        values = _slot_values(prep, asin, attribute)
        if not values:
            # Missing slot: keep the product in the pool, but do not create an
            # "unknown" bucket. Missingness is handled by known_ratio below.
            continue
        known += 1
        # Multi-value slots share one product's weight across their values.
        share = 1.0 / len(values)
        for value in values:
            counts[value] = counts.get(value, 0.0) + share

    if known == 0 or len(counts) < 2:
        # No split if nobody has the slot, or everyone has the same value.
        return 0.0

    # Shannon entropy over known slot values only.
    entropy = -sum((n / known) * math.log2(n / known) for n in counts.values())
    # Normalize so attributes with many possible values do not automatically win.
    normalized_entropy = entropy / math.log2(len(counts))

    # Missing-value penalty: sparse slots are useful, but less reliable.
    known_ratio = known / len(pool)
    return normalized_entropy * known_ratio


def _message(attribute: str | None) -> str:
    """The sentence we say back.

    Fixed templates are enough here; ask_attribute is the structured decision.
    """
    if attribute is None:
        return _DEFAULT_MESSAGE
    return _QUESTIONS.get(attribute, _DEFAULT_MESSAGE)


def decide(prep: Preprocessing, state: dict, pool: list[str], turn: int) -> TurnPolicy:
    """Pick this turn's question and list width."""
    if turn >= 10:
        return TurnPolicy(
            ask_attribute=None,
            list_width=_LIST_WIDTH,
            message=_message(None),
        )

    best: str | None = None
    best_score = _MIN_SPLIT

    if pool:
        for attribute in _candidates(state):
            score = _split_quality(prep, pool, attribute)
            if score > best_score:
                best, best_score = attribute, score

    return TurnPolicy(
        ask_attribute=best,
        list_width=_LIST_WIDTH,
        message=_message(best),
    )
