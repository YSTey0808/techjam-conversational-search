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
from starter.schema import REACHABLE_ATTRIBUTES, SessionState, TurnPolicy

# ask.py owns list width. The harness convention is a top-10 list; if that ever
# changes, change it here -- it is the only place that decides how many
# recommendations we return.
_LIST_WIDTH = 10

# Below this, no attribute splits the pool usefully, so we say nothing and let
# the customer lead rather than burning a turn on a useless question.
_MIN_SPLIT = 0.05

# The fallback question when no attribute splits the pool. See decide().
_OTHER_ATTRIBUTE = "other"

# The shared team config the module docstring asks for, rather than a local
# copy: schema.REACHABLE_ATTRIBUTES = material, color, style, use_case, budget,
# feature.
#
# "category" and "brand" are deliberately NOT here even though the slot table
# covers both (100% and 99.4%). They would win on split quality every turn --
# brand has 19,745 distinct values, so its normalized entropy is near maximal --
# and evaluator.classify_constraint can never return either label, so the
# customer replies "I don't have an additional preference for brand" every
# time. A question that cannot be answered is a spent turn.
#
# "size" and "other" have no column in catalog_normalised.jsonl, so they score
# 0.0 and could never be picked anyway. "other" still reaches the customer, via
# the no-split fallback in decide().
_ASK_ATTRIBUTES = REACHABLE_ATTRIBUTES

# catalog_normalised.jsonl also has audience and region; ignore them for ask
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


def _known_attributes(state: SessionState) -> set[str]:
    """Attributes already filled by query/reply extraction."""
    known = set()
    for attribute in _ASK_ATTRIBUTES:
        slot = state.slots.get(_slot_name(attribute))
        # Slot.filled is exactly the "has the customer told us something" test
        # this used to hand-roll against the old dict shape.
        if slot is not None and slot.filled:
            known.add(attribute)
    return known


def _asked_attributes(state: SessionState) -> set[str]:
    """Attributes already asked, plus the ones the customer has ruled out.

    Both are permanent: re-asking something already answered, or something the
    customer said they have no preference on, buys nothing and costs a turn.
    """
    return set(state.asked) | set(state.dead_attributes)


def _candidates(state: SessionState) -> list[str]:
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


def decide(prep: Preprocessing, state: SessionState, pool: list[str], turn: int) -> TurnPolicy:
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

    if best is not None:
        # Without this the same attribute wins every turn: _candidates() reads
        # state.asked, and nothing else in the pipeline writes it.
        state.note_asked(best)
    else:
        # Nothing splits the pool. Saying nothing is not neutral -- the
        # simulator answers ask_attribute=None with "Ask me about one specific
        # attribute", which discloses nothing at all. "other" is answerable:
        # it skips the attribute classifier and returns the next two
        # undisclosed requirements verbatim, whatever type they are.
        #
        # Deliberately NOT noted as asked. An intent card holds four
        # requirements and each "other" drains two, so the second one still pays.
        best = _OTHER_ATTRIBUTE

    return TurnPolicy(
        ask_attribute=best,
        list_width=_LIST_WIDTH,
        message=_message(best),
    )
