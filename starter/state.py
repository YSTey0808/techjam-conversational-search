"""OWNER B -- fold an Extraction into the running SessionState.

    update(state, extraction) -> SessionState          <- the only public name

Three operations, in the order a conversation actually needs them:

    wipe     the customer retracted something: drop the older constraints
             that contradict what they just said
    replace  same attribute, new value ("black" then "white"): swap, because
             a person has one favourite colour at a time
    append   genuinely new information: add it

Attributes the customer declined are marked dead so ask.py never asks again.

user_profile is SOFT SIGNAL ONLY. It is never turned into a constraint and
never filters a candidate out -- it may only nudge ordering. A profile-derived
hard filter would start excluding correct answers for no gain.
"""

from __future__ import annotations

from starter import preprocessing
from starter.schema import Constraint, Extraction, SessionState

# A person has one colour, one size, one budget in mind at a time; a new value
# replaces the old. Everything else accumulates.
_SINGLE_VALUED = {"color", "size", "budget", "material"}

# Category inference, built lazily per index.
_TOKEN_INDEX: dict[int, dict[str, set[str]]] = {}
_MIN_CATEGORY_TOKENS = 1


def _token_index(prep) -> dict[str, set[str]]:
    """word -> coarse categories containing it. Built once per index."""
    cached = _TOKEN_INDEX.get(id(prep))
    if cached is not None:
        return cached
    index: dict[str, set[str]] = {}
    for category in prep.cat_index:
        for token in category.lower().replace("&", " ").replace(",", " ").split():
            if len(token) > 2:
                index.setdefault(token, set()).add(category)
    _TOKEN_INDEX[id(prep)] = index
    return index


def _infer_category(state: SessionState, extraction: Extraction) -> str | None:
    """Guess the coarse category the customer is shopping in.

    Free-form messages do not announce a category, so we look for category
    words in what they said. Deliberately conservative: a wrong category is a
    soft gate that costs recall, so we only commit on a clear winner.
    """
    prep = preprocessing.active()
    if prep is None:
        return None
    words: set[str] = set()
    for constraint in extraction.constraints:
        words.update(t for t in constraint.text.lower().split() if len(t) > 2)
    if not words:
        return None

    index = _token_index(prep)
    scores: dict[str, int] = {}
    for word in words:
        for category in index.get(word, ()):
            scores[category] = scores.get(category, 0) + 1
    if not scores:
        return None
    best = max(scores.values())
    if best < _MIN_CATEGORY_TOKENS:
        return None
    winners = [c for c, n in scores.items() if n == best]
    # Ambiguous match: prefer the most specific (smallest) bucket, and only if
    # it is genuinely small. Otherwise stay uncommitted.
    winners.sort(key=lambda c: len(prep.cat_index.get(c, ())))
    if len(winners) > 1 and len(prep.cat_index.get(winners[0], ())) > 2000:
        return None
    return winners[0]


def _same(a: Constraint, b: Constraint) -> bool:
    """Two constraints we should not hold twice."""
    if a.key and b.key:
        return a.key == b.key
    return a.text.strip().lower() == b.text.strip().lower()


def _wipe(state: SessionState, incoming: list[Constraint]) -> None:
    """Drop older constraints contradicted by what the customer just said."""
    attributes = {c.attribute for c in incoming}
    state.constraints = [
        held for held in state.constraints
        if held.attribute not in attributes or held in incoming
    ]


def _replace_or_append(state: SessionState, constraint: Constraint) -> None:
    for position, held in enumerate(state.constraints):
        if _same(held, constraint):
            state.constraints[position] = constraint      # refresh turn/weight
            return
        if constraint.attribute in _SINGLE_VALUED and held.attribute == constraint.attribute:
            state.constraints[position] = constraint      # replace
            return
    state.constraints.append(constraint)


def update(state: SessionState, extraction: Extraction) -> SessionState:
    """Apply one Extraction. Mutates and returns the same state object."""
    if state is None:
        state = SessionState()
    if extraction is None:
        return state

    state.turn = max(state.turn, min(c.turn for c in extraction.constraints)) \
        if extraction.constraints else state.turn
    if extraction.intent:
        state.scenario = extraction.intent

    if extraction.no_preference:
        state.dead_attributes.add(extraction.no_preference)

    if extraction.override and extraction.constraints:
        _wipe(state, extraction.constraints)

    for constraint in extraction.constraints:
        _replace_or_append(state, constraint)

    if state.category is None:
        state.category = _infer_category(state, extraction)

    return state
