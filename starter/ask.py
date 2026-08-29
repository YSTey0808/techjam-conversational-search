"""OWNER E -- choose what to ask next, and what to say.

    decide(prep, state, pool, turn) -> TurnPolicy       <- the only public name

Two resolved constraints is the threshold that collapses the pool, so the
objective is to reach it in as few turns as possible. That means asking about
the attribute that best SPLITS the current pool, not the one that sounds most
natural: an attribute every candidate shares tells us nothing.

Split quality = coverage * Gini impurity of the value distribution.
  coverage   fraction of the pool that has any value for the attribute --
             asking about something only 3% of candidates mention is wasted
  impurity   1 - sum(p^2); maximal when values are evenly spread, zero when
             every candidate gives the same answer

NEVER ask "brand" or "category". They are not in REACHABLE_ATTRIBUTES and must
not be added: category is already inferred from what the customer said, and
brand is not something our index can act on as a constraint.
"""

from __future__ import annotations

from starter.preprocessing import Preprocessing
from starter.schema import REACHABLE_ATTRIBUTES, SessionState, TurnPolicy

# ask.py owns list width. The harness convention is a top-10 list; if that ever
# changes, change it here -- it is the only place that decides how many
# recommendations we return.
_LIST_WIDTH = 10

# Below this, no attribute splits the pool usefully, so we say nothing and let
# the customer lead rather than burning a turn on a useless question.
_MIN_SPLIT = 0.05
_MAX_POOL_SAMPLE = 400

_QUESTIONS = {
    "material": "What material are you after?",
    "color": "Any particular colour in mind?",
    "size": "What size do you need?",
    "style": "Any particular style you prefer?",
    "use_case": "What will you mainly be using it for?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Is there a specific feature you need?",
}
_DEFAULT_MESSAGE = "Here are the closest matches I have so far."


def _candidates(state: SessionState) -> list[str]:
    """Attributes still worth asking about."""
    asked = set(state.asked)
    return [
        attribute for attribute in REACHABLE_ATTRIBUTES
        if attribute not in asked and attribute not in state.dead_attributes
    ]


def _split_quality(prep: Preprocessing, pool: list[str], attribute: str) -> float:
    """How well does this attribute divide the pool? 0 = uselessly."""
    counts: dict[str, int] = {}
    covered = 0
    sample = pool[:_MAX_POOL_SAMPLE]
    for asin in sample:
        values = {
            key for key in prep.product_keys.get(asin, ())
            if prep.attribute_of(key) == attribute
        }
        if not values:
            continue
        covered += 1
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    if not covered or not counts:
        return 0.0

    total = sum(counts.values())
    impurity = 1.0 - sum((n / total) ** 2 for n in counts.values())
    coverage = covered / len(sample)
    return coverage * impurity


def _message(attribute: str | None, state: SessionState) -> str:
    """The sentence we say back.

    Template for now. SEAM: an LLM could generate this from `state` for a more
    natural turn -- it is isolated here so swapping it in touches nothing else.
    Whatever replaces it must still return a plain `str`: the contract discards
    the entire turn if `message` is not a string.
    """
    if attribute is None:
        return _DEFAULT_MESSAGE
    return _QUESTIONS.get(attribute, _DEFAULT_MESSAGE)


def decide(prep: Preprocessing, state: SessionState, pool: list[str], turn: int) -> TurnPolicy:
    """Pick this turn's question and list width."""
    best: str | None = None
    best_score = _MIN_SPLIT

    if pool:
        for attribute in _candidates(state):
            score = _split_quality(prep, pool, attribute)
            if score > best_score:
                best, best_score = attribute, score

    if best is not None:
        state.asked.append(best)

    return TurnPolicy(
        ask_attribute=best,
        list_width=_LIST_WIDTH,
        message=_message(best, state),
    )
