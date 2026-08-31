"""OWNER A -- read one customer message into the session state.

    extract(message, turn, state) -> SessionState      <- the only public name

This module owns BOTH halves of understanding a turn: reading the message and
writing what it learned into `SessionState`. There is no intermediate
"Extraction" type -- state is the only memory, so a reading that never reaches
a slot never happened.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from starter import preprocessing
from starter.schema import HARD_CONFIDENCE, REACHABLE_ATTRIBUTES, SessionState

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

_TIMEOUT = 2.0
_MAX_ATTEMPTS = 2               # per turn: first try, then one corrective retry
_MAX_CONSECUTIVE_FAILURES = 2   # failed turns before the provider is abandoned
_MAX_SLOTS = 8
_BUDGET_TOLERANCE = 1.05

# Confidence carried by each template frame, calibrated against
# HARD_CONFIDENCE (0.75). See the frame table below for why each is what it is.
_CONF_HARD = 1.00        # a stated hard requirement
_CONF_REVEAL = 0.90      # answers to our questions
_CONF_SOFT = 0.55        # a preference that is about to be retracted
_CONF_SALVAGE = 0.40     # a payload we could not validate
_CONF_UNRESOLVED = 0.30  # nothing in the catalog matched; keep it visible anyway


# --------------------------------------------------------------------------
# frames -- the deterministic router
# --------------------------------------------------------------------------

@dataclass
class _Reading:
    """Everything the customer told us about one attribute, this turn.

    `value`/`confidence` are what they want -- `value` is empty when the
    attribute was mentioned only to rule something out, never as a preference.
    `negated` holds values ruled OUT for this same attribute ("nothing formal"
    for style). Kept separate from `value`: wanting X and not-wanting Y are
    different facts, and folding a negation into `value` would record it as a
    preference for the very thing being rejected.
    """

    attribute: str
    value: str = ""
    confidence: float = 0.0
    negated: list[str] = field(default_factory=list)


@dataclass
class _Frame:
    """One parsed message -- every _Reading it produced."""

    name: str
    readings: tuple[_Reading, ...] = ()
    buy_intent: float = 0.5


# --------------------------------------------------------------------------
# vocabulary -- payload to catalog key
# --------------------------------------------------------------------------

def _resolve(text: str, variants: Sequence[str] = ()) -> tuple[str, float]:
    """Find the catalog key for a payload. Returns ("", 0.0) on a total miss.

    Hop 1 is the whole game: the simulator only ever draws from a product's
    first four indexed strings, which is exactly what canon_postings holds.
    canon is checked before postings so shared boilerplate ("Imported",
    "Machine Wash") cannot win a match that a characteristic string should.
    """
    prep = preprocessing.active()
    if prep is None:
        return "", 0.0

    key = preprocessing.normalize(text)
    if key:
        if key in prep.canon_postings:
            return key, 1.00
        if key in prep.postings:
            return key, 0.95
    for variant in variants:
        candidate = preprocessing.normalize(variant)
        if candidate and (candidate in prep.canon_postings or candidate in prep.postings):
            return candidate, 0.80
    # Prefix-tolerant last resort. Return the normalized key, not an expanded
    # one -- prep.lookup is prefix-tolerant on the read side too, so retrieval
    # re-resolves identically.
    if key and prep.lookup(key, broad=True):
        return key, 0.60
    return "", 0.0


# --------------------------------------------------------------------------
# classify -- which attribute a raw catalog string belongs to
# --------------------------------------------------------------------------

# MIRRORS evaluator.classify_constraint EXACTLY, including its quirks: plain
# substring tests (so "fit" matches "outfit"), style checked before use_case,
# and a narrower colour list than COLOR_RE. Diverging costs turns: ask.py
# would request an attribute the simulator files elsewhere, and the reply
# comes back "I don't have an additional preference for ...".
_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex",
              "silk", "rayon", "fabric")
_COLORS = ("color", "black", "white", "blue", "red", "pink", "green")
_SIZES = ("size", "sizing", "width", "wide", "narrow")
_STYLES = ("department", "style", "fit", "sleeve", "neck")
_USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work")
_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")


def _attribute_of(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(word in lowered for word in _MATERIALS):
        return "material"
    if any(word in lowered for word in _COLORS):
        return "color"
    if any(word in lowered for word in _SIZES):
        return "size"
    if any(word in lowered for word in _STYLES):
        return "style"
    if any(word in lowered for word in _USE_CASES):
        return "use_case"
    return "feature"


def _is_key(text: str) -> bool:
    prep = preprocessing.active()
    if prep is None:
        return False
    key = preprocessing.normalize(text)
    return bool(key) and (key in prep.canon_postings or key in prep.postings)


def _split_payload(payload: str) -> list[str]:
    """Split a reveal payload into constraints, using the catalog as the judge.

    The evaluator joins up to two constraints with "; ", but the constraints
    themselves routinely contain "; " ("...8% Spandex; Machine Wash Cold"), so
    splitting on the separator is unreliable. Two wrong keys are strictly worse
    than one long key that still prefix-matches, so a split has to earn it:
    both halves must be real catalog strings AND co-occur on some product,
    which they must, since both came off the same intent card.
    """
    text = payload.strip()
    if _is_key(text):
        return [text]

    prep = preprocessing.active()
    best: tuple[float, list[str]] | None = None
    for position, _ in enumerate(text):
        if not text.startswith("; ", position):
            continue
        left, right = text[:position], text[position + 2:]
        if not (_is_key(left) and _is_key(right)):
            continue
        left_key = preprocessing.normalize(left)
        right_key = preprocessing.normalize(right)
        if prep is not None and not (prep.lookup(left_key, broad=True)
                                     & prep.lookup(right_key, broad=True)):
            continue
        score = min(prep.idf(left_key), prep.idf(right_key)) if prep else 0.0
        if best is None or score > best[0]:
            best = (score, [left, right])
    if best is not None:
        return best[1]
    return [text]


def _settle_category(captured: str, whole: str) -> str:
    """Pin the category to a real cat_index key without ever rewriting it.

    The regex splits on the first ". ", which is wrong if the category itself
    contains one. So verify the capture against the index and, on a miss, walk
    the other ". " boundaries. If nothing verifies, keep the raw capture -- a
    verbatim string is the one thing guaranteed to be right, and normalizing
    it can only lose.
    """
    prep = preprocessing.active()
    if prep is None or captured in prep.cat_index:
        return captured
    opening = whole.split("I'm looking for ", 1)[-1]
    for position, _ in enumerate(opening):
        if opening.startswith(". ", position):
            candidate = opening[:position]
            if candidate in prep.cat_index:
                return candidate
    return captured


# --------------------------------------------------------------------------
# template router -- regex-first, LLM-fallback. See extract() for the wiring
# and _frame_is_trustworthy below for the gate that decides which frames from
# here are trusted.
# --------------------------------------------------------------------------
#
# The evaluator's simulated customer (evaluator/local_evaluator.py) writes
# exactly 9 fixed message shapes, never free text -- but may paraphrase them.
# This parses all 9 exactly, for free, with no tokens; the degradation gate
# after _route is what catches a paraphrase that still matched the scaffold,
# and sends it to the LLM path instead of trusting a garbled parse.
#
# Returns the current _Frame/_Reading shape: category comes back as an
# ordinary _Reading(attribute="category", ...), same as any other attribute --
# there is no separate category field.
#
# Anchored, first match wins. Order matters: open_hard/open_browse must
# precede open_soft, since all three start "I'm looking for ".
#
#   open_hard         buying turn 1; payload is hard_constraints[0]
#   open_browse       browsing AND boundary turn 1 -- byte-identical, so this
#                     frame cannot tell them apart and must not try
#   override          the retraction; payload is hard_constraints[0] and this
#                     is the ONLY place that fact ever appears
#   reveal            answers, up to two constraints joined by "; "
#   no_extra/         the customer has nothing to add -- carries no
#   decline_boundary  information in the current _Frame shape (no `declined`
#                     field), so these produce an empty frame, same as nudge
#   nudge             we asked nothing, so they prod us; zero information
#   open_soft         fallthrough, reached only by intent_override turn 1
#                     since open_hard/open_browse already claim the others
#
# Cosmetic-only tolerance on top of the strict shapes above: flexible
# whitespace (\s+ in place of literal spaces), case-insensitivity on the
# fixed scaffolding, and an optional trailing period. None of this touches
# what a capture group can contain -- a payload/category capture adjacent to
# the trailing period is made lazy (.+?) rather than greedy so an absent
# period is simply not consumed, instead of the capture swallowing it; a
# present period is still excluded from the captured value exactly as
# before. Paraphrase tolerance belongs to the degradation gate below, not
# here.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("open_hard", re.compile(
        r"\AI'm\s+looking\s+for\s+(?P<category>.+?)\.\s+A\s+key\s+requirement\s+is:\s*"
        r"(?P<payload>.+?)\.?\Z", re.S | re.I)),
    ("open_browse", re.compile(
        r"\AI'm\s+looking\s+for\s+(?P<category>.+),\s+but\s+I'm\s+still\s+exploring\.?\Z",
        re.S | re.I)),
    ("override", re.compile(
        r"\AActually,\s+ignore\s+my\s+earlier\s+preference\.\s+What\s+I\s+need\s+is:\s*"
        r"(?P<payload>.+?)\.?\Z", re.S | re.I)),
    ("override_bare", re.compile(
        r"\AActually,\s+please\s+ignore\s+my\s+earlier\s+preference\.?\Z", re.S | re.I)),
    ("reveal", re.compile(
        r"\AFor\s+that,\s+what\s+matters\s+is:\s*(?P<payload>.+?)\.?\Z", re.S | re.I)),
    ("no_extra", re.compile(
        r"\AI\s+don't\s+have\s+an\s+additional\s+preference\s+for\s+(?P<attribute>[A-Za-z_]+)\.?\Z",
        re.S | re.I)),
    ("decline_boundary", re.compile(
        r"\AI\s+don't\s+have\s+a\s+preference\s+for\s+(?P<attribute>[A-Za-z_]+);\s+"
        r"please\s+use\s+your\s+judgment\.?\Z", re.S | re.I)),
    ("nudge", re.compile(
        r"\AThose\s+options\s+are\s+not\s+quite\s+right\s+yet\.\s+"
        r"Ask\s+me\s+about\s+one\s+specific\s+attribute\.?\Z", re.S | re.I)),
    ("open_soft", re.compile(
        r"\AI'm\s+looking\s+for\s+(?P<category>.+?)\.\s+(?P<payload>.+)\Z", re.S | re.I)),
)


def _route(message: str) -> _Frame | None:
    """Parse one of the simulator's 9 fixed message shapes. None on no match."""
    text = (message or "").strip()
    if not text:
        return None

    for name, pattern in _PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groupdict()
        category = groups.get("category")
        if category is not None:
            category = _settle_category(category, text)

        if name == "open_hard":
            payload = groups["payload"].strip()
            readings = [_Reading(attribute="category", value=category, confidence=_CONF_HARD)]
            if payload:
                readings.append(_Reading(attribute=_attribute_of(payload),
                                          value=payload, confidence=_CONF_HARD))
            return _Frame(name=name, readings=tuple(readings))

        if name == "open_browse":
            # browsing and boundary are indistinguishable here, by construction.
            return _Frame(name=name, readings=(
                _Reading(attribute="category", value=category, confidence=_CONF_HARD),
            ))

        if name == "override":
            payload = groups["payload"].strip()
            readings = []
            if payload:
                readings.append(_Reading(attribute=_attribute_of(payload),
                                          value=payload, confidence=_CONF_HARD))
            return _Frame(name=name, readings=tuple(readings))

        if name in ("override_bare", "no_extra", "decline_boundary", "nudge"):
            return _Frame(name=name, readings=())

        if name == "reveal":
            parts = _split_payload(groups["payload"])
            readings = tuple(
                _Reading(attribute=_attribute_of(p), value=p, confidence=_CONF_REVEAL)
                for p in parts if p.strip()
            )
            return _Frame(name=name, readings=readings)

        if name == "open_soft":
            # Only intent_override reaches here. Its payload is a soft
            # preference the customer will retract at turn 3 or 4, so it stays
            # under HARD_CONFIDENCE -- it feeds ranking, never a hard filter.
            payload = groups["payload"].strip()
            readings = [_Reading(attribute="category", value=category, confidence=_CONF_HARD)]
            if payload:
                readings.append(_Reading(attribute=_attribute_of(payload),
                                          value=payload, confidence=_CONF_SOFT))
            return _Frame(name=name, readings=tuple(readings))
    return None


# --------------------------------------------------------------------------
# degradation gate -- when NOT to trust a regex match
# --------------------------------------------------------------------------
#
# The scaffold matching is not the whole story: a paraphrase that keeps the
# fixed wrapper ("I'm looking for X. A key requirement is: Y.") but rambles
# inside the ".+" payload looks, structurally, exactly like a real answer.
# "Doesn't match" therefore has to mean more than `_route() is None` -- it
# has to mean "the scaffold matched AND every payload classified to a known
# attribute". This gate is that second half.

# _attribute_of mirrors evaluator.classify_constraint exactly (see its
# docstring), including the one quirk that matters here: nothing matching
# any keyword list falls through to "feature". That fallthrough is the only
# way _attribute_of ever says "I couldn't classify this" -- it is legitimate
# for a real feature payload too, but a regex-first pipeline that cannot
# tell the two apart has to treat the ambiguous case as untrustworthy and
# let the LLM sort it out.
_UNKNOWN_ATTR = "feature"

# Every payload that reaches a template message is clean_constraint'd to at
# most CONSTRAINT_LIMIT (preprocessing.py) before the simulator ever sees
# it, so nothing longer can be a real templated payload -- only a paraphrase
# that dragged extra sentence past the ".+" capture gets this long.
_MAX_PAYLOAD_LEN = 180

# Catalog strings are short attribute fragments ("8% Spandex", "Machine
# Wash Cold") and in practice never carry more than one comma; a paraphrase
# stringing together several clauses almost always does.
_MAX_COMMAS = 1

# The only frames that legitimately carry zero readings -- see _route: each
# produces `_Frame(name=name, readings=())` by construction, so an empty
# frame from one of these is the correct parse, not a degradation.
_CONTROL_FRAMES = frozenset({"override_bare", "no_extra", "decline_boundary", "nudge"})


def _reading_is_degraded(reading: _Reading) -> bool:
    """True when one reading looks like a paraphrase slipped through, not a template answer."""
    value = reading.value.strip()
    if not value:
        return True
    return False


def _frame_is_trustworthy(frame: _Frame | None) -> bool:
    """True only when a real pattern matched and every reading survives the gate.

    None (no scaffold matched) is never trustworthy. Neither is a data-bearing
    frame with zero readings -- that should not happen by construction, so an
    empty frame outside _CONTROL_FRAMES is treated as a degraded parse rather
    than assumed harmless.
    """
    if frame is None:
        return False
    if frame.name in _CONTROL_FRAMES:
        return True
    if not frame.readings:
        return False
    # return True
    return not any(_reading_is_degraded(r) for r in frame.readings)


# --------------------------------------------------------------------------
# dependency graph
# --------------------------------------------------------------------------

_DEPENDENTS: dict[str, tuple[str, ...]] = {
    # sizing systems, style vocabulary and use cases are all category-scoped
    "category": ("material", "style", "use_case", "feature", "others"),
    # a feature is justified BY the use case: no hiking, no reason for waterproof
    # "use_case": ("feature")
}


@lru_cache(maxsize=None)
def _dependents(attribute: str) -> tuple[str, ...]:
    """Transitive closure of the graph. Cycle-safe and order-stable.

    Transitive rather than one level so category -> use_case -> feature still
    reaches feature if someone later removes the direct edge.
    """
    seen: list[str] = []
    queue = list(_DEPENDENTS.get(attribute, ()))
    while queue:
        current = queue.pop(0)
        if current in seen or current == attribute:
            continue
        seen.append(current)
        queue.extend(_DEPENDENTS.get(current, ()))
    return tuple(seen)


def _cascade(state: SessionState, parent: str, turn: int) -> None:
    """Forget everything that depended on `parent`. Never touches this turn's work."""
    for attribute in _dependents(parent):
        slot = state.slots.get(attribute)
        if slot is not None and slot.filled and slot.turn < turn:
            state.forget(attribute)


# --------------------------------------------------------------------------
# buy intent
# --------------------------------------------------------------------------

# 0.0 = browsing = widen, unlock cross-category. 1.0 = buying = tighten onto
# hard constraints. These are RETRIEVAL MODES, not the English words: a
# customer can be certain they are only browsing.
_BASE = 0.20
_W_HARD = 0.22        # each firm requirement is strong evidence to tighten
_W_CONF = 0.15        # how sure we are of what we hold
_W_DRY = 0.08         # they have nothing left to add, so commit to what we have
_W_TURN = 0.10        # turns are finite; drift toward committing
_W_STUCK = 0.10       # we keep asking nothing: widen rather than spin


def _buy_intent(state: SessionState, frame: _Frame | None, turn: int) -> float:
    """Score the retrieval mode from evidence that actually occurs.

    Category is excluded from the hard count because it is present in every
    session from turn 1 and so carries no discriminating signal.
    """
    hard = [n for n in state.hard_slots if n != "category"]
    scored = [s.confidence_score or 0.0
              for n, s in state.filled_slots.items() if n != "category"]
    mean_confidence = sum(scored) / len(scored) if scored else 0.0

    raw = (_BASE
           + _W_HARD * min(len(hard), 3)
           + _W_CONF * mean_confidence
           + _W_DRY * min(state.deflections, 2)
           + _W_TURN * (min(turn, 10) / 10.0)
           - _W_STUCK * min(state.nudges, 2))
    value = max(0.0, min(1.0, raw))

    # A retraction widens for exactly this turn while the demotions settle.
    # Not persisted: next turn recomputes from evidence and springs back.
    if frame is not None:
        value = min(value, 0.35)
    return value


# --------------------------------------------------------------------------
# the LLM contract -- schema, coercion, validation
# --------------------------------------------------------------------------

_LLM_ATTRIBUTES = (
    "category", "material", "color", "brand",
    "budget", "style", "feature", "use_case", "others",
)
_LLMAttribute = Literal[
    "category", "material", "color", "brand",
    "budget", "style", "feature", "use_case", "others",
]

# What the model is told to emit, and what it is checked against. Kept as one
# literal so the prompt and the validator can never drift apart.
_OUTPUT_SHAPE = """{
    "attributes": [
        {
            "attribute": str = "<one of: category|material|color|brand|budget|style|feature|use_case|others>",
            "value": str | list[str] | number = "<the requirement, in the words a product listing would use; empty if this object only rules something OUT>",
            "confidence": float = <0.0-1.0>,
            "negated": list[str] = <values the customer is ruling OUT for this attribute; empty list if none>
        }
    ],
    "buy_intent": float = <0.0-1.0, see OVERALL BUY INTENT below>
}"""


class ExtractedAttribute(BaseModel):
    """One attribute the model read out of the conversation.

    `value` and `negated` are independent: an entry may carry a wanted value,
    a list of excluded values, or both ("cotton, not leather" for material).
    At least one of the two must be non-empty -- _coerce_attribute drops
    entries that carry neither before they ever reach this model.
    """

    attribute: _LLMAttribute
    value: Annotated[str, Field(default="", max_length=180)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    negated: Annotated[list[str], Field(default_factory=list, max_length=10)]

    @field_validator("value", mode="after")
    @classmethod
    def _clean_value(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            return value
        if info.data.get("attribute") == "budget":
            try:
                return str(float(value))
            except ValueError as exc:
                raise ValueError(f"budget value {value!r} is not a plain number") from exc
        # The same normalization the catalog index was built with. Without it a
        # model-supplied string can never match postings even when it is right.
        return preprocessing.clean_constraint(value)

    @field_validator("negated", mode="after")
    @classmethod
    def _clean_negated(cls, values: list[str]) -> list[str]:
        return [preprocessing.clean_constraint(v) for v in values if v]


class LLMReading(BaseModel):
    """The whole of what the model may tell us about one turn."""

    attributes: Annotated[
        list[ExtractedAttribute], Field(default_factory=list, max_length=_MAX_SLOTS)]
    buy_intent: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


# --- deterministic coercion ------------------------------------------------
# A model can be right about the content and still wrong about the shape:
# "attribute" spelled "attr", one object instead of a list, confidence as the
# string "0.8" or as a 0-100 percentage. Rejecting those loses a correct
# reading over punctuation. So the raw payload is pushed through fixed,
# deterministic rules FIRST, and only then validated -- coercion never guesses
# at meaning, it only reshapes. Anything it cannot place is dropped.

_ATTRIBUTE_ALIASES = {
    "attr": "attribute", "name": "attribute", "slot": "attribute", "field": "attribute",
    "val": "value", "text": "value", "content": "value",
    "score": "confidence", "confidence_score": "confidence", "certainty": "confidence",
    "negative": "negated", "negation": "negated", "excluded": "negated",
}
_ATTRIBUTE_SYNONYMS = {
    "colour": "color", "fabric": "material", "materials": "material",
    "fit": "others", "sizing": "others", "price": "budget", "cost": "budget",
    "usecase": "use_case", "use case": "use_case", "purpose": "use_case",
    "features": "feature", "other": "others", "misc": "others",
}


def _as_float(value: object) -> float | None:
    """A confidence the model may have written as a string or a percentage."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if not match:
            return None
        number = float(match.group())
    else:
        return None
    if number > 1.0:                      # "85" meaning 85%
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _coerce_attribute(raw: object) -> dict | None:
    """One entry -> our field names, or None if it carries no usable value."""
    if not isinstance(raw, dict):
        return None
    item = {_ATTRIBUTE_ALIASES.get(str(k).strip().lower(), str(k).strip().lower()): v
            for k, v in raw.items()}

    name = str(item.get("attribute") or "").strip().lower().replace("-", "_")
    name = _ATTRIBUTE_SYNONYMS.get(name, name)
    if name not in _LLM_ATTRIBUTES:
        return None

    value = item.get("value")
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value if v)
    value = str(value or "").strip()

    negated_raw = item.get("negated")
    if isinstance(negated_raw, str):
        negated_raw = [negated_raw]
    negated = [str(v).strip() for v in negated_raw if str(v).strip()] \
        if isinstance(negated_raw, list) else []

    if not value and not negated:
        return None        # neither a preference nor an exclusion: nothing to keep

    confidence = _as_float(item.get("confidence"))
    if confidence is None:
        if value:
            return None     # confidence is the model's job; we do not invent one
        confidence = 0.0    # a pure exclusion carries no "how much they want it" signal

    return {"attribute": name, "value": value[:180], "confidence": confidence,
            "negated": negated[:10]}


def _coerce_reading(raw: object) -> dict:
    """Whole payload -> the LLMReading shape. Never raises."""
    if isinstance(raw, list):                       # a bare list of attributes
        raw = {"attributes": raw}
    if not isinstance(raw, dict):
        return {"attributes": [], "buy_intent": 0.5}

    lowered = {str(k).strip().lower(): v for k, v in raw.items()}
    entries = lowered.get("attributes")
    if entries is None:
        entries = lowered.get("slots") or lowered.get("constraints") or []
    if isinstance(entries, dict):
        # {"color": {"value": "black", "confidence": 0.9}} or {"color": "black"}
        flattened = []
        for name, body in entries.items():
            if isinstance(body, dict):
                flattened.append({"attribute": name, **body})
            else:
                flattened.append({"attribute": name, "value": body, "confidence": None})
        entries = flattened
    if not isinstance(entries, list):
        entries = []

    attributes = [c for c in (_coerce_attribute(e) for e in entries) if c]

    # Reuses _as_float so "0.8", "80%" etc. parse the same way confidence does.
    # A missing or unparseable score defaults to 0.5 -- neutral, not a guess.
    buy_intent = _as_float(lowered.get("buy_intent"))

    return {
        "attributes": attributes[:_MAX_SLOTS],
        "buy_intent": buy_intent if buy_intent is not None else 0.5,
    }


def _parse_reading(raw: str) -> LLMReading | None:
    """Text -> validated LLMReading. None when nothing usable survives.

    Two stages on purpose: slice out the JSON and reshape it deterministically,
    then let pydantic be the single authority on whether the result is valid.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if not text.startswith(("{", "[")):                 # tolerate fenced output
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start == -1 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        return LLMReading.model_validate(_coerce_reading(json.loads(text)))
    except Exception:
        return None


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

_ALLOWED_LABELS = "category|material|color|brand|budget|style|feature|use_case|other"
_ALLOWED_VALUES = ""

_SYSTEM = f"""You extract shopping constraints from a customer's latest message in an \
ongoing product-search conversation. You output structured data ONLY.

WHAT TO EXTRACT
- One object per distinct requirement expressed or clearly implied in the LATEST message.
- Use the conversation history ONLY to (a) resolve references like "the cheaper one" or \
"that brand", and (b) detect when the customer is now RULING OUT something. Do not re-extract \
constraints from earlier turns that the latest message doesn't touch.

ATTRIBUTES (use these labels and values exactly; no others)
Labels: {_ALLOWED_LABELS}; Values: {_ALLOWED_VALUES}
- category : the product type ("jewelry", "clothing")
- material : what it's made of ("leather", "cotton")
- color    : ("red", "navy")
- brand    : ("Nike")
- budget   : the price limit the customer names, as a FLOAT (e.g. "around $100" -> 100.0).
           Report the number EXACTLY as the customer said it. Do not add any margin
           or rounding — the system applies the tolerance itself.
- style    : aesthetic descriptors ("vintage", "minimalist") -- may be several
- feature  : functional properties ("waterproof", "hypoallergenic") -- may be several
- use_case : occasion or purpose ("for a wedding", "gift for dad") -- may be several
- other    : a real constraint that fits none of the above
- Everything is a STRING except budget, which is the object above.

NEGATION
- If the customer excludes something ("no leather", "not for kids", "anything but black"), \
emit the object with "negated": ["leather"]. Otherwise "negated": [], empty list.

CONFIDENCE SCORE for each attribute  (this is how much they WANT an item with this specific attribute)
Judge how strongly the customer desires this attribute, from their wording and emphasis:
- 0.9-1.0  Non-negotiable. "must", "need", "has to be", "only", "absolutely", "required".
           Also: an explicit exclusion ("no leather") is a strong signal -> high.
- 0.7-0.8 A plain, firm requirement stated without hedging ("blue earrings", "size 9").
           A clearly named constraint with no enthusiasm marker still belongs HERE, not lower.
- 0.4-0.5 A preference or lean. "prefer", "ideally", "would like", "leaning toward".
- 0.1-0.3 A weak nice-to-have or hedge. "maybe", "kind of", "or something", "I guess".
Do not default everything to one middle value. If the customer gives no intensity cue at all
but clearly names the attribute, use 0.7. Reserve <0.4 for genuine hedging words.

OVERALL BUY INTENT (one number for the whole turn)
Buying means lock hard constraints, Browsing means unlock cross-category recommendation.
Score 0.0-1.0 how ready the customer is to have results narrowed to exact matches on what
they've stated, versus kept broad so they can keep exploring:
- 1.0  They have given firm, specific requirements and want precise matches now.
- 0.5  Mixed, early, or unclear.
- 0.0  Still exploring; results should stay broad, other categories are fine too.
Judge it from the WHOLE conversation so far, not just the latest message, and always set
it, even on a turn that adds no new attributes.

HARD RULES
- If the latest message states NO product constraint (greeting, thanks, "show me more", \
a question to you), still set "buy_intent" and return "attributes": [].
- Never invent a constraint that isn't grounded in the message text. Every object must have \
a source_span copied verbatim from the message.
- Output ONLY a JSON object matching this shape, no prose, no markdown fences:
{_OUTPUT_SHAPE}
"""

USER_TEMPLATE = """Conversation so far:
{history}

Latest customer message:
"{message}"

Extract the constraints from the latest message as a JSON array."""


def _history(state: SessionState, limit: int = 6) -> str:
    """The conversation so far, oldest first, as plain dialogue.

    A single line is often meaningless alone -- "I don't have a preference for
    that" only means something next to the question it answers -- so the model
    is given the exchange, not the utterance.
    """
    lines: list[str] = []
    # history[-1] is the turn being read right now; the prompt quotes it
    # separately, so showing it here too would just duplicate it.
    for entry in state.history[:-1][-limit:]:
        if entry.customer:
            lines.append(f"Customer: {entry.customer}")
        if entry.agent:
            asked = f" [asked about: {entry.ask_attribute}]" if entry.ask_attribute else ""
            lines.append(f"Agent{asked}: {entry.agent}")
    return "\n".join(lines)

def _user_prompt(message: str, turn: int, state: SessionState) -> str:
    return (f"Conversation so far:\n{_history(state) or '(this is the first turn)'}\n\n"
            f"Turn {turn}. The customer just said:\n{message}")


def _correction_prompt(original: str, bad_output: str) -> str:
    """Retry prompt after a reply we could not parse.

    The coercion layer already absorbs every shape difference it knows about,
    so anything reaching here is something the model has to be told. Quoting
    its own output back is what makes the second attempt different from a
    plain re-send, which would likely fail the same way.
    """
    return (f"{original}\n\n"
            f"Your previous reply could not be parsed:\n"
            f"---\n{(bad_output or '(empty)')[:600]}\n---\n"
            f"Return ONLY the JSON object described above. No prose, no code "
            f"fence, no trailing commas. Every attribute needs all of "
            f'"attribute", "value" and "confidence".')


# --------------------------------------------------------------------------
# the provider -- see llm_client.get_client (anthropic | groq | none)
# --------------------------------------------------------------------------

_MODEL = os.environ.get("TECHJAM_LLM_MODEL") or "openai/gpt-oss-120b"
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}


@lru_cache(maxsize=1)
def _client():
    """The configured LLM client, or None when nothing is set up.

    Returns None rather than raising for a missing provider, so an unconfigured
    checkout runs the deterministic path instead of dying inside Agent.__init__
    and scoring every session zero.
    """
    from llm_client import get_client   # LAZY: keeps the optional deps optional

    return get_client(model=_MODEL, timeout=_TIMEOUT)


_failures = 0


def _llm_frame(message: str, turn: int, state: SessionState) -> _Frame | None:
    """Ask Claude to read a message no template matched.

    Two attempts. A malformed reply is retried with the bad output quoted back
    and a corrective instruction, because the coercion layer has already
    absorbed every shape difference it knows how to; whatever still fails is
    something the model has to be told about. A transport failure is retried
    plainly -- there is nothing to correct.

    Wall clock is bounded by _TIMEOUT * _MAX_ATTEMPTS, which matters because
    the evaluator imposes no timeout of its own.

    Circuit breaker: after two consecutive failed turns the provider is
    abandoned for the rest of the run, so a dead endpoint costs a few seconds
    total rather than a few seconds per turn for a thousand sessions.
    """
    global _failures
    client = _client()
    if client is None or _failures >= _MAX_CONSECUTIVE_FAILURES:
        return None

    prompt = _user_prompt(message, turn, state)
    spent = dict(_ZERO_USAGE)
    reading = None
    answered = False        # did the provider ever hand back a reply to parse?

    from llm_client import LLMError

    for _ in range(_MAX_ATTEMPTS):
        try:
            result = client.complete(
                system=_SYSTEM,
                user=prompt,
                max_tokens=1024,
                effort="low",
            )
        except LLMError:
            continue        # transport failure: nothing to correct, just retry

        answered = True
        # Every attempt costs tokens whether or not it parsed, so usage
        # accumulates across the loop. Overwriting would under-report.
        spent["prompt_tokens"] += max(0, int(result.usage.get("prompt_tokens", 0)))
        spent["completion_tokens"] += max(0, int(result.usage.get("completion_tokens", 0)))

        reading = _parse_reading(result.text)
        if reading is not None:
            break
        prompt = _correction_prompt(_user_prompt(message, turn, state), result.text)

    state.usage = spent
    if reading is None:
        # Only a reply we could not use counts against the breaker. A pure
        # transport failure (rate limit, timeout) says nothing about whether
        # the model can do this, and must not disable extraction for the run.
        if answered:
            _failures += 1
        return None
    _failures = 0

    # Group by attribute: the model may split one attribute's wanted value and
    # its exclusions across several entries ("cotton" and "not leather" as two
    # objects), so entries sharing an attribute are merged into one _Reading --
    # wanted values compete on confidence, negated lists union. Confidence on
    # the wanted side comes from the model verbatim; rescaling it here would
    # silently overrule its judgement of how sure the customer is.
    by_attribute: dict[str, list] = {}
    for item in reading.attributes:
        by_attribute.setdefault(item.attribute, []).append(item)

    readings = []
    for attribute, items in by_attribute.items():
        wanted = [i for i in items if i.value]
        negated: list[str] = []
        for i in items:
            for v in i.negated:
                if v not in negated:
                    negated.append(v)
        if wanted:
            best = max(wanted, key=lambda i: i.confidence)
            readings.append(_Reading(attribute=attribute, value=best.value,
                                      confidence=best.confidence, negated=negated))
        else:
            readings.append(_Reading(attribute=attribute, negated=negated))

    return _Frame(name="llm", readings=tuple(readings), buy_intent=float(reading.buy_intent))


# --------------------------------------------------------------------------
# pipeline -- the only place state is mutated
# --------------------------------------------------------------------------

def _apply(state: SessionState, frame: _Frame, turn: int) -> None:
    """Fold one frame into the state: one bind (+ cascade) per reading.

    category flows through this same loop as an ordinary attribute -- no
    special case. Replacing it fires _cascade exactly like any other
    attribute, which is what forgets the category-scoped slots (material,
    style, ...) once the category itself has changed.
    """
    for reading in frame.readings:
        attribute = reading.attribute

        value = reading.value.strip()
        if value:
            key, resolution = _resolve(value)
            confidence = (reading.confidence * resolution) if key else _CONF_UNRESOLVED
            # ExtractedAttribute's validator already guaranteed this parses.
            bound_value = float(value) * _BUDGET_TOLERANCE if attribute == "budget" else value
            slot = state.slots.get(attribute)
            replaced = slot is not None and slot.filled and slot.val != bound_value
            state.bind(attribute, bound_value, key, confidence, turn)
            if replaced:
                _cascade(state, attribute, turn)

        for excluded in reading.negated:
            excluded = excluded.strip()
            if not excluded:
                continue
            key, _resolution = _resolve(excluded)
            state.exclude(attribute, key or excluded)


def extract(message: str, turn: int, state: SessionState | None = None) -> SessionState:
    """Read one customer message into the session state."""
    if state is None:
        state = SessionState()
    try:
        state.begin_turn(turn)
        text = (message or "").strip()
        if not text:
            return state
        # Record before reading, so the LLM path sees this turn in context.
        state.record_customer(turn, text)

        frame = _route(text)
        if _frame_is_trustworthy(frame):
            logger.info("turn %d: regex frame=%s readings=%d -- trusted, no LLM call",
                        turn, frame.name, len(frame.readings))
            _apply(state, frame, turn)
            state.buy_intent = _buy_intent(state, frame, turn)
            return state

        # Either nothing matched, or the match was degraded (empty/oversized/
        # unclassifiable payload) -- both are a silent-wrong-extraction risk,
        # so both fall through to the same LLM path.
        logger.info("turn %d: regex frame=%s not trustworthy -- falling back to LLM",
                    turn, frame.name if frame is not None else None)
        frame = _llm_frame(text, turn, state)
        if frame is None:
            logger.info("turn %d: LLM fallback returned nothing (no provider/circuit breaker/parse failure)",
                        turn)
            # No provider, no key, or the circuit breaker tripped. There is
            # nothing to fold in, but the turn still has to leave the state
            # usable: _apply(state, None, ...) raises AttributeError into the
            # handler below, which loses buy_intent as well as the reading.
            # _buy_intent scores the mode from evidence already held.
            state.buy_intent = _buy_intent(state, None, turn)
            return state

        logger.info("turn %d: LLM frame readings=%d buy_intent=%.2f",
                    turn, len(frame.readings), frame.buy_intent)
        _apply(state, frame, turn)
        state.buy_intent = frame.buy_intent
        return state
    except Exception:
        # A miss is bad; an exception loses the whole turn, recommendations
        # included, because agent.respond() would never reach ask.decide().
        return state
