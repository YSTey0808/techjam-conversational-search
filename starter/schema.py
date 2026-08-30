"""Shared dataclasses. NO OWNER.

Every module imports this; this imports nothing from them. If you need a new
field, say so in the team channel first -- a change here touches all five.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Every slot the state carries, matching docs/session_state.json.
# NOTE: the last one is "others" here but "other" in the ask_attribute enum of
# docs/agent_api_contract.json. Translate at the boundary in ask.py.
SLOT_ATTRIBUTES = (
    "category", "material", "color", "brand",
    "budget", "style", "feature", "use_case", "others",
)

# The subset worth asking a customer about. "brand" and "category" are
# deliberately absent: see ask.py. They exist as slots because the customer may
# still volunteer them; they are just never the subject of a question.
REACHABLE_ATTRIBUTES = ("material", "color", "style", "use_case", "budget", "feature")

# Confidence threshold, tunable
HARD_CONFIDENCE = 0.75


@dataclass
class Slot:
    """One attribute of what the customer wants.

    `val`: numeric for budget, string otherwise. 
    
    `key`: the catalog phrasing it resolved to and is what retrieval looks up

    `confidence_score`: a floating point number represents the confidence of that attribution value from user message

    `excluded`: values the customer has ruled OUT for this attribute ("nothing
    formal" -> excluded=["formal"]). 
    """

    val: str | float | list[str] = ""
    confidence_score: float | None = 0.0
    key: str = ""
    turn: int = 0
    keys: list[str] = field(default_factory=list)     # every key bound here, newest first
    excluded: list[str] = field(default_factory=list)  # catalog keys ruled out

    @property
    def filled(self) -> bool:
        """True once the customer has told us something."""
        return bool(self.val)

    @property
    def hard(self) -> bool:
        return bool(self.confidence_score) and self.confidence_score >= HARD_CONFIDENCE


def _empty_slots() -> dict[str, Slot]:
    return {attribute: Slot() for attribute in SLOT_ATTRIBUTES}


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0}


@dataclass
class UserProfile:
    """The anonymized profile handed to reset().

    SOFT SIGNAL ONLY. Never turn a field here into a hard filter.
    Shape is fixed by docs/agent_api_contract.json.
    """

    purchase_frequency: str = ""
    average_prior_rating: float | None = None
    rating_style: str = ""
    preference_tags: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, data: dict | None) -> UserProfile:
        """Build from the raw dict the harness passes. Never raises."""
        data = data or {}
        rating = data.get("average_prior_rating")
        numeric = isinstance(rating, (int, float)) and not isinstance(rating, bool)
        tags = data.get("preference_tags")
        return cls(
            purchase_frequency=str(data.get("purchase_frequency") or ""),
            average_prior_rating=float(rating) if numeric else None,
            rating_style=str(data.get("rating_style") or ""),
            preference_tags=[str(t) for t in tags] if isinstance(tags, list) else [],
            summary=str(data.get("summary") or ""),
        )

    
@dataclass
class SessionStep:
    """One turn of the conversation, as it actually happened.

    Kept so extraction can read the dialogue rather than one isolated line:
    "I don't have a preference for that" only means something next to the
    question it answers.
    """

    turn: int = 0
    customer: str = ""
    agent: str = ""
    ask_attribute: str | None = None


@dataclass
class SessionState:
    """Agent memory on client preferences across one conversation. Owned by extract.py.

    Slots are the single source of truth for what the customer wants. This is
    the ONLY memory of the session. extract.py reads a message and writes 
    straight into slots through the mechanics below.
    """

    session_id: str = ""
    turn: int = 0
    buy_intent: float = 0.5     # 0.0 browsing/widen .. 1.0 buying/tighten
    slots: dict[str, Slot] = field(default_factory=_empty_slots)
    user_profile: UserProfile = field(default_factory=UserProfile)

    # ask-strategy bookkeeping
    asked: list[str] = field(default_factory=list)
    dead_attributes: set[str] = field(default_factory=set)

    # token count
    usage: dict = field(default_factory=_zero_usage)

    deflections: int = 0        # times the customer had nothing to add
    nudges: int = 0             # times we asked nothing and they prodded us

    # what was actually said, so extraction can read the dialogue not one line
    history: list[SessionStep] = field(default_factory=list)

    # -- reads ---------------------------------------------------------------

    def slot(self, attribute: str) -> Slot:
        """The slot for `attribute`, creating it if it is not a known one.

        Never raises and never returns None, so callers can read
        `state.slot("color").val` without a guard.
        """
        existing = self.slots.get(attribute)
        if existing is None:
            existing = self.slots[attribute] = Slot()
        return existing

    @property
    def filled_slots(self) -> dict[str, Slot]:
        return {name: s for name, s in self.slots.items() if s.filled}

    @property
    def hard_slots(self) -> dict[str, Slot]:
        """Firm requirements. Replaces the old `hard_constraints`."""
        return {name: s for name, s in self.slots.items() if s.filled and s.hard}

    @property
    def category(self) -> str | None:
        """Convenience read of the category slot, which retrieval gates on."""
        value = self.slots.get("category")
        return str(value.val) if value is not None and value.filled else None

    @property
    def scenario(self) -> str:
        """buy_intent as the coarse label retrieve._WEIGHTS keys on.

        Threshold is exclusive so the 0.5 default stays "browsing", matching
        the behaviour before buy_intent replaced the old string field.
        """
        return "buying" if self.buy_intent > 0.5 else "browsing"

    # -- mechanics -----------------------------------------------------------
    # No policy lives here. Callers decide what to bind and what to demote.

    def begin_turn(self, turn: int) -> None:
        """Start a turn. Resets per-turn usage; the evaluator sums across turns."""
        self.turn = max(self.turn, int(turn))
        self.usage = _zero_usage()

    def bind(
        self,
        attribute: str,
        val: str | float,
        key: str = "",
        confidence: float = 1.0,
        turn: int | None = None,
    ) -> Slot:
        """Record one thing the customer said. The only way a slot is written.

        A later turn always wins -- the customer changing their mind is the
        common case, and refusing the newer value would strand the session on a
        stale answer. Within one turn the more confident reading wins.

        `key` accumulates in `keys` rather than replacing, because one reveal
        can carry two catalog strings for the same attribute and dropping
        either loses an exact posting list.
        """
        target = self.slot(attribute)
        at = self.turn if turn is None else int(turn)
        held = target.confidence_score or 0.0
        stale = at < target.turn
        outranked = at == target.turn and confidence < held
        if key and key not in target.keys:
            target.keys.insert(0, key)
        if target.filled and (stale or outranked):
            return target
        target.val = val
        target.confidence_score = float(confidence)
        target.key = key
        target.turn = at
        return target

    def demote(self, attribute: str, factor: float) -> None:
        """Scale a slot's confidence down without forgetting its value.

        Demoting below HARD_CONFIDENCE drops the slot out of `hard_slots`, so
        retrieval stops intersecting on it, while `filled_slots` keeps it for
        ranking. Deleting instead would throw away a value the customer may
        well restate.
        """
        target = self.slots.get(attribute)
        if target is not None and target.filled:
            target.confidence_score = (target.confidence_score or 0.0) * factor

    def mark_dead(self, attribute: str) -> None:
        """The customer has no preference here. Never ask again."""
        if attribute:
            self.dead_attributes.add(attribute)

    def exclude(self, attribute: str, negated_val: str) -> None:
        """Record a value the customer ruled OUT for this attribute.

        Does not touch `val`/`key` -- ruling something out is not the same as
        choosing something, so this must not make an empty slot look filled.
        """
        if not attribute or not negated_val:
            return
        target = self.slot(attribute)
        if negated_val not in target.excluded:
            target.excluded.append(negated_val)

    def note_asked(self, attribute: str) -> None:
        if attribute and attribute not in self.asked:
            self.asked.append(attribute)

    def record_customer(self, turn: int, message: str) -> SessionStep:
        """Open a new SessionStep with what the customer just said."""
        entry = SessionStep(turn=int(turn), customer=message or "")
        self.history.append(entry)
        return entry

    def record_agent(self, message: str, ask_attribute: str | None) -> None:
        """Close the open SessionStep with what we said back."""
        if self.history:
            self.history[-1].agent = message or ""
            self.history[-1].ask_attribute = ask_attribute


@dataclass
class TurnPolicy:
    """What ask.py decided for this turn, including the text we say back."""

    ask_attribute: str | None = None
    list_width: int = 10
    message: str = "Here are the closest matches I have so far."
