"""Shared dataclasses. NO OWNER.

Every module imports this; this imports nothing from them. If you need a new
field, say so in the team channel first -- a change here touches all five.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The attributes worth asking a customer about. "brand" and "category" are
# deliberately absent: see ask.py.
REACHABLE_ATTRIBUTES = ("material", "color", "size", "style", "use_case", "budget", "feature")


@dataclass
class Constraint:
    """One requirement the customer has expressed.

    `text` is what they actually said ("waterproof"). `key` is the catalog
    phrasing it resolved to ("water resistant") and is what retrieval looks
    up -- it is empty when nothing in the catalog matched, which is normal and
    must not be treated as an error.
    """

    text: str
    key: str = ""
    attribute: str = "feature"
    hard: bool = False
    weight: float = 1.0
    turn: int = 0

    @property
    def resolved(self) -> bool:
        """True when this constraint reached real catalog vocabulary."""
        return bool(self.key)


@dataclass
class Extraction:
    """What one customer utterance told us. Produced by extract.py."""

    constraints: list[Constraint] = field(default_factory=list)
    intent: str = "browsing"                # "buying" | "browsing"
    override: bool = False                  # customer retracted something
    no_preference: str | None = None        # attribute they declined to state
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})


@dataclass
class SessionState:
    """Everything accumulated across one conversation. Owned by state.py."""

    session_id: str = ""
    profile: dict = field(default_factory=dict)
    turn: int = 0
    scenario: str = "browsing"
    category: str | None = None
    constraints: list[Constraint] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    dead_attributes: set[str] = field(default_factory=set)

    @property
    def hard_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.hard]

    @property
    def resolved_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.resolved]


@dataclass
class TurnPolicy:
    """What ask.py decided for this turn, including the text we say back."""

    ask_attribute: str | None = None
    list_width: int = 10
    message: str = "Here are the closest matches I have so far."
