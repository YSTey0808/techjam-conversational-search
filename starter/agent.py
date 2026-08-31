"""TechJam conversational search agent -- WIRING ONLY.

Four modules do the work, one owner each, one public function each:

    extract.extract(message, turn, state)   -> SessionState    Owner A
    retrieve.retrieve(prep, state)          -> list[asin]      Owner C
    ask.decide(prep, state, pool, turn)     -> TurnPolicy      Owner E
    adapter.rerank(prep, state, pool, k)    -> list[asin]      Owner D

There is deliberately no branching, no logic and no fallback in this file.
Anything that needs a decision belongs to the module that owns it -- including
degradation: extract() never raises, and retrieve() never returns empty.

The evaluator-facing contract is fixed:
    Agent(catalog_path) / reset(session_id, user_profile) / respond(...)
"""

from __future__ import annotations

from collections import defaultdict

from reranker import adapter
from starter import ask, extract, preprocessing, retrieve
from starter.schema import SessionState, UserProfile


class Agent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        # Built once, never per turn.
        self.prep = preprocessing.build(catalog_path)
        # defaultdict so respond() can index without a guard, keeping this
        # file branch-free even if the harness ever skips reset().
        self._sessions: dict[str, SessionState] = defaultdict(SessionState)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            user_profile=UserProfile.from_dict(user_profile),
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = extract.extract(user_message, turn, self._sessions[session_id])
        pool = retrieve.retrieve(self.prep, state)
        policy = ask.decide(self.prep, state, pool, turn)
        items = adapter.rerank(self.prep, state, pool, policy.list_width, False)
        state.record_agent(policy.message, policy.ask_attribute)
        self._sessions[session_id] = state
        return {
            "message": policy.message,
            "ask_attribute": policy.ask_attribute,
            "recommendations": [{"parent_asin": a} for a in items],
            "usage": state.usage,
        }
