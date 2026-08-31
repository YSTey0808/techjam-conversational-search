#!/usr/bin/env python3
"""Run each retrieval package inside the team's real agent pipeline.

    python3 scripts/score_integrated.py

scripts/score_retrieval.py and scripts/score_multi_retrieval.py both feed their
retriever from a scaffolding parser that reads the simulator's templates. That
isolates retrieval quality, which is what you want when comparing two
retrievers -- but it is not what shipping looks like.

This script replaces only the retrieval stage of `starter/agent.py`, keeping
the team's real extract -> state -> ask chain, and measures what the pipeline
would actually score. It writes to nothing in starter/; each variant is a small
subclass built here.

The difference between a package's isolated score and its integrated score is
the cost of the extraction layer, and that is the number worth arguing about.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from reranker import adapter  # noqa: E402
from starter import ask, extract, preprocessing  # noqa: E402
from starter import state as state_mod  # noqa: E402
from starter.schema import SessionState  # noqa: E402

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")

# --- oracle extractor --------------------------------------------------------
# A stand-in for a FIXED extract.py: it keeps stated requirements whole instead
# of shredding them into words. Everything downstream (state, ask) is unchanged,
# so the gap between a row using this and the same row using the real extractor
# is exactly what the extraction layer is costing.
#
# This is a measuring instrument, not a proposal -- it reads the simulator's
# sentence templates, which a real extractor cannot do.
CATEGORY_RE = re.compile(r"I'm looking for (.+?)(?:,\s*but I'm still exploring|\.)", re.I)
FACT_RES = (
    re.compile(r"A key requirement is:\s*(.+?)\.?$", re.I),
    re.compile(r"what matters is:\s*(.+?)\.?$", re.I),
    re.compile(r"What I need is:\s*(.+?)\.?$", re.I),
)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple")


def _attribute_of(text: str) -> str:
    lowered = text.lower()
    if "budget" in lowered:
        return "budget"
    if any(w in lowered for w in MATERIALS):
        return "material"
    if any(w in lowered for w in COLORS) or lowered.startswith("color"):
        return "color"
    if any(w in lowered for w in ("size", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in ("style", "fit", "sleeve", "neck", "department")):
        return "style"
    return "feature"


def oracle_extract(message: str, turn: int, state) -> "Extraction":
    from starter import preprocessing
    from starter.schema import Constraint, Extraction

    prep = preprocessing.active()
    constraints, intent, override = [], "browsing", False

    if "ignore my earlier preference" in message.lower():
        override = True
    for pattern in FACT_RES:
        found = pattern.search(message)
        if not found:
            continue
        if pattern is not FACT_RES[1]:
            intent = "buying"
        for part in found.group(1).split("; "):
            phrase = part.strip().rstrip(".")
            if not phrase:
                continue
            key = ""
            if prep is not None:
                candidate = preprocessing.normalize(phrase)
                if candidate and prep.lookup(candidate, broad=True):
                    key = candidate
            constraints.append(Constraint(
                text=phrase, key=key, attribute=_attribute_of(phrase),
                hard=True, weight=1.0, turn=turn,
            ))
    match = CATEGORY_RE.search(message)
    if match:
        phrase = match.group(1).strip()
        key = ""
        if prep is not None:
            candidate = preprocessing.normalize(phrase)
            if candidate and prep.lookup(candidate, broad=True):
                key = candidate
        constraints.append(Constraint(text=phrase, key=key, attribute="category",
                                      hard=False, weight=1.0, turn=turn))
    return Extraction(constraints=constraints, intent=intent, override=override,
                      no_preference=None, usage={"prompt_tokens": 0, "completion_tokens": 0})

# Their Constraint.attribute vocabulary -> multi_retrieval slot names. The two line
# up well because both are typed by attribute; "use_case" has no direct slot so
# it lands on the nearest one.
ATTRIBUTE_TO_SLOT = {
    "material": "material", "color": "color", "size": "size",
    "style": "style", "use_case": "occasion", "brand": "brand",
}


class BaseAgent:
    """The team's pipeline with the retrieval stage left abstract."""

    def __init__(self, catalog_path: str) -> None:
        self.prep = preprocessing.build(catalog_path)
        self._sessions: dict[str, SessionState] = defaultdict(SessionState)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(
            session_id=session_id, profile=dict(user_profile or {}),
        )

    # ask.decide() measures how well each attribute splits the candidate pool,
    # so it needs a real pool. Their retrieve() hands it ~200; handing it only
    # the final ten would starve the question-picking of anything to work with.
    POOL_SIZE = 200

    def _pool(self, state: SessionState, message: str, size: int) -> list[str]:
        raise NotImplementedError

    ORACLE = False        # True swaps in a fixed extractor, nothing else

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        extractor = oracle_extract if self.ORACLE else extract.extract
        extraction = extractor(user_message, turn, state)
        state = state_mod.update(state, extraction)
        pool = self._pool(state, user_message, self.POOL_SIZE)
        policy = ask.decide(self.prep, state, pool, turn)
        self._sessions[session_id] = state
        return {
            "message": policy.message,
            "ask_attribute": policy.ask_attribute,
            "recommendations": [{"parent_asin": a} for a in pool[: policy.list_width]],
            "usage": extraction.usage,
        }


class TheirAgent(BaseAgent):
    """Their retrieve + the reranker.

    This was the pre-reranker baseline (retrieve + starter/rank.py). rank.py has
    been deleted, so the row now measures retrieve + reranker/adapter.py and is
    no longer a fixed reference point for the old IDF + popularity + budget
    scoring -- compare against a recorded number, not against this row.
    """

    def _pool(self, state, message, size):
        from starter import retrieve
        candidates = retrieve.retrieve(self.prep, state)
        return adapter.rerank(self.prep, state, candidates, size, False)


class MultiRetrievalAgent(BaseAgent):
    """Their extract/state, our multi_retrieval in place of retrieve + rank."""

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        from multi_retrieval import DualTrackRetriever
        from multi_retrieval.embed import HashingEmbedder
        self.retriever = DualTrackRetriever(catalog_path, embedder=HashingEmbedder())

    @staticmethod
    def _slots(state: SessionState):
        from multi_retrieval import Slots
        slots = Slots(category=state.category or "")
        for constraint in state.constraints:
            name = ATTRIBUTE_TO_SLOT.get(constraint.attribute)
            if name and not getattr(slots, name):
                setattr(slots, name, constraint.text)
            elif constraint.text not in slots.free_text:
                slots.free_text.append(constraint.text)
        return slots

    def _pool(self, state, message, size):
        from multi_retrieval import DualQuery
        result = self.retriever.retrieve(DualQuery(
            slots=self._slots(state), intent=state.scenario,
            raw_message=message, top_k=size,
        ))
        return result.parent_asins


class MultiRetrievalPoolAgent(MultiRetrievalAgent):
    """multi_retrieval as pure candidate generation; the reranker orders it.

    This is what the architecture diagram actually draws: routes feed a candidate
    pool, and a separate stage ranks it. The ranking stage is Owner D and
    independent of Owner C, so the two can be mixed. It also means
    multi_retrieval's own fusion order is thrown away and replaced by the
    reranker's RRF fusion (and its LLM step, when a key is present).
    """

    def _pool(self, state, message, size):
        candidates = super()._pool(state, message, self.retriever.pool_limit)
        return adapter.rerank(self.prep, state, candidates, size, False)

def scenario_score(metrics: dict) -> float:
    efficiency = max(0.0, min(1.0, (11.0 - metrics["mttc"]) / 10.0))
    return 0.5 * metrics["hit_rate_at_10"] + 0.3 * metrics["mrr"] + 0.2 * efficiency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    ids, cats, prods = catalog_index(args.catalog)

    header = f"{'retrieval stage':40s} {'score':>8s} {'hit':>6s} {'mrr':>6s} {'mttc':>6s}"
    header += "".join(f"{n[:9]:>10s}" for n in SCENARIOS) + f"{'secs':>7s}"
    print("\nINSIDE THE TEAM PIPELINE  (their extract -> state -> ask, unchanged)")
    print(header)
    print("-" * len(header))

    variants = []
    stages = (("their retrieve + reranker", TheirAgent),
              ("multi_retrieval/ own ranking", MultiRetrievalAgent),
              ("multi_retrieval/ + reranker", MultiRetrievalPoolAgent))

    for label, factory in stages:
        variants.append((label + "  [real extract]", factory, False))
    for label, factory in stages:
        variants.append((label + "  [FIXED extract]", factory, True))

    for label, factory, oracle in variants:
        started = time.monotonic()
        agent = factory(args.catalog)
        agent.ORACLE = oracle
        result = evaluate(agent, samples, ids, cats, prods)
        elapsed = time.monotonic() - started
        line = (f"{label:40s} {result['recommended_technical_score']:8.4f}"
                f"{result['hit_rate_at_10']:7.3f}{result['mrr']:7.3f}{result['mttc']:7.2f}")
        for name in SCENARIOS:
            metrics = result["scenario_metrics"].get(name)
            line += f"{scenario_score(metrics):10.4f}" if metrics else f"{'-':>10s}"
        print(line + f"{elapsed:6.0f}s")


if __name__ == "__main__":
    main()
