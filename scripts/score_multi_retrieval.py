#!/usr/bin/env python3
"""Score multi_retrieval against the real evaluator.

    python3 scripts/score_multi_retrieval.py
    python3 scripts/score_multi_retrieval.py --fusion additive --no-layered

WARNING — the slot filler in this file is scaffolding, not the product. It reads
the simulated customer's sentence templates so multi_retrieval can be measured on
its own. The real Agent gets its slots from the state component. Nothing here
belongs in multi_retrieval/, and nothing here should ship as the submitted Agent.

It is deliberately written to fill slots about as well as
scripts/score_retrieval.py fills its own query object. If one harness parses
better than the other, the comparison measures the harnesses rather than the
retrievers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from multi_retrieval import DualQuery, DualTrackRetriever, Slots  # noqa: E402

# --- scaffolding: the simulator's fixed phrasings -----------------------------
CATEGORY_RE = re.compile(r"I'm looking for (.+?)(?:,\s*but I'm still exploring|\.)", re.I)
REQUIREMENT_RE = re.compile(r"A key requirement is:\s*(.+?)\.?$", re.I)
MATTERS_RE = re.compile(r"what matters is:\s*(.+?)\.?$", re.I)
NEED_RE = re.compile(r"What I need is:\s*(.+?)\.?$", re.I)
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")

COLORS = ("black", "white", "blue", "red", "pink", "green", "brown",
          "gray", "grey", "purple", "yellow", "orange")
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex",
             "silk", "rayon", "fabric")
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("style", "fit", "sleeve", "neck", "department")

ASK_ORDER = ("feature", "material", "color", "style", "size", "use_case", "brand", "other")


def slot_for(fact: str) -> str:
    """Which slot a stated fact belongs in. Independent of the evaluator's own
    classifier — this package does not import from evaluator/ or retrieval/."""
    lowered = fact.lower()
    if PRICE_RE.search(lowered) or "budget" in lowered:
        return "price_max"
    if any(word in lowered for word in MATERIALS):
        return "material"
    if any(word in lowered for word in COLORS) or lowered.startswith("color"):
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    return "free_text"


class DualTrackAgent:
    """Adapter so multi_retrieval can run through the official evaluator."""

    def __init__(self, retriever: DualTrackRetriever, *, list_width: int = 10) -> None:
        self.retriever = retriever
        self.list_width = list_width
        self.sessions: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {"slots": Slots(), "intent": "browsing", "asked": set()}

    def _absorb(self, state: dict, message: str) -> None:
        slots: Slots = state["slots"]

        match = CATEGORY_RE.search(message)
        if match and not slots.category:
            # parsed word-for-word out of the customer's sentence, so it is safe
            # to filter on -- unlike a category inferred from scattered words
            slots.category = match.group(1).strip()
            slots.category_trusted = True

        for pattern, intent in ((REQUIREMENT_RE, "buying"), (NEED_RE, "buying"),
                                (MATTERS_RE, None)):
            found = pattern.search(message)
            if not found:
                continue
            if intent:
                state["intent"] = intent
            for part in found.group(1).split("; "):
                fact = part.strip().rstrip(".")
                if fact:
                    self._place(slots, fact)

    @staticmethod
    def _place(slots: Slots, fact: str) -> None:
        name = slot_for(fact)
        if name == "price_max":
            found = PRICE_RE.search(fact)
            if found:
                slots.price_max = float(found.group(1))
            return                          # "budget around $-" has no number: drop it
        if name == "free_text":
            if fact not in slots.free_text:
                slots.free_text.append(fact)
            return
        if not getattr(slots, name):
            setattr(slots, name, fact)
        elif fact not in slots.free_text:
            slots.free_text.append(fact)    # keep the second value rather than lose it

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions[session_id]
        self._absorb(state, user_message)

        result = self.retriever.retrieve(DualQuery(
            slots=state["slots"],
            intent=state["intent"],
            raw_message=user_message,
            top_k=top_k,
        ))

        ask = next((a for a in ASK_ORDER if a not in state["asked"]), None)
        if ask:
            state["asked"].add(ask)
        width = min(top_k, self.list_width)
        return {
            "message": "Here are the closest matches I have so far.",
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": a} for a in result.parent_asins[:width]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--fusion", default="rrf", choices=("rrf", "additive"))
    parser.add_argument("--no-layered", action="store_true",
                        help="skip the hard-constraint layer entirely")
    parser.add_argument("--list-width", type=int, default=10)
    parser.add_argument("--vector", default="off",
                        choices=("off", "hashing", "sentence-transformers"),
                        help="which embedder backs the vector route")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedder = None
    if args.vector == "hashing":
        from multi_retrieval.embed import HashingEmbedder
        embedder = HashingEmbedder()
    elif args.vector == "sentence-transformers":
        from multi_retrieval.embed import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    started = time.monotonic()
    retriever = DualTrackRetriever(
        args.catalog, fusion=args.fusion, layered=not args.no_layered,
        embedder=embedder,
    )
    build_seconds = time.monotonic() - started

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    started = time.monotonic()
    agent = DualTrackAgent(retriever, list_width=args.list_width)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    eval_seconds = time.monotonic() - started

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))
        return

    print(f"score  {result['recommended_technical_score']:.4f}"
          f"   hit {result['hit_rate_at_10']:.3f}"
          f"   mrr {result['mrr']:.3f}"
          f"   mttc {result['mttc']:.2f}")
    for name, metrics in sorted(result["scenario_metrics"].items()):
        efficiency = max(0.0, min(1.0, (11.0 - metrics["mttc"]) / 10.0))
        scenario = 0.5 * metrics["hit_rate_at_10"] + 0.3 * metrics["mrr"] + 0.2 * efficiency
        print(f"  {name:16s} {scenario:.4f}   hit {metrics['hit_rate_at_10']:.3f}"
              f"   mrr {metrics['mrr']:.3f}   n={metrics['sample_count']}")
    print(f"  index build {build_seconds:.1f}s   200 sessions {eval_seconds:.1f}s")
    print(f"  {retriever.diagnostics}")


if __name__ == "__main__":
    main()
