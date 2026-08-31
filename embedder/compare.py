"""Turn-1 retrieval comparison between embedding variants.

    python3 -m embedder.compare
    python3 -m embedder.compare --variants v1 v2 --k 10

Scores each variant on the 200 `public_set` sessions using the evaluator's own
turn-1 message, so the queries are exactly what the real eval sends. No LLM and
no agent policy: this isolates embedding quality from conversation strategy.

Note what the queries are. `intent_card()` lifts the user's constraints straight
out of the ground-truth product's own `features`/`details`, so turn-1 text is
often a near-verbatim listing bullet. Scores here are flattered by that; treat
them as a relative measure of a change, not an absolute accuracy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from embedder.embed import DEFAULT_MODEL, MODELS, load_model, paths
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    materialize_hidden_fields,
)

PUBLIC_SET = Path("data/public_set.jsonl")
CATALOG = Path("data/catalog.jsonl")


def build_queries() -> list[dict]:
    """Reconstruct the evaluator's turn-1 message for every session."""
    _, categories, products = catalog_index(CATALOG)
    sessions = []
    with PUBLIC_SET.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            target = str(sample["ground_truth"]["parent_asin"])
            sessions.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "difficulty_bucket": sample["difficulty_bucket"],
                "target": target,
                # `disclosed` is mutated by initial_message; a throwaway set is fine
                # because we never generate a turn 2.
                "query": initial_message(
                    effective, coarse_category(categories.get(target, [])), set()
                ),
            })
    return sessions


def rank_of_target(scores: np.ndarray, target_row: int) -> int:
    """1-indexed rank of the target. Ties resolve pessimistically."""
    return int((scores > scores[target_row]).sum()) + 1


def score_variant(variant: str, sessions: list[dict], k: int, encoder,
                  model: str = DEFAULT_MODEL) -> list[dict]:
    vec_path, ids_path, _, _ = paths(variant, model)
    if not ids_path.exists():
        raise SystemExit(
            f"{variant}/{model} not built -- run "
            f"`embed build --variant {variant} --model {model}`"
        )
    vectors = np.asarray(np.load(vec_path, mmap_mode="r"))
    row_of = {asin: i for i, asin in enumerate(json.loads(ids_path.read_text()))}

    queries = encoder.encode(
        [MODELS[model]["query_prefix"] + s["query"] for s in sessions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    ).astype(np.float32)

    results = []
    for session, query in zip(sessions, queries):
        target_row = row_of.get(session["target"])
        if target_row is None:      # product not in the encoded corpus
            continue
        scores = vectors @ query
        rank = rank_of_target(scores, target_row)
        results.append({**session, "rank": rank, "hit": rank <= k,
                        "rr": 1.0 / rank if rank <= k else 0.0})
    return results


def summarise(results: list[dict], k: int) -> dict:
    n = len(results)
    ranks = [r["rank"] for r in results]
    return {
        "n": n,
        f"hit@{k}": sum(r["hit"] for r in results) / n,
        "mrr": sum(r["rr"] for r in results) / n,
        "hit@1": sum(r["rank"] == 1 for r in results) / n,
        "hit@100": sum(r["rank"] <= 100 for r in results) / n,
        "median_rank": int(np.median(ranks)),
    }


def table(title: str, rows: dict[str, dict], k: int) -> None:
    print(f"\n{title}")
    header = f"{'':22} {'n':>5} {'hit@1':>7} {f'hit@{k}':>7} {'mrr':>7} {'hit@100':>8} {'med rank':>9}"
    print(header)
    print("-" * len(header))
    for label, m in rows.items():
        print(f"{label:22} {m['n']:5d} {m['hit@1']:7.3f} {m[f'hit@{k}']:7.3f} "
              f"{m['mrr']:7.3f} {m['hit@100']:8.3f} {m['median_rank']:9d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["v2"])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--k", type=int, default=10, help="evaluator uses TOP_K=10")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(f"data/embeddings/compare_turn1_{args.model}.json")

    sessions = build_queries()
    print(f"{len(sessions)} sessions; example query:\n  {sessions[0]['query']}")

    # Query-side max_seq_length only needs to cover the messages, which are short.
    encoder = load_model(512, args.model)

    all_results = {}
    overall = {}
    for variant in args.variants:
        results = score_variant(variant, sessions, args.k, encoder, args.model)
        all_results[variant] = results
        overall[variant] = summarise(results, args.k)

    table("OVERALL", overall, args.k)

    for facet in ("scenario_type", "difficulty_bucket"):
        rows = {}
        for variant in args.variants:
            grouped = defaultdict(list)
            for r in all_results[variant]:
                grouped[r[facet]].append(r)
            for value in sorted(grouped):
                rows[f"{variant}  {value}"] = summarise(grouped[value], args.k)
        table(f"BY {facet.upper()}", rows, args.k)

    args.out.write_text(json.dumps(
        {"k": args.k, "model": args.model, "overall": overall,
         "sessions": {v: [{kk: r[kk] for kk in ("sample_id", "scenario_type",
                                                "difficulty_bucket", "rank", "hit")}
                          for r in all_results[v]] for v in args.variants}},
        indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
