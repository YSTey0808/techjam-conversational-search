"""Merge the routes into one candidate pool.

Reciprocal rank fusion is the default, matching the convention already used in
``starter/retrieve.py``: each route contributes ``weight / (k + rank)``, so a
route can promote its own best candidates without needing its scores to be
comparable with anyone else's. That is the property that matters when BM25
scores, IDF overlap fractions and cosine similarities all arrive on different
scales.

Weighted-additive fusion is available behind the same call. It keeps magnitude —
"three strong route hits" outranks "one" — at the cost of needing each route
normalised first. Which is better here is a measurement, not a preference, so
both ship and the harness can sweep them.
"""

from __future__ import annotations

RRF_K = 60


def _ranked(scores: dict[int, float]) -> list[int]:
    return [product for product, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _normalized(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    largest = max(scores.values())
    smallest = min(scores.values())
    spread = largest - smallest
    if spread <= 0:
        return {product: 1.0 for product in scores}
    return {product: (value - smallest) / spread for product, value in scores.items()}


def fuse(
    routes: dict[str, dict[int, float]],
    weights: dict[str, float],
    *,
    mode: str = "rrf",
    k: int = RRF_K,
) -> dict[int, float]:
    """Combine per-route scores into one score per product."""
    combined: dict[int, float] = {}

    for name, scores in routes.items():
        weight = weights.get(name, 0.0)
        if weight <= 0.0 or not scores:
            continue
        if mode == "additive":
            for product, value in _normalized(scores).items():
                combined[product] = combined.get(product, 0.0) + weight * value
        else:
            for position, product in enumerate(_ranked(scores), start=1):
                combined[product] = combined.get(product, 0.0) + weight / (k + position)

    return combined


def order(combined: dict[int, float], ids: list[str], limit: int) -> list[tuple[int, float]]:
    """Best first, ties broken on parent_asin so output is reproducible."""
    ranked = sorted(combined.items(), key=lambda item: (-item[1], ids[item[0]]))
    return ranked[:limit]


__all__ = ["fuse", "order", "RRF_K"]
