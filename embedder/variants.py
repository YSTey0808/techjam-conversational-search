"""What text each product turns into before it is embedded.

V2 is the only variant. A facets-only variant (V1) was tested and dropped -- it
scored hit@10 0.060 against V2's 0.295. See EMBEDDER_README.md section 4.

Builders take (norm, raw): the normalised row and the raw catalog row.
"""
from __future__ import annotations

# Order matters: it is stable across rows, so the model sees a consistent shape.
FACET_SCALARS = ("category", "audience", "brand")
FACET_LISTS = ("material", "color", "feature", "style", "use_case")
FACET_TAIL = ("region",)

# The only `details` keys that carry semantics. The rest -- model numbers,
# package dimensions, availability dates -- are logistics noise with very high
# cardinality, which is exactly what you do not want in an embedding.
DETAIL_KEYS = ("Color", "Material", "Department", "Style")


def facet_text(norm: dict) -> str:
    """The ten normalised columns as a token string.

    `region` uses the string "unknown" for "not stated"; emitting it would put the
    same token on 67% of rows, which is noise, not signal. List columns use [] for
    the same meaning and drop out naturally.
    """
    parts = [norm.get(key) or "" for key in FACET_SCALARS]
    for key in FACET_LISTS:
        parts.extend(norm.get(key) or [])
    for key in FACET_TAIL:
        value = norm.get(key)
        if value and value != "unknown":
            parts.append(value)
    return " ".join(part for part in parts if part)


def v2_text(norm: dict, raw: dict) -> str:
    """V2 -- everything that carries semantics, no truncation.

    Segments are ordered cheapest-signal-first so that if a shorter-context model
    is ever swapped in, what gets cut is the tail (description), which is the
    lowest-precision part.
    """
    details = raw.get("details") or {}
    detail_text = " ".join(
        f"{key}: {details[key]}" for key in DETAIL_KEYS if details.get(key)
    )
    segments = [
        raw.get("title") or "",
        # categories[0] is the "Clothing, Shoes & Jewelry" root on all 50,000
        # rows -- constant, so it carries no signal.
        " > ".join((raw.get("categories") or [])[1:]),
        facet_text(norm),
        detail_text,
        " ".join(raw.get("features") or []),
        " ".join(raw.get("description") or []),
    ]
    return " | ".join(segment.strip() for segment in segments if segment.strip())


BUILDERS = {"v2": v2_text}

# 2048 covers every product: the longest tokenises to 1731, so nothing is cut.
ENCODE_DEFAULTS = {
    "v2": {"max_seq_length": 2048, "batch_size": 32},
}
