#!/usr/bin/env python3
"""Build the slot inverted index from the normalised catalog.

    python scripts/build_inverted_index.py

Every attribute value in data/catalog_normalised.jsonl gets a posting list of
the products carrying it -- red -> [B00..., B01...], blue -> [B07..., B09...],
for all nine attribute columns, not just colour. That is the whole artifact:
a rule-based lookup with no scoring, no model, and no ranking opinion.

    {"field": "material", "value": "cotton", "df": 9126, "ids": ["B07K34RX5J", ...]}

WHY THIS FILE EXISTS AT ALL
    multi_retrieval/index.py builds its FTS5 table in ":memory:" and throws it
    away on exit -- there is no stored index in this repo today. This one is
    written to disk, so a consumer loads it in a fraction of a second instead
    of re-deriving it every run.

WHAT IT IS NOT
    Not a replacement for the BM25 keyword index. Of the 800 phrases the
    simulator utters, only ~28% appear as a normalised slot value; the
    discriminative ones ("Triple Moon Pentagram Symbol") live only in raw
    feature text. The slot table says what TYPE a product is. Fuse the two --
    do not swap one for the other.

Field vocabulary and value normalisation come from starter.preprocessing, the
same functions ask.py reads the slot table through, so the two can never drift
onto different spellings of the same value.

Stdlib only. Deterministic: run it twice, get identical bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.preprocessing import SLOT_FIELDS, SLOT_PATH, row_slots  # noqa: E402

DEFAULT_OUT = "data/inverted_index.jsonl"

# Bumped whenever the line shape changes, so a loader can refuse an index it
# does not understand instead of misreading it.
FORMAT_VERSION = 1


def build_postings(
    source: Path, fields: tuple[str, ...],
) -> tuple[dict[tuple[str, str], list[str]], int, int]:
    """(field, value) -> [parent_asin], in source order.

    Returns the postings, the row count, and the number of rows skipped as
    unusable. Ids stay in file order rather than sorted: it is already stable,
    costs nothing, and keeps the artifact aligned with the catalog it came from.
    """
    postings: dict[tuple[str, str], list[str]] = {}
    rows = skipped = 0

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                asin = str(row["parent_asin"])
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            rows += 1
            slots = row_slots(row)
            for field in fields:
                for value in slots.get(field, ()):
                    postings.setdefault((field, value), []).append(asin)

    return postings, rows, skipped


def write_index(
    out: Path,
    source: Path,
    fields: tuple[str, ...],
    postings: dict[tuple[str, str], list[str]],
    rows: int,
    min_df: int,
) -> tuple[int, int]:
    """Write manifest + one line per term. Returns (terms, entries) written.

    Terms are ordered by (field, -df, value) so the file reads top-down as
    "the most common value of each attribute first", and so two builds of the
    same source produce byte-identical output.

    `df` is stored; `idf` is not. It is derivable from `df` and the row count,
    and a stored idf would be a second copy of the truth that can go stale.
    """
    stat = source.stat()
    kept = [
        (field, value, ids)
        for (field, value), ids in postings.items()
        if len(ids) >= min_df
    ]
    kept.sort(key=lambda item: (item[0], -len(item[2]), item[1]))

    manifest = {
        "manifest": FORMAT_VERSION,
        "source": source.as_posix(),
        # Size and mtime together are the staleness check: a loader compares
        # them against the source it was handed and rebuilds on a mismatch,
        # rather than silently serving postings for a catalog that has moved on.
        "source_bytes": stat.st_size,
        "source_mtime": int(stat.st_mtime),
        "rows": rows,
        "fields": list(fields),
        "terms": len(kept),
        "entries": sum(len(ids) for _f, _v, ids in kept),
        "min_df": min_df,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for field, value, ids in kept:
            handle.write(json.dumps(
                # parent_asin, not a row number: the index stays valid if the
                # catalog is reordered or rebuilt.
                {"field": field, "value": value, "df": len(ids), "ids": ids},
                ensure_ascii=False,
            ) + "\n")

    return manifest["terms"], manifest["entries"]


def summarise(postings: dict[tuple[str, str], list[str]], fields: tuple[str, ...],
              rows: int, min_df: int) -> str:
    lines = [f"{'field':12s}{'terms':>8s}{'entries':>10s}{'coverage':>10s}"]
    for field in fields:
        terms = [ids for (f, _v), ids in postings.items() if f == field and len(ids) >= min_df]
        covered = len({asin for ids in terms for asin in ids})
        lines.append(f"{field:12s}{len(terms):8d}{sum(len(i) for i in terms):10d}"
                     f"{covered / rows if rows else 0:9.1%}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SLOT_PATH,
                        help=f"normalised catalog JSONL (default: {SLOT_PATH})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output JSONL (default: {DEFAULT_OUT})")
    parser.add_argument("--fields", default="",
                        help="comma-separated subset of: " + ", ".join(SLOT_FIELDS))
    parser.add_argument("--min-df", type=int, default=1,
                        help="drop values held by fewer than this many products")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        print(f"error: no such source file: {source}", file=sys.stderr)
        return 1

    if args.fields:
        requested = tuple(f.strip() for f in args.fields.split(",") if f.strip())
        unknown = [f for f in requested if f not in SLOT_FIELDS]
        if unknown:
            print(f"error: unknown field(s): {', '.join(unknown)}\n"
                  f"       known fields: {', '.join(SLOT_FIELDS)}", file=sys.stderr)
            return 2
        fields = requested
    else:
        fields = SLOT_FIELDS

    started = time.monotonic()
    postings, rows, skipped = build_postings(source, fields)
    if not rows:
        print(f"error: {source} held no usable rows", file=sys.stderr)
        return 1

    out = Path(args.out)
    terms, entries = write_index(out, source, fields, postings, rows, args.min_df)
    elapsed = time.monotonic() - started

    print(f"source  {source}  ({rows:,} rows"
          + (f", {skipped:,} skipped" if skipped else "") + ")")
    print(summarise(postings, fields, rows, args.min_df))
    print(f"\nwrote   {out}  {terms:,} terms  {entries:,} entries  "
          f"{out.stat().st_size / 1048576:.1f} MB  in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
