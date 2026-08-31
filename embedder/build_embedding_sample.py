"""Build a 25% spread-out sample of the catalog.

Sampling is systematic (every 4th row from a fixed offset), which spreads the
sample evenly across the file and avoids the head-of-file price skew that
`--limit` suffers from. Deterministic: no RNG, no seed to carry around.

This file decides *which* rows are in the sample, not what text they embed to --
that lives in variants.py and is rebuilt from the catalog at encode time.

    python3 -m embedder.build_embedding_sample
    python3 -m embedder.build_embedding_sample --rate 4 --offset 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATALOG = Path("data/catalog.jsonl")
NORMALISED = Path("data/normalised/catalog_normalised.jsonl")
DEFAULT_OUT = Path("data/catalog_sample_25.jsonl")


def make_record(product: dict, norm: dict, index: int) -> dict:
    return {
        "parent_asin": product["parent_asin"],
        "row_index": index,
        # carried through for metadata filtering at query time
        "category": norm.get("category"),
        "audience": norm.get("audience"),
        "brand": norm.get("brand"),
        "price": norm.get("price"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=4, help="keep 1 row in N (default 4 = 25%%)")
    parser.add_argument("--offset", type=int, default=0, help="first row index to keep")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--normalised", type=Path, default=NORMALISED)
    args = parser.parse_args()

    norms = {}
    if args.normalised.exists():
        with args.normalised.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                norms[row["parent_asin"]] = row
    else:
        print(f"warning: {args.normalised} missing -- facets will be empty")

    kept = 0
    with args.catalog.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8") as sink:
        for index, line in enumerate(source):
            if index % args.rate != args.offset % args.rate:
                continue
            product = json.loads(line)
            record = make_record(product, norms.get(product["parent_asin"], {}), index)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} rows to {args.out}")


if __name__ == "__main__":
    main()
