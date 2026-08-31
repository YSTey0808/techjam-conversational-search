"""Embed a catalog variant with nomic-embed-text-v1.5.

    python3 -m embedder.embed build --variant v1
    python3 -m embedder.embed search --variant v1 "warm waterproof winter boots"

Writes a preallocated memmap so an interrupted run resumes where it stopped:
re-running `build` picks up from the row count in the .progress file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embedder.variants import BUILDERS, ENCODE_DEFAULTS

# Prefixes are per-model and not interchangeable. nomic is trained with
# asymmetric task prefixes; omitting them -- or using the same prefix on both
# sides -- costs real accuracy and raises no error. gte-v1.5 takes none at all,
# and bolting nomic's onto it would only add noise.
MODELS = {
    "nomic": {
        "id": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
}
# Read any new model's prefixes off its own config_sentence_transformers.json
# `prompts` map, not from memory -- a wrong prefix is silent and costly.
#
# Two models were tested and dropped: Snowflake/snowflake-arctic-embed-l-v2.0
# (4x bigger, no better -- a paired test says tie), and Alibaba-NLP/gte-large-en-v1.5
# (unusable: its custom modeling.py crashes inside encode() on both mps and cpu).

DEFAULT_MODEL = "nomic"

SAMPLE = Path("data/catalog_sample_25.jsonl")
NORMALISED = Path("data/normalised/catalog_normalised.jsonl")
OUT_DIR = Path("data/embeddings")


def paths(variant: str, model: str = DEFAULT_MODEL) -> tuple[Path, Path, Path, Path]:
    stem = OUT_DIR / f"{variant}_{model}"
    return (
        stem.with_suffix(".npy"),
        stem.with_name(stem.name + "_ids.json"),
        stem.with_name(stem.name + ".progress"),
        stem.with_name(stem.name + "_meta.json"),
    )


def load_documents(variant: str) -> tuple[list[str], list[str]]:
    """Return (asins, texts) for the sampled rows, in sample order."""
    if not SAMPLE.exists():
        sys.exit(f"missing {SAMPLE} -- run embedder.build_embedding_sample first")
    if not NORMALISED.exists():
        sys.exit(f"missing {NORMALISED} -- run dataset_normalisation.pipeline first")

    norms = {}
    with NORMALISED.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            norms[row["parent_asin"]] = row

    raws = {}
    with Path("data/catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raws[row["parent_asin"]] = row

    build = BUILDERS[variant]
    asins, texts = [], []
    with SAMPLE.open(encoding="utf-8") as handle:
        for line in handle:
            asin = json.loads(line)["parent_asin"]
            asins.append(asin)
            texts.append(build(norms[asin], raws[asin]))
    return asins, texts


def load_model(max_seq_length: int, model: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    import torch

    spec = MODELS[model]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {spec['id']} on {device} (first run downloads weights)")
    encoder = SentenceTransformer(spec["id"], trust_remote_code=True, device=device)
    encoder.max_seq_length = max_seq_length
    return encoder


def build(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    dim = spec["dim"]
    vec_path, ids_path, progress_path, meta_path = paths(args.variant, args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    asins, texts = load_documents(args.variant)
    total = len(asins)
    lengths = [len(t) for t in texts]
    print(f"{args.variant}: {total} docs, mean {sum(lengths)/total:.0f} chars")
    print(f"  example: {texts[0][:120]}")

    done = 0
    if args.resume and progress_path.exists() and vec_path.exists():
        done = int(progress_path.read_text().strip())
        print(f"resuming at row {done}")

    if done >= total:
        print("already complete -- nothing to do")
        return

    # Preallocate on disk so a kill -9 costs at most one checkpoint interval.
    mode = "r+" if done else "w+"
    if mode == "w+":
        vectors = np.lib.format.open_memmap(
            vec_path, mode="w+", dtype=np.float32, shape=(total, dim)
        )
    else:
        vectors = np.lib.format.open_memmap(vec_path, mode="r+")
        if vectors.shape[1] != dim:
            sys.exit(f"dim mismatch: {vec_path} is {vectors.shape}, expected (*, {dim})")
        if vectors.shape[0] < total:
            # The manifest grew (rows appended at the end). Rows are only
            # ever appended, so the vectors already written stay valid -- copy them
            # into a larger file and carry on from the same progress count.
            print(f"growing {vec_path}: {vectors.shape[0]} -> {total} rows")
            kept = np.array(vectors[:done])
            del vectors
            vectors = np.lib.format.open_memmap(
                vec_path, mode="w+", dtype=np.float32, shape=(total, dim)
            )
            vectors[:done] = kept
        elif vectors.shape[0] > total:
            sys.exit(f"{vec_path} has {vectors.shape[0]} rows but the sample has {total}")

    encoder = load_model(args.max_seq_length, args.model)
    started = time.time()

    while done < total:
        chunk = texts[done : done + args.checkpoint]
        vectors[done : done + len(chunk)] = encoder.encode(
            [spec["doc_prefix"] + text for text in chunk],
            batch_size=args.batch_size,
            normalize_embeddings=True,   # cosine similarity becomes a dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        done += len(chunk)
        vectors.flush()
        progress_path.write_text(str(done))
        rate = done / (time.time() - started)
        eta = (total - done) / rate if rate else 0
        print(f"  {done}/{total}  {rate:.0f} docs/s  eta {eta/60:.1f} min", flush=True)

    ids_path.write_text(json.dumps(asins))
    meta_path.write_text(json.dumps({
        "variant": args.variant,
        "model": spec["id"],
        "model_key": args.model,
        "dim": dim,
        "count": total,
        "max_seq_length": args.max_seq_length,
        "doc_prefix": spec["doc_prefix"],
        "query_prefix": spec["query_prefix"],
        "normalized": True,
        "seconds": round(time.time() - started, 1),
    }, indent=2))
    print(f"wrote {vec_path} ({total}x{dim}) in {(time.time()-started)/60:.1f} min")


def search(args: argparse.Namespace) -> None:
    vec_path, ids_path, _, _ = paths(args.variant, args.model)
    if not ids_path.exists():
        sys.exit(f"{args.variant}/{args.model} not built yet -- run `build --variant {args.variant} --model {args.model}`")

    vectors = np.load(vec_path, mmap_mode="r")
    asins = json.loads(ids_path.read_text())
    encoder = load_model(args.max_seq_length, args.model)

    query = encoder.encode(
        [MODELS[args.model]["query_prefix"] + args.query],
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)[0]
    scores = np.asarray(vectors) @ query
    top = np.argsort(-scores)[: args.k]

    titles = {}
    with Path("data/catalog.jsonl").open(encoding="utf-8") as handle:
        wanted = {asins[i] for i in top}
        for line in handle:
            row = json.loads(line)
            if row["parent_asin"] in wanted:
                titles[row["parent_asin"]] = row["title"]

    for rank, index in enumerate(top, 1):
        asin = asins[index]
        print(f"{rank:2}. {scores[index]:.3f}  {asin}  {titles.get(asin, '')[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--variant", default="v2", choices=sorted(BUILDERS))
    common.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    # Defaults come from ENCODE_DEFAULTS per variant; None means "use the default".
    common.add_argument("--max-seq-length", type=int, default=None)

    build_parser = sub.add_parser("build", parents=[common])
    build_parser.add_argument("--batch-size", type=int, default=None)
    build_parser.add_argument("--checkpoint", type=int, default=2048,
                              help="rows between disk flushes")
    build_parser.add_argument("--no-resume", dest="resume", action="store_false")
    build_parser.set_defaults(func=build, resume=True)

    search_parser = sub.add_parser("search", parents=[common])
    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=10)
    search_parser.set_defaults(func=search)

    args = parser.parse_args()
    defaults = ENCODE_DEFAULTS[args.variant]
    if args.max_seq_length is None:
        args.max_seq_length = defaults["max_seq_length"]
    if getattr(args, "batch_size", None) is None:
        args.batch_size = defaults["batch_size"]
    args.func(args)


if __name__ == "__main__":
    main()
