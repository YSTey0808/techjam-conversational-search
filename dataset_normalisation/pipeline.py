"""Reshape data/catalog.jsonl into the normalised table.

Two stages, because they run on wildly different clocks:

    rules   all 9 deterministic columns, ~16s for 50k, stdlib only
    labels  style + use_case via the Batch API -- minutes to hours

The batch is the long pole, so `start` submits it FIRST and runs the rules
while it cooks. `finish` collects and merges. Both are resumable: the batch id
and the collected labels are cached on disk, so re-running never re-submits and
never re-spends.

    python -m dataset_normalisation.pipeline start              # submit + rules
    python -m dataset_normalisation.pipeline finish             # collect + merge
    python -m dataset_normalisation.pipeline run                # both, blocking on the batch
    python -m dataset_normalisation.pipeline run --no-llm       # rules only, no SDK needed

Output is JSONL: five columns are list[str], which CSV cannot round-trip.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dataset_normalisation.columns.audience import audience
from dataset_normalisation.columns.budget import budget
from dataset_normalisation.columns.category import product_family
from dataset_normalisation.columns.color import extract_colors
from dataset_normalisation.columns.feature import extract_features
from dataset_normalisation.columns.material import extract_materials
from dataset_normalisation.columns.region import region

CATALOG = Path("data/catalog.jsonl")
STATE = Path("data/.pipeline_batch")      # batch id, so `finish` can find it
LABELS = Path("data/labels.jsonl")        # collected style/use_case cache
OUTPUT = Path("data/catalog_normalised.jsonl")

COLUMNS = [
    "parent_asin", "category", "audience", "brand", "material", "color",
    "feature", "style", "use_case", "budget", "price", "region",
]


def load_catalog(limit=None, path=CATALOG):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# --------------------------------------------------------------------------
# stage 1: rules
# --------------------------------------------------------------------------
def apply_rules(row, with_source=False):
    family, family_src = product_family(row)
    colors, color_src = extract_colors(row)
    band, price = budget(row)
    origin, origin_src = region(row)
    group, audience_src = audience(row)

    record = {
        "parent_asin": row["parent_asin"],
        "category": family,
        "audience": group,
        "brand": (row.get("store") or "").strip().lower(),
        "material": extract_materials(row),
        "color": colors,
        "feature": extract_features(row),
        "style": [],        # filled by stage 2
        "use_case": [],     # filled by stage 2
        "budget": band,
        "price": price,
        "region": origin,
    }
    if with_source:
        record.update({
            "category_source": family_src,
            "color_source": color_src,
            "region_source": origin_src,
            "audience_source": audience_src,
        })
    return record


def run_rules(rows, with_source=False):
    start = time.time()
    records = [apply_rules(r, with_source) for r in rows]
    print(f"  rules: {len(records)} rows in {time.time() - start:.1f}s")
    return records


# --------------------------------------------------------------------------
# stage 2: LLM labels
# --------------------------------------------------------------------------
def _labels_module():
    """Import the batch module lazily -- it needs the anthropic SDK."""
    try:
        from dataset_normalisation.columns import style_use_case
        return style_use_case
    except ImportError as exc:
        print(f"  ! LLM stage unavailable ({exc}). "
              f"style/use_case will stay empty.\n"
              f"    fix: pip install anthropic, and set ANTHROPIC_API_KEY",
              file=sys.stderr)
        return None


def uncached(rows, labels):
    """Rows with no label yet -- never pay twice for the same parent_asin."""
    return [r for r in rows if r["parent_asin"] not in labels]


def _client():
    """(module, client), or (None, None) if the SDK or credentials are missing.

    The SDK raises at client construction when no key is resolvable -- not at
    import -- so catching ImportError alone is not enough to degrade cleanly.
    """
    module = _labels_module()
    if module is None:
        return None, None
    try:
        return module, module.anthropic.Anthropic()
    except Exception as exc:
        print(f"  ! Anthropic client unavailable: {exc.__class__.__name__}. "
              f"style/use_case will stay empty.\n"
              f"    fix: export ANTHROPIC_API_KEY=...", file=sys.stderr)
        return None, None


def submit_batch(rows):
    module, client = _client()
    if client is None:
        return None
    try:
        batch = client.messages.batches.create(
            requests=[module.build_request(r) for r in rows]
        )
    except Exception as exc:
        print(f"  ! batch submit failed: {exc.__class__.__name__}: {exc}\n"
              f"    style/use_case will stay empty",
              file=sys.stderr)
        return None
    STATE.write_text(batch.id)
    print(f"  batch submitted: {batch.id}  ({len(rows)} requests)")
    print(f"  id cached in {STATE}")
    return batch.id


def poll_batch(batch_id, interval=60):
    _, client = _client()
    if client is None:
        return
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except Exception as exc:
            print(f"  ! poll failed: {exc.__class__.__name__}: {exc}",
                  file=sys.stderr)
            return
        counts = batch.request_counts
        print(f"  {batch.processing_status}: processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            return
        time.sleep(interval)


def collect_labels(batch_id, existing=None):
    """Merge this batch's results into the label cache.

    `existing` is carried through so a full run after a pilot keeps the pilot's
    labels instead of clobbering them.
    """
    module, client = _client()
    if client is None:
        return dict(existing or {})
    labels = dict(existing or {})
    new = failed = 0
    try:
        results = list(client.messages.batches.results(batch_id))
    except Exception as exc:
        print(f"  ! collect failed: {exc.__class__.__name__}: {exc}",
              file=sys.stderr)
        return labels
    with open(LABELS, "w") as out:
        for result in results:
            if result.result.type != "succeeded":
                failed += 1
                continue
            text = next((b.text for b in result.result.message.content
                         if b.type == "text"), "")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                failed += 1
                continue
            entry = module.normalise_labels(data)
            labels[result.custom_id] = entry
            new += 1
        for asin, entry in labels.items():
            out.write(json.dumps({"parent_asin": asin, **entry}) + "\n")
    print(f"  labels: {new} new, {failed} failed, "
          f"{len(labels)} cached total -> {LABELS}")
    return labels


def load_label_cache():
    if not LABELS.exists():
        return None
    labels = {}
    with open(LABELS) as fh:
        for line in fh:
            row = json.loads(line)
            labels[row["parent_asin"]] = {
                "style": row.get("style") or [],
                "use_case": row.get("use_case") or [],
            }
    print(f"  labels: {len(labels)} loaded from cache {LABELS}")
    return labels


# --------------------------------------------------------------------------
# stage 3: merge + write
# --------------------------------------------------------------------------
def merge(records, labels):
    if not labels:
        return records
    hit = 0
    for record in records:
        entry = labels.get(record["parent_asin"])
        if entry:
            record["style"] = entry["style"]
            record["use_case"] = entry["use_case"]
            hit += 1
    print(f"  merged: {hit}/{len(records)} rows matched a label")
    return records


def resolve_out(args):
    """Keep a --limit run from silently overwriting the full 50k table."""
    if args.limit and args.out == str(OUTPUT):
        path = OUTPUT.with_suffix(f".limit{args.limit}.jsonl")
        print(f"  (--limit set; writing to {path} to protect {OUTPUT})")
        return path
    return Path(args.out)


def write(records, path=OUTPUT):
    with open(path, "w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} rows -> {path}")


def to_dataframe(records):
    """Return a pandas DataFrame if pandas is installed, else None."""
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.DataFrame(records, columns=list(records[0].keys()))


def summarise(records):
    total = len(records)
    print(f"\n  {'column':16s} {'filled':>8s} {'%':>7s}")
    print("  " + "-" * 33)
    for column in records[0]:
        filled = sum(
            1 for r in records
            if r[column] not in (None, "", [], "unknown")
        )
        print(f"  {column:16s} {filled:8d} {100 * filled / total:6.1f}%")


# --------------------------------------------------------------------------
def cmd_status(args):
    batch_id = args.batch_id or (
        STATE.read_text().strip() if STATE.exists() else None)
    if not batch_id:
        sys.exit("no batch id (run `start` first, or pass --batch-id)")
    _, client = _client()
    if client is None:
        return
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except Exception as exc:
            sys.exit(f"could not retrieve {batch_id}: "
                     f"{exc.__class__.__name__}: {exc}")
        counts = batch.request_counts
        print(f"  {batch.id}  {batch.processing_status}: "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored} expired={counts.expired}")
        if batch.processing_status == "ended":
            exe = os.path.basename(sys.executable) or "python3"
            print(f"  ready: {exe} -m dataset_normalisation.pipeline finish")
            return
        if not args.wait:
            return
        time.sleep(args.interval)


def cmd_start(args):
    rows = load_catalog(args.limit)
    print(f"catalog: {len(rows)} rows")
    if not args.no_llm:
        todo = uncached(rows, load_label_cache() or {})
        if todo:
            submit_batch(todo)                  # long pole first
        else:
            print("  every row already labelled; nothing to submit")
    records = run_rules(rows, args.with_source)
    write(records, resolve_out(args))
    exe = os.path.basename(sys.executable) or "python3"
    tail = f" --limit {args.limit}" if args.limit else ""
    print(f"\nnext: {exe} -m dataset_normalisation.pipeline status --wait"
          f"\n      {exe} -m dataset_normalisation.pipeline finish{tail}")


def cmd_finish(args):
    labels = load_label_cache() or {}
    rows = load_catalog(args.limit)
    todo = uncached(rows, labels)
    if todo:
        batch_id = args.batch_id or (
            STATE.read_text().strip() if STATE.exists() else None)
        if not batch_id:
            sys.exit(f"{len(todo)} rows have no label and there is no batch id "
                     "(pass --batch-id, or run `start` first)")
        labels = collect_labels(batch_id, existing=labels)
    else:
        print("  every row already labelled from cache")
    records = merge(run_rules(rows, args.with_source), labels)
    write(records, resolve_out(args))
    summarise(records)


def cmd_run(args):
    rows = load_catalog(args.limit)
    print(f"catalog: {len(rows)} rows")
    labels = load_label_cache() or {}
    if not args.no_llm:
        todo = uncached(rows, labels)
        if todo:
            print(f"  {len(labels)} already cached, {len(todo)} to label")
            batch_id = submit_batch(todo)
            if batch_id:
                poll_batch(batch_id, args.interval)
                labels = collect_labels(batch_id, existing=labels)
        else:
            print("  every row already labelled from cache")
    records = merge(run_rules(rows, args.with_source), labels)
    write(records, resolve_out(args))
    summarise(records)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--limit", type=int, help="process only the first N rows")
    common.add_argument("--out", default=str(OUTPUT))
    common.add_argument("--with-source", action="store_true",
                        help="emit *_source columns for precision filtering")
    common.add_argument("--no-llm", action="store_true",
                        help="rules only; style/use_case stay empty")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("start", parents=[common],
                   help="submit the batch, then run rules").set_defaults(func=cmd_start)

    finish = sub.add_parser("finish", parents=[common],
                            help="collect labels and merge")
    finish.add_argument("--batch-id")
    finish.set_defaults(func=cmd_finish)

    status = sub.add_parser("status", help="check batch progress")
    status.add_argument("--batch-id")
    status.add_argument("--wait", action="store_true", help="poll until ended")
    status.add_argument("--interval", type=int, default=60)
    status.set_defaults(func=cmd_status)

    run = sub.add_parser("run", parents=[common],
                         help="everything, blocking until the batch ends")
    run.add_argument("--interval", type=int, default=60)
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
