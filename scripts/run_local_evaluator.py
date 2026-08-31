#!/usr/bin/env python3
"""Run the official local evaluator and save a compact per-run metrics report.

This wrapper deliberately calls `python -m evaluator.local_evaluator` instead of
editing the evaluator. Use it for local optimisation tracking; use the evaluator
directly when you want the untouched official harness command.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _efficiency_from_mttc(mttc: float | None) -> float:
    if mttc is None:
        return 0.0
    return max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))


def _technical_score(metrics: dict) -> float:
    efficiency = _efficiency_from_mttc(metrics.get("mttc"))
    return (
        0.50 * float(metrics.get("hit_rate_at_10") or 0.0)
        + 0.30 * float(metrics.get("mrr") or 0.0)
        + 0.20 * efficiency
    )


def _top_1_hit_rate(sessions: list[dict]) -> float:
    if not sessions:
        return 0.0
    return round(sum(1 for item in sessions if item.get("best_rank") == 1) / len(sessions), 6)


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-")


def report_path_for_run(report_dir: str | Path, config: str | None, timestamp: datetime) -> Path:
    directory = Path(report_dir)
    timestamp_part = timestamp.strftime("%Y%m%d%H%M%S")
    config_part = _safe_filename_part(config or "")
    stem = f"local_{config_part}_{timestamp_part}" if config_part else f"local_{timestamp_part}"
    path = directory / f"{stem}.json"
    suffix = 2
    while path.exists():
        path = directory / f"{stem}_{suffix}.json"
        suffix += 1
    return path


def build_report_entry(
    result: dict,
    *,
    catalog: str,
    dataset: str,
    output: str,
    config: str | None,
    elapsed_seconds: float,
    timestamp: datetime | None = None,
) -> dict:
    timestamp = timestamp or datetime.now().astimezone()
    overall = {
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "top_1_hit_rate": _top_1_hit_rate(result.get("sessions", [])),
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
    }

    sessions_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for session in result.get("sessions", []):
        scenario = session.get("scenario_type")
        if isinstance(scenario, str):
            sessions_by_scenario[scenario].append(session)

    scenario_metrics = {}
    for name, metrics in result.get("scenario_metrics", {}).items():
        scenario_metrics[name] = {
            **metrics,
            "top_1_hit_rate": _top_1_hit_rate(sessions_by_scenario.get(name, [])),
            "efficiency": round(_efficiency_from_mttc(metrics.get("mttc")), 6),
            "technical_score": round(_technical_score(metrics), 6),
        }

    # Keep per-run reports compact: full per-session traces remain in results.json.
    return {
        "run_id": uuid.uuid4().hex,
        "timestamp_local": timestamp.isoformat(timespec="seconds"),
        "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "configuration": config,
        "inputs": {
            "catalog": catalog,
            "dataset": dataset,
            "output": output,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "overall": overall,
        "scenario_metrics": scenario_metrics,
        "reported_token_usage": result["reported_token_usage"],
    }


def write_report_file(path: str | Path, entry: dict) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--report-dir", default="evaluation_reports")
    parser.add_argument("--config", "--report-config", dest="config", default=None,
                        help="configuration name for local_<config>_<timestamp>.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "evaluator.local_evaluator",
        "--catalog",
        args.catalog,
        "--dataset",
        args.dataset,
        "--output",
        args.output,
    ]

    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed_seconds = time.monotonic() - started

    result = json.loads(Path(args.output).read_text(encoding="utf-8"))
    timestamp = datetime.now().astimezone()
    report_path = report_path_for_run(args.report_dir, args.config, timestamp)
    entry = build_report_entry(
        result,
        catalog=args.catalog,
        dataset=args.dataset,
        output=args.output,
        config=args.config,
        elapsed_seconds=elapsed_seconds,
        timestamp=timestamp,
    )
    write_report_file(report_path, entry)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
