from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.run_local_evaluator import build_report_entry, report_path_for_run, write_report_file


class RunLocalEvaluatorReportTest(unittest.TestCase):
    def test_report_entry_tracks_rates_without_sessions(self) -> None:
        result = {
            "sample_count": 2,
            "hit_rate_at_10": 0.5,
            "mrr": 0.25,
            "mttc": 6.5,
            "efficiency": 0.45,
            "recommended_technical_score": 0.415,
            "reported_token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "scenario_metrics": {
                "buying": {"sample_count": 2, "hit_rate_at_10": 0.5, "mrr": 0.25, "mttc": 6.5},
            },
            "sessions": [
                {"scenario_type": "buying", "best_rank": 1},
                {"scenario_type": "buying", "best_rank": None},
            ],
        }

        entry = build_report_entry(
            result,
            catalog="catalog.jsonl",
            dataset="public_set.jsonl",
            output="results.json",
            config="reranker-v2",
            elapsed_seconds=1.2345,
        )

        self.assertEqual(entry["configuration"], "reranker-v2")
        self.assertEqual(entry["overall"]["top_1_hit_rate"], 0.5)
        self.assertNotIn("sessions", entry)
        self.assertEqual(entry["scenario_metrics"]["buying"]["technical_score"], 0.415)

    def test_report_path_uses_config_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timestamp = datetime(2026, 8, 31, 22, 8, 22)

            path = report_path_for_run(directory, "a", timestamp)

            self.assertEqual(path.name, "local_a_20260831220822.json")

    def test_report_path_sanitizes_config_and_avoids_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timestamp = datetime(2026, 8, 31, 22, 8, 22)
            existing = Path(directory) / "local_reranker_v2_20260831220822.json"
            existing.write_text("{}\n", encoding="utf-8")

            path = report_path_for_run(directory, "reranker/v2", timestamp)

            self.assertEqual(path.name, "local_reranker_v2_20260831220822_2.json")

    def test_write_report_file_writes_one_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports" / "local_a_20260831220822.json"
            write_report_file(path, {"run": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["run"], 1)


if __name__ == "__main__":
    unittest.main()
