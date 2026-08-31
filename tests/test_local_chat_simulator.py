from __future__ import annotations

import tempfile
import unittest
import os
from datetime import datetime
from pathlib import Path

from scripts.local_chat_simulator import (
    aggregate_metrics,
    aggregate_scenario_metrics,
    choose_targets,
    choose_scenarios,
    compact_turns_for_transcript,
    configure_catalog_for_agent,
    groq_payload,
    product_profile,
    recommendation_ids,
    resolve_catalog_path,
    scenario_plan,
    session_metrics,
    transcript_path,
)


class LocalChatSimulatorTest(unittest.TestCase):
    def test_choose_targets_can_handpick_asins(self) -> None:
        products = [{"parent_asin": "A"}, {"parent_asin": "B"}]

        self.assertEqual(choose_targets(products, ["B"], count=5, seed=None), [{"parent_asin": "B"}])

    def test_choose_targets_uses_seeded_random_sample(self) -> None:
        products = [{"parent_asin": str(index)} for index in range(10)]

        first = choose_targets(products, [], count=3, seed=7)
        second = choose_targets(products, [], count=3, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_choose_scenarios_can_force_one_scenario(self) -> None:
        self.assertEqual(choose_scenarios(3, "boundary", seed=7), ["boundary", "boundary", "boundary"])

    def test_choose_scenarios_mixed_is_seeded(self) -> None:
        first = choose_scenarios(5, "mixed", seed=7)
        second = choose_scenarios(5, "mixed", seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_scenario_plan_sets_override_turn_only_for_intent_override(self) -> None:
        override = scenario_plan("intent_override", index=1, seed=7)
        browsing = scenario_plan("browsing", index=1, seed=7)

        self.assertEqual(override["scenario_type"], "intent_override")
        self.assertIn(override["override_turn"], [3, 4])
        self.assertIsNone(browsing["override_turn"])

    def test_product_profile_keeps_compact_catalog_fields(self) -> None:
        profile = product_profile({
            "parent_asin": "A",
            "title": "  Test   Shoe ",
            "store": "Example",
            "price": 12.5,
            "categories": ["Clothing", "Shoes"],
            "features": [" leather upper ", "waterproof"],
            "description": ["long description"],
            "details": {"Department": "mens"},
        })

        self.assertEqual(profile["title"], "Test Shoe")
        self.assertEqual(profile["features"], ["leather upper", "waterproof"])
        self.assertEqual(profile["details"], ["Department: mens"])

    def test_recommendation_ids_accepts_dict_and_string_items(self) -> None:
        ids = recommendation_ids({
            "recommendations": [{"parent_asin": "A"}, "B", {"parent_asin": ""}],
        })

        self.assertEqual(ids, ["A", "B"])

    def test_compact_turns_keeps_final_recommendations_without_per_turn_metrics(self) -> None:
        turns = [
            {"turn": 1, "customer": "x", "agent_recommendations": ["A"], "target_hit": False, "target_rank": None},
            {"turn": 2, "customer": "y", "agent_recommendations": ["B"], "target_hit": True, "target_rank": 1},
        ]

        compacted = compact_turns_for_transcript(turns)

        self.assertNotIn("agent_recommendations", compacted[0])
        self.assertNotIn("target_hit", compacted[0])
        self.assertNotIn("target_rank", compacted[0])
        self.assertEqual(compacted[1]["agent_recommendations"], ["B"])
        self.assertNotIn("target_hit", compacted[1])
        self.assertNotIn("target_rank", compacted[1])

    def test_session_metrics_assigns_max_turn_plus_one_to_miss(self) -> None:
        self.assertEqual(session_metrics(None, None, 10), {
            "hit": False,
            "first_hit_turn": None,
            "mttc_turn": 11,
            "best_rank": None,
            "reciprocal_rank": 0.0,
        })

    def test_session_metrics_records_reciprocal_rank(self) -> None:
        self.assertEqual(session_metrics(3, 2, 10), {
            "hit": True,
            "first_hit_turn": 3,
            "mttc_turn": 3,
            "best_rank": 2,
            "reciprocal_rank": 0.5,
        })

    def test_aggregate_metrics_matches_evaluator_shape(self) -> None:
        sessions = [
            {"metrics": session_metrics(3, 2, 10)},
            {"metrics": session_metrics(None, None, 10)},
        ]

        self.assertEqual(aggregate_metrics(sessions, 10), {
            "sample_count": 2,
            "hit_rate_at_10": 0.5,
            "mrr": 0.25,
            "mttc": 7.0,
        })

    def test_aggregate_scenario_metrics_groups_sessions(self) -> None:
        sessions = [
            {"scenario_type": "buying", "metrics": session_metrics(2, 1, 10)},
            {"scenario_type": "browsing", "metrics": session_metrics(None, None, 10)},
            {"scenario_type": "buying", "metrics": session_metrics(None, None, 10)},
        ]

        self.assertEqual(aggregate_scenario_metrics(sessions, 10), {
            "browsing": {
                "sample_count": 1,
                "hit_rate_at_10": 0.0,
                "mrr": 0.0,
                "mttc": 11.0,
            },
            "buying": {
                "sample_count": 2,
                "hit_rate_at_10": 0.5,
                "mrr": 0.5,
                "mttc": 6.5,
            },
        })

    def test_transcript_path_uses_config_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timestamp = datetime(2026, 8, 31, 22, 8, 22)

            path = transcript_path(directory, "groq/a", timestamp)

            self.assertEqual(path.name, "chat_groq_a_20260831220822.json")

    def test_resolve_catalog_path_falls_back_to_root_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                (root / "catalog.jsonl").write_text("", encoding="utf-8")
                self.assertEqual(resolve_catalog_path("data/catalog.jsonl"), Path("catalog.jsonl"))
            finally:
                os.chdir(old_cwd)

    def test_configure_catalog_for_agent_sets_retrieval_env(self) -> None:
        old_value = os.environ.get("TECHJAM_CATALOG")
        try:
            configure_catalog_for_agent("catalog.jsonl")
            self.assertEqual(os.environ["TECHJAM_CATALOG"], "catalog.jsonl")
        finally:
            if old_value is None:
                os.environ.pop("TECHJAM_CATALOG", None)
            else:
                os.environ["TECHJAM_CATALOG"] = old_value

    def test_groq_payload_reduces_gpt_oss_reasoning_output(self) -> None:
        payload = groq_payload([], model="openai/gpt-oss-20b", temperature=0.7, max_tokens=300)

        self.assertEqual(payload["max_tokens"], 300)
        self.assertFalse(payload["include_reasoning"])
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_groq_payload_hides_qwen_reasoning_output(self) -> None:
        payload = groq_payload([], model="qwen/qwen3.6-27b", temperature=0.7, max_tokens=300)

        self.assertEqual(payload["reasoning_format"], "hidden")
        self.assertEqual(payload["reasoning_effort"], "none")


if __name__ == "__main__":
    unittest.main()
