from __future__ import annotations

import unittest
from unittest import mock

from llm_client import MockClient
from reranker.reranker import CONFIG, llm_rerank


def _fixtures():
    asins = ["A", "B", "C"]
    shrunk = {"products": [{"parent_asin": a} for a in asins]}
    request = {
        "query": "q",
        "constraints": [],
        "user_profile": {},
        "pool": [{"parent_asin": a, "title": "", "text": ""} for a in asins],
    }
    return request, shrunk


class LLMRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self._mock = CONFIG["llm_mock"]
        CONFIG["llm_mock"] = False

    def tearDown(self) -> None:
        CONFIG["llm_mock"] = self._mock

    def test_a_valid_ranking_is_applied(self) -> None:
        request, shrunk = _fixtures()
        out = llm_rerank(request, shrunk, client=MockClient('{"ranking": ["C", "A", "B"]}'), top_k=3)
        self.assertTrue(out["llm_used"])
        self.assertEqual([p["parent_asin"] for p in out["products"]], ["C", "A", "B"])

    def test_an_unparseable_reply_falls_back_to_stage_b(self) -> None:
        request, shrunk = _fixtures()
        out = llm_rerank(request, shrunk, client=MockClient("not json"), top_k=3)
        self.assertFalse(out["llm_used"])
        self.assertEqual([p["parent_asin"] for p in out["products"]], ["A", "B", "C"])

    def test_mock_mode_never_calls_the_client(self) -> None:
        CONFIG["llm_mock"] = True
        request, shrunk = _fixtures()
        client = mock.Mock()
        out = llm_rerank(request, shrunk, client=client, top_k=3)
        client.complete.assert_not_called()
        self.assertFalse(out["llm_used"])
        self.assertEqual(out["error"], "mock mode")

    def test_no_provider_falls_back(self) -> None:
        request, shrunk = _fixtures()
        with mock.patch("llm_client.get_client", return_value=None):
            out = llm_rerank(request, shrunk, top_k=3)
        self.assertFalse(out["llm_used"])
        self.assertEqual([p["parent_asin"] for p in out["products"]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
