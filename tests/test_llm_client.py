from __future__ import annotations

import io
import json
import unittest
import urllib.error
from contextlib import contextmanager
from unittest import mock

import llm_client
from llm_client import GroqClient, LLMError, MockClient, get_client


@contextmanager
def _env(**pairs):
    """Set env vars for the block, restore after; also silence .env loading."""
    import os

    saved = {k: os.environ.get(k) for k in pairs}
    os.environ.update({k: v for k, v in pairs.items() if v is not None})
    for k, v in pairs.items():
        if v is None:
            os.environ.pop(k, None)
    with mock.patch.object(llm_client, "_load_env", lambda: None):
        try:
            yield
        finally:
            for k, old in saved.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old


def _fake_urlopen(body: dict):
    @contextmanager
    def opener(request, timeout=None):        # noqa: ARG001
        yield io.BytesIO(json.dumps(body).encode("utf-8"))

    return opener


class GetClientTest(unittest.TestCase):
    def test_no_provider_and_no_keys_yields_none(self) -> None:
        with _env(TECHJAM_LLM_PROVIDER=None, ANTHROPIC_API_KEY=None, GROQ_API_KEY=None):
            self.assertIsNone(get_client())

    def test_explicit_none_provider_yields_none(self) -> None:
        with _env(TECHJAM_LLM_PROVIDER="none", GROQ_API_KEY="x"):
            self.assertIsNone(get_client())

    def test_groq_key_selects_groq(self) -> None:
        with _env(TECHJAM_LLM_PROVIDER=None, ANTHROPIC_API_KEY=None, GROQ_API_KEY="gk"):
            client = get_client()
            self.assertIsInstance(client, GroqClient)

    def test_anthropic_provider_without_key_yields_none(self) -> None:
        with _env(TECHJAM_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=None):
            self.assertIsNone(get_client())


class GroqClientTest(unittest.TestCase):
    BODY = {
        "model": "llama-3.3-70b-versatile",
        "choices": [{"message": {"content": '{"ranking": ["A", "B"]}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }

    def test_complete_parses_text_usage_and_model(self) -> None:
        with _env(GROQ_API_KEY="gk"):
            client = GroqClient()
        with mock.patch.object(llm_client.urllib.request, "urlopen", _fake_urlopen(self.BODY)):
            result = client.complete(system="s", user="u", max_tokens=50)
        self.assertEqual(json.loads(result.text)["ranking"], ["A", "B"])
        self.assertEqual(result.usage, {"prompt_tokens": 12, "completion_tokens": 3})
        self.assertEqual(result.model, "llama-3.3-70b-versatile")

    def test_json_schema_sets_response_format_and_a_json_hint(self) -> None:
        with _env(GROQ_API_KEY="gk"):
            client = GroqClient()
        captured = {}

        @contextmanager
        def capture(request, timeout=None):        # noqa: ARG001
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            yield io.BytesIO(json.dumps(self.BODY).encode("utf-8"))

        with mock.patch.object(llm_client.urllib.request, "urlopen", capture):
            client.complete(system="rank things", user="u", max_tokens=50,
                            json_schema={"type": "object"})
        payload = captured["payload"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("json", payload["messages"][0]["content"].lower())

    def test_a_non_retryable_http_error_becomes_llm_error(self) -> None:
        with _env(GROQ_API_KEY="gk"):
            client = GroqClient()

        def boom(request, timeout=None):        # noqa: ARG001
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b"nope"))

        with mock.patch.object(llm_client.urllib.request, "urlopen", boom):
            with self.assertRaises(LLMError):
                client.complete(system="s", user="u", max_tokens=10)

    def test_a_429_is_retried_then_succeeds(self) -> None:
        with _env(GROQ_API_KEY="gk"):
            client = GroqClient()
        calls = {"n": 0}

        @contextmanager
        def flaky(request, timeout=None):       # noqa: ARG001
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "0"}, io.BytesIO(b""))
            yield io.BytesIO(json.dumps(self.BODY).encode("utf-8"))

        with mock.patch.object(llm_client.time, "sleep", lambda _s: None), \
             mock.patch.object(llm_client.urllib.request, "urlopen", flaky):
            result = client.complete(system="s", user="u", max_tokens=10)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(result.model, "llama-3.3-70b-versatile")

    def test_missing_key_raises(self) -> None:
        with _env(GROQ_API_KEY=None):
            with self.assertRaises(LLMError):
                GroqClient()


class MockClientTest(unittest.TestCase):
    def test_returns_its_canned_text(self) -> None:
        result = MockClient('{"ranking": ["Z"]}').complete(
            system="s", user="u", max_tokens=1
        )
        self.assertEqual(result.text, '{"ranking": ["Z"]}')
        self.assertEqual(result.model, "mock")


if __name__ == "__main__":
    unittest.main()
