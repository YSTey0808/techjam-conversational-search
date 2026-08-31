"""Provider-agnostic LLM client.

One interface, a few implementations:

    AnthropicClient  -- the production provider (needs `anthropic` + a key)
    GroqClient       -- free, OpenAI-compatible; used for local testing
    MockClient       -- returns a canned reply; for unit tests

``get_client()`` picks one from the environment and returns ``None`` when
nothing is configured, so callers keep their existing keyless fallbacks
(``extract`` returns an empty frame, the reranker returns its Stage B order).

Environment:

    TECHJAM_LLM_PROVIDER  anthropic | groq | mock | none   (optional; auto-detected)
    ANTHROPIC_API_KEY / GROQ_API_KEY
    TECHJAM_LLM_MODEL     optional model override; ignored when it does not
                          match the chosen provider

Nothing here is imported at module load by ``extract`` or ``reranker`` -- they
pull it in lazily -- so an unconfigured checkout never needs this file.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

_GROQ_MAX_RETRIES = 5           # 429 / 5xx are retried with backoff before giving up

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"


class LLMError(RuntimeError):
    """Any failure getting a completion: transport, auth, refusal, bad reply."""


@dataclass
class LLMResult:
    text: str
    model: str = ""
    usage: dict = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )


class LLMClient(ABC):
    """One system+user turn in, text out. Raise ``LLMError`` on any failure.

    ``effort`` and ``thinking`` are provider hints -- honoured by Anthropic,
    silently ignored by providers that have no equivalent.
    """

    model: str

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        effort: str | None = None,
        thinking: bool = False,
    ) -> LLMResult:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, *, model: str | None = None, timeout: float = 60.0) -> None:
        import anthropic

        self._anthropic = anthropic
        # max_retries=0: the evaluator imposes no timeout of its own, so wall
        # clock per turn must stay bounded by `timeout` alone.
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        self.model = model if (model and model.startswith("claude")) else _ANTHROPIC_DEFAULT_MODEL

    def complete(self, *, system, user, max_tokens, json_schema=None,
                 temperature=0.0, effort=None, thinking=False) -> LLMResult:
        output_config: dict = {}
        if effort:
            output_config["effort"] = effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as error:                       # noqa: BLE001 - any failure is an LLMError
            raise LLMError(f"anthropic: {error}") from error

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("anthropic: model refused")

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            model=getattr(response, "model", self.model),
            usage={
                "prompt_tokens": max(0, int(getattr(usage, "input_tokens", 0) or 0)),
                "completion_tokens": max(0, int(getattr(usage, "output_tokens", 0) or 0)),
            },
        )


class GroqClient(LLMClient):
    """Groq's OpenAI-compatible chat endpoint. Raw HTTP, no SDK dependency."""

    def __init__(self, *, model: str | None = None, timeout: float = 60.0) -> None:
        self._key = os.environ.get("GROQ_API_KEY", "")
        if not self._key:
            raise LLMError("GROQ_API_KEY is not set")
        self._timeout = timeout
        # A model override only applies when it is not a Claude id handed down
        # from an Anthropic-shaped default.
        self.model = model if (model and not model.startswith("claude")) else _GROQ_DEFAULT_MODEL

    def complete(self, *, system, user, max_tokens, json_schema=None,
                 temperature=0.0, effort=None, thinking=False) -> LLMResult:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Keep reasoning models from prefixing the answer with their scratch
        # work -- the callers json.loads() the reply.
        if self.model.startswith("openai/gpt-oss"):
            payload["include_reasoning"] = False
            payload["reasoning_effort"] = effort or "low"
        elif self.model.startswith("qwen/qwen3"):
            payload["reasoning_format"] = "hidden"
            payload["reasoning_effort"] = "none"
        if json_schema is not None:
            # Groq's json_object mode needs the word "json" somewhere in the
            # prompt; the caller's schema-bearing prompts do not always have it.
            if "json" not in (system + user).lower():
                payload["messages"][0]["content"] = (
                    system + "\n\nReturn your answer as a single valid JSON object."
                )
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            _GROQ_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                # Groq sits behind Cloudflare, which 403s the default
                # `Python-urllib` agent (error 1010).
                "User-Agent": "techjam-llm-client/1.0",
            },
            method="POST",
        )
        body = self._send(request)

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"groq: no completion in response: {body}") from error

        usage = body.get("usage") or {}
        return LLMResult(
            text=content or "",
            model=body.get("model", self.model),
            usage={
                "prompt_tokens": max(0, int(usage.get("prompt_tokens", 0) or 0)),
                "completion_tokens": max(0, int(usage.get("completion_tokens", 0) or 0)),
            },
        )

    def _send(self, request: urllib.request.Request) -> dict:
        """POST with bounded backoff on 429 / 5xx. Raises LLMError on give-up."""
        for attempt in range(_GROQ_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == _GROQ_MAX_RETRIES:
                    detail = error.read().decode("utf-8", "replace")[:200]
                    raise LLMError(f"groq http {error.code}: {detail}") from error
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(30.0, 2.0 * 2 ** attempt)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                if attempt == _GROQ_MAX_RETRIES:
                    raise LLMError(f"groq request failed: {error}") from error
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
        raise LLMError("groq: retries exhausted")       # unreachable


class MockClient(LLMClient):
    """Returns a fixed reply. For unit tests and offline dry runs."""

    model = "mock"

    def __init__(self, text: str = "{}") -> None:
        self._text = text

    def complete(self, *, system, user, max_tokens, json_schema=None,
                 temperature=0.0, effort=None, thinking=False) -> LLMResult:
        return LLMResult(text=self._text, model="mock")


def _load_env() -> None:
    """Best-effort ``.env`` -> ``os.environ``. Existing vars win. No dependency."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    path = next((p / ".env" for p in here.parents if (p / ".env").is_file()), None)
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def get_client(*, model: str | None = None, timeout: float = 60.0) -> LLMClient | None:
    """The configured client, or ``None`` when nothing usable is set up."""
    _load_env()
    provider = (os.environ.get("TECHJAM_LLM_PROVIDER") or "").strip().lower()
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            return None
    if provider in ("none", "off", "mock"):
        return None
    try:
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return None
            return AnthropicClient(model=model, timeout=timeout)
        if provider == "groq":
            return GroqClient(model=model, timeout=timeout)
    except (ImportError, LLMError):
        return None
    return None


__all__ = [
    "LLMClient", "LLMResult", "LLMError",
    "AnthropicClient", "GroqClient", "MockClient", "get_client",
]
