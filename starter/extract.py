"""OWNER A -- turn a free-form sentence into an Extraction.

    extract(message, turn, state) -> Extraction        <- the only public name

Customers write like people: "something waterproof for hiking, nothing over
fifty bucks". They do not use templates, and they do not use catalog wording.

THE VOCABULARY BRIDGE is the hard part of this module. "waterproof" has to
reach products whose features say "Water Resistant". So the model is asked for
`variants` -- likely catalog phrasings -- and each is resolved against the
real index. The first variant with a non-empty posting list becomes
Constraint.key, which is what retrieval actually looks up. An unresolved
constraint keeps key="" and is still carried: it may match semantically later.

ROBUSTNESS. A slow turn can cost the session, so the LLM is on a short leash:
hard timeout, one retry, then a degraded rule-based path. extract() NEVER
raises and never returns an empty-handed Extraction for a non-empty message.

The degraded path is not a token dump -- it is a real rule-based extractor,
because it is also what runs in CI (NullClient) and on any provider outage.

CONFIG (environment only, no secrets in the repo):
    TECHJAM_LLM_PROVIDER   null | ollama | openai      default: null
    TECHJAM_LLM_MODEL      model name
    TECHJAM_LLM_BASE_URL   e.g. http://localhost:11434  or an OpenAI-compatible base
    TECHJAM_LLM_API_KEY    openai-compatible only
    TECHJAM_LLM_TIMEOUT    seconds, default 2.0
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache

from starter import preprocessing
from starter.schema import REACHABLE_ATTRIBUTES, Constraint, Extraction, SessionState

_TIMEOUT = float(os.environ.get("TECHJAM_LLM_TIMEOUT", "2.0"))
_RETRIES = 1
_SOFT_WEIGHT = 0.35          # weight for salvaged tokens in the degraded path
_MAX_CONSTRAINTS = 8


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------

class _NullClient:
    """Always fails. Makes the degraded path the tested path offline."""

    name = "null"

    def complete(self, prompt: str) -> tuple[str, dict]:
        raise RuntimeError("no LLM provider configured")


class _OllamaClient:
    name = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/api/generate"

    def complete(self, prompt: str) -> tuple[str, dict]:
        body = json.dumps({
            "model": self._model, "prompt": prompt,
            "stream": False, "format": "json",
        }).encode()
        request = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload = json.loads(response.read().decode())
        usage = {
            "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
            "completion_tokens": int(payload.get("eval_count") or 0),
        }
        return payload.get("response", ""), usage


class _OpenAICompatClient:
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key

    def complete(self, prompt: str) -> tuple[str, dict]:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(self._url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload = json.loads(response.read().decode())
        raw = payload["choices"][0]["message"]["content"]
        reported = payload.get("usage") or {}
        usage = {
            "prompt_tokens": int(reported.get("prompt_tokens") or 0),
            "completion_tokens": int(reported.get("completion_tokens") or 0),
        }
        return raw, usage


def _client():
    provider = os.environ.get("TECHJAM_LLM_PROVIDER", "null").strip().lower()
    model = os.environ.get("TECHJAM_LLM_MODEL", "")
    base = os.environ.get("TECHJAM_LLM_BASE_URL", "")
    if provider == "ollama":
        return _OllamaClient(model or "llama3.1", base or "http://localhost:11434")
    if provider in ("openai", "openai_compat"):
        return _OpenAICompatClient(
            model or "gpt-4o-mini", base or "https://api.openai.com/v1",
            os.environ.get("TECHJAM_LLM_API_KEY", ""),
        )
    return _NullClient()


# --------------------------------------------------------------------------
# prompt + strict validation
# --------------------------------------------------------------------------

_PROMPT = """You extract shopping requirements from one customer message.

Return ONLY a JSON object, no prose, with exactly these fields:
{{
  "constraints": [
    {{"text": "<what they want, short>",
      "attribute": "material|color|size|style|use_case|budget|feature",
      "hard": true|false,
      "variants": ["<phrasings a product listing might use>", "..."]}}
  ],
  "intent": "buying" | "browsing",
  "override": true|false,
  "no_preference": "<attribute they explicitly do not care about>" or null
}}

Rules:
- "variants" matter most. Give 2-5 literal phrasings a clothing product page
  would use. For "waterproof" give ["water resistant","waterproof","weatherproof"].
- "hard" is true only for firm requirements (a budget cap, a stated must-have).
- "override" is true if they retract or replace something said earlier.
- Prefer few precise constraints over many vague ones.

Turn: {turn}
Already known: {known}
Customer says: {message}
"""


def _prompt_for(message: str, turn: int, known: str) -> str:
    return _PROMPT.format(turn=turn, known=known or "nothing yet", message=message)


def _coerce_attribute(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in REACHABLE_ATTRIBUTES else "feature"


def _validate(raw: str) -> dict | None:
    """Strict schema validation. Returns None on anything unexpected."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if not text.startswith("{"):                      # tolerate fenced output
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("constraints"), list):
        return None

    constraints = []
    for item in data["constraints"][:_MAX_CONSTRAINTS]:
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        variants = item.get("variants")
        variants = [str(v).strip() for v in variants if str(v).strip()] if isinstance(variants, list) else []
        constraints.append({
            "text": body,
            "attribute": _coerce_attribute(item.get("attribute")),
            "hard": bool(item.get("hard")),
            "variants": variants[:6],
        })
    if not constraints:
        return None

    no_preference = data.get("no_preference")
    no_preference = str(no_preference).strip().lower() if isinstance(no_preference, str) else None
    if no_preference not in REACHABLE_ATTRIBUTES:
        no_preference = None
    return {
        "constraints": constraints,
        "intent": "buying" if str(data.get("intent", "")).lower() == "buying" else "browsing",
        "override": bool(data.get("override")),
        "no_preference": no_preference,
    }


@lru_cache(maxsize=512)
def _cached_call(message: str, turn: int, known: str) -> str | None:
    """LLM call behind an LRU. Returns validated JSON text, or None.

    Cached on (message, turn) plus a digest of what we already know, because
    the digest changes the prompt -- caching on the message alone would serve
    a stale reading back into a different conversation.
    """
    client = _client()
    prompt = _prompt_for(message, turn, known)
    for _ in range(_RETRIES + 1):
        try:
            raw, usage = client.complete(prompt)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError, TimeoutError):
            continue
        data = _validate(raw)
        if data is not None:
            data["usage"] = usage
            return json.dumps(data)
    return None


# --------------------------------------------------------------------------
# degraded rule-based path (also the offline/CI path)
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do",
    "does", "for", "from", "get", "give", "has", "have", "i", "im", "in", "is",
    "it", "its", "just", "like", "looking", "me", "my", "need", "of", "on", "one",
    "ones", "or", "please", "really", "show", "so", "some", "something", "that",
    "the", "them", "then", "there", "these", "they", "this", "to", "too", "up",
    "want", "was", "we", "what", "with", "would", "you", "your", "want", "wanna",
    "find", "got", "am", "about", "any", "all", "much", "more", "not", "no",
}
_OVERRIDE_CUES = (
    "actually", "forget", "instead", "never mind", "nevermind", "scratch that",
    "no longer", "changed my mind", "on second thought", "rather than", "not the",
    "drop the", "skip the",
)
_NO_PREF_CUES = (
    "no preference", "doesn't matter", "does not matter", "dont care", "don't care",
    "whatever", "up to you", "your judgment", "your judgement", "either is fine",
    "any is fine", "no strong",
)
_NUMBER_WORDS = {
    "ten": 10, "fifteen": 15, "twenty": 20, "twenty-five": 25, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "seventy-five": 75,
    "eighty": 80, "ninety": 90, "hundred": 100, "two hundred": 200,
}
_BUDGET_CUES = ("under", "below", "less than", "no more than", "cheaper than",
                "max", "budget", "up to", "nothing over", "not over", "within")
_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
              "rayon", "denim", "suede", "fleece", "cashmere", "linen", "mesh")
_COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray",
           "grey", "purple", "yellow", "orange", "navy", "beige", "tan")
_USE_CASES = ("hiking", "running", "gym", "workout", "winter", "summer", "rain",
              "outdoor", "work", "office", "travel", "wedding", "party", "beach",
              "waterproof", "warm", "breathable", "casual", "formal", "athletic")
_SIZE_CUES = ("size", "petite", "plus size", "tall", "wide", "narrow", "slim",
              "loose", "oversized", "fitted", "true to size")

# Free-form word -> phrasings a product listing plausibly uses. The LLM supplies
# these itself; this table only backs the degraded path.
_SYNONYMS = {
    "waterproof": ("water resistant", "waterproof", "weatherproof", "water-resistant"),
    "warm": ("insulated", "fleece", "thermal", "warm"),
    "breathable": ("breathable", "moisture wicking", "mesh"),
    "workout": ("athletic", "performance", "activewear"),
    "gym": ("athletic", "performance", "activewear"),
    "rain": ("water resistant", "rain", "waterproof"),
    "winter": ("insulated", "thermal", "winter"),
    "stretchy": ("spandex", "elastane", "stretch"),
    "comfy": ("comfortable", "soft"),
    "comfortable": ("comfortable", "soft"),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _find_budget(text: str) -> float | None:
    lowered = text.lower()
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)", lowered)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd|quid|pounds)", lowered)
    if match:
        return float(match.group(1))
    if any(cue in lowered for cue in _BUDGET_CUES):
        match = re.search(r"(\d+(?:\.\d+)?)", lowered)
        if match:
            return float(match.group(1))
        for word, value in _NUMBER_WORDS.items():
            if word in lowered:
                return float(value)
    for word, value in _NUMBER_WORDS.items():
        if f"{word} bucks" in lowered or f"{word} dollars" in lowered:
            return float(value)
    return None


def _degraded(message: str, turn: int) -> dict:
    """Rule-based reading. Never raises; never empty for a non-empty message."""
    lowered = (message or "").lower()
    constraints: list[dict] = []
    claimed: set[str] = set()

    def add(text: str, attribute: str, hard: bool, variants=()) -> None:
        if text and text not in claimed and len(constraints) < _MAX_CONSTRAINTS:
            claimed.add(text)
            constraints.append({
                "text": text, "attribute": attribute, "hard": hard,
                "variants": list(variants) or [text],
            })

    budget = _find_budget(lowered)
    if budget is not None:
        add(f"budget around ${budget:g}", "budget", True, [f"budget around ${budget:g}"])

    for word in _COLORS:
        if re.search(rf"\b{word}\b", lowered):
            add(word, "color", True, [f"color: {word}", word])
    for word in _MATERIALS:
        if re.search(rf"\b{word}\b", lowered):
            add(word, "material", True, [word, f"100% {word}"])
    for word in _USE_CASES:
        if re.search(rf"\b{word}\b", lowered):
            add(word, "use_case", True, _SYNONYMS.get(word, (word,)))
    for cue in _SIZE_CUES:
        if cue in lowered:
            add(cue, "size", False, [cue])
            break

    # Anything left over becomes a low-confidence soft constraint, so a word we
    # have no rule for still has a chance of matching catalog text.
    for token in _tokens(lowered):
        if len(constraints) >= _MAX_CONSTRAINTS:
            break
        if len(token) > 3 and token not in _STOPWORDS and token not in claimed:
            add(token, "feature", False, _SYNONYMS.get(token, (token,)))

    if not constraints and message and message.strip():
        add(message.strip()[:80], "feature", False)

    no_preference = None
    if any(cue in lowered for cue in _NO_PREF_CUES):
        no_preference = next(
            (a for a in REACHABLE_ATTRIBUTES if a.replace("_", " ") in lowered), None
        )

    return {
        "constraints": constraints,
        "intent": "buying" if budget is not None else "browsing",
        "override": any(cue in lowered for cue in _OVERRIDE_CUES),
        "no_preference": no_preference,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


# --------------------------------------------------------------------------
# vocabulary bridge
# --------------------------------------------------------------------------

def _resolve(text: str, variants: list[str]) -> str:
    """First phrasing that actually exists in the catalog wins.

    Returns "" when nothing resolves. That is a normal outcome, not an error:
    the constraint is still carried and may be matched semantically later.
    """
    prep = preprocessing.active()
    if prep is None:
        return ""
    for candidate in [*variants, text]:
        key = preprocessing.normalize(candidate)
        if key and prep.lookup(key, broad=True):
            return key
    return ""


def _build(data: dict, turn: int) -> Extraction:
    constraints = [
        Constraint(
            text=item["text"],
            key=_resolve(item["text"], item.get("variants") or []),
            attribute=item["attribute"],
            hard=bool(item["hard"]),
            weight=1.0 if item["hard"] else _SOFT_WEIGHT,
            turn=turn,
        )
        for item in data["constraints"]
    ]
    return Extraction(
        constraints=constraints,
        intent=data.get("intent", "browsing"),
        override=bool(data.get("override")),
        no_preference=data.get("no_preference"),
        usage=dict(data.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0}),
    )


def _known(state: SessionState | None) -> str:
    if state is None or not state.constraints:
        return ""
    return ", ".join(c.text for c in state.constraints[-6:])


def extract(message: str, turn: int, state: SessionState | None = None) -> Extraction:
    """Parse one free-form customer message. Never raises."""
    try:
        text = (message or "").strip()
        if not text:
            return Extraction()
        payload = _cached_call(text, int(turn), _known(state))
        data = json.loads(payload) if payload else _degraded(text, turn)
        return _build(data, turn)
    except Exception:
        # Absolute last resort: a miss is bad, an exception is worse.
        try:
            return _build(_degraded(str(message or ""), turn), turn)
        except Exception:
            return Extraction()
