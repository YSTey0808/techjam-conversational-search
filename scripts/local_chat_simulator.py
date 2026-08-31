#!/usr/bin/env python3
"""Run local conversations between the Agent and a Groq-simulated customer."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_TOKENS = 160
SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}


def resolve_catalog_path(path: str) -> Path:
    requested = Path(path)
    if requested.exists():
        return requested
    if path == "data/catalog.jsonl" and Path("catalog.jsonl").exists():
        return Path("catalog.jsonl")
    return requested


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_catalog(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def configure_catalog_for_agent(catalog_path: str | Path) -> None:
    # starter.retrieve reads this during import; set it before importing Agent.
    os.environ["TECHJAM_CATALOG"] = str(catalog_path)


def choose_targets(products: list[dict], target_asins: list[str], count: int, seed: int | None) -> list[dict]:
    by_asin = {str(product.get("parent_asin")): product for product in products}
    if target_asins:
        missing = [asin for asin in target_asins if asin not in by_asin]
        if missing:
            raise SystemExit(f"Unknown target ASIN(s): {', '.join(missing)}")
        return [by_asin[asin] for asin in target_asins]
    rng = random.Random(seed)
    return rng.sample(products, k=min(count, len(products)))


def choose_scenarios(count: int, requested: str, seed: int | None) -> list[str]:
    if requested != "mixed":
        return [requested] * count
    rng = random.Random(None if seed is None else seed + 1009)
    names = list(SCENARIO_WEIGHTS)
    weights = [SCENARIO_WEIGHTS[name] for name in names]
    return rng.choices(names, weights=weights, k=count)


def scenario_plan(scenario: str, index: int, seed: int | None) -> dict:
    rng = random.Random(f"{seed}:{index}:{scenario}")
    return {
        "scenario_type": scenario,
        "override_turn": rng.choice([3, 4]) if scenario == "intent_override" else None,
    }


def _flatten(value: object, limit: int) -> list[str]:
    if isinstance(value, dict):
        items = [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    elif isinstance(value, list):
        items = [str(item) for item in value if item not in (None, "")]
    elif value not in (None, ""):
        items = [str(value)]
    else:
        items = []
    return [_clean(item) for item in items[:limit] if _clean(item)]


def _clean(value: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit].rstrip()


def product_profile(product: dict) -> dict:
    return {
        "parent_asin": str(product.get("parent_asin") or ""),
        "title": _clean(str(product.get("title") or "")),
        "store": _clean(str(product.get("store") or "")),
        "price": product.get("price"),
        "categories": _flatten(product.get("categories"), 6),
        "features": _flatten(product.get("features"), 8),
        "description": _flatten(product.get("description"), 3),
        "details": _flatten(product.get("details"), 8),
    }


def public_target_summary(product: dict) -> dict:
    profile = product_profile(product)
    return {
        "parent_asin": profile["parent_asin"],
        "title": profile["title"],
        "store": profile["store"],
        "price": profile["price"],
        "categories": profile["categories"],
    }


def _conversation_brief(turns: list[dict]) -> str:
    lines: list[str] = []
    for item in turns[-4:]:
        turn = item.get("turn")
        customer = item.get("customer")
        agent = item.get("agent_message")
        ask = item.get("agent_ask_attribute")
        if customer:
            lines.append(f"Turn {turn} customer: {customer}")
        if agent:
            lines.append(f"Turn {turn} agent: {agent} (ask_attribute={ask})")
    return "\n".join(lines) if lines else "No previous messages."


def _customer_prompt(
    product: dict,
    turns: list[dict],
    *,
    turn: int,
    agent_message: str | None,
    ask_attribute: str | None,
    plan: dict,
) -> list[dict]:
    profile = product_profile(product)
    hidden_profile = json.dumps(profile, ensure_ascii=True, indent=2)
    scenario = str(plan["scenario_type"])
    override_turn = plan.get("override_turn")
    agent_context = "This is the first message." if turn == 1 else (
        f"Agent message: {agent_message or ''}\nAgent ask_attribute: {ask_attribute}"
    )
    scenario_rules = {
        "buying": (
            "Scenario: buying. On turn 1, say the broad item type and reveal exactly "
            "one important constraint from the hidden profile. On later turns, answer "
            "the agent's ask_attribute with at most one additional supported attribute."
        ),
        "browsing": (
            "Scenario: browsing. On turn 1, be vague and exploratory; do not reveal a "
            "specific constraint yet. On later turns, answer the agent's ask_attribute "
            "with at most one supported attribute."
        ),
        "intent_override": (
            f"Scenario: intent_override. Before turn {override_turn}, mention one vague "
            "preference that is not the strongest requirement. On the override turn, "
            "start with 'Actually, ignore my earlier preference' and reveal exactly one "
            "stronger supported attribute from the hidden profile. After that, answer "
            "ask_attribute normally with at most one attribute."
        ),
        "boundary": (
            "Scenario: boundary. On turn 1, be vague. The first time the agent asks a "
            "specific ask_attribute, say you do not have a strong preference for that "
            "attribute. On later turns, answer with at most one supported attribute."
        ),
    }[scenario]
    boundary_already_used = any(item.get("boundary_deflection") for item in turns)
    boundary_instruction = ""
    if scenario == "boundary" and turn > 1 and ask_attribute and not boundary_already_used:
        boundary_instruction = (
            "For this reply, perform the boundary behavior: say you do not have a strong "
            f"preference for {ask_attribute}. Do not reveal another attribute."
        )
    system = (
        "You are a simulated ecommerce customer. You want the hidden target product, "
        "but you must behave like a real customer who does not know the product ID. "
        "Never mention the parent_asin. Do not copy the full title. Be vague. "
        "Reveal at most one new attribute in this reply. If the agent asks a specific "
        "ask_attribute, answer only that attribute when it is supported by the hidden "
        "profile. If the profile does not support that attribute, say you do not have "
        "a strong preference. Return only one customer message, 25 words or fewer."
    )
    user = (
        f"{scenario_rules}\n"
        f"{boundary_instruction}\n\n"
        f"Hidden target product profile:\n{hidden_profile}\n\n"
        f"Recent conversation:\n{_conversation_brief(turns)}\n\n"
        f"{agent_context}\n\n"
        "Write the next customer message."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def groq_payload(messages: list[dict], *, model: str, temperature: float, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.startswith("openai/gpt-oss"):
        payload["include_reasoning"] = False
        payload["reasoning_effort"] = "low"
    elif model.startswith("qwen/qwen3."):
        payload["reasoning_format"] = "hidden"
        payload["reasoning_effort"] = "none"
    return payload


def call_groq(messages: list[dict], *, model: str, temperature: float, timeout: float, max_tokens: int) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing. Set it in your shell or .env.")

    payload = json.dumps(groq_payload(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )).encode("utf-8")
    request = urllib.request.Request(
        GROQ_CHAT_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "techjam-local-chat-simulator/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Groq API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Groq API request failed: {exc}") from exc

    choices = data.get("choices")
    if not choices:
        raise SystemExit(f"Groq API returned no choices: {data}")
    choice = choices[0]
    content = choice.get("message", {}).get("content", "")
    text = str(content).strip()
    if not text:
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage")
        raise SystemExit(
            "Groq returned an empty customer message "
            f"(finish_reason={finish_reason!r}, usage={usage!r}). "
            "Try --model qwen/qwen3.6-27b or increase --max-tokens."
        )
    return text


def sanitize_customer_message(message: str, product: dict) -> str:
    text = message.strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    asin = str(product.get("parent_asin") or "")
    if asin:
        text = text.replace(asin, "[product id hidden]")
    title = str(product.get("title") or "").strip()
    if title and title.lower() in text.lower():
        text = re.sub(re.escape(title), "this item", text, flags=re.I)
    return text[:500].rstrip()


def recommendation_ids(response: Any) -> list[str]:
    if not isinstance(response, dict):
        return []
    ids = []
    for item in response.get("recommendations") or []:
        if isinstance(item, dict):
            value = item.get("parent_asin")
        else:
            value = item
        if value:
            ids.append(str(value))
    return ids


def compact_turns_for_transcript(turns: list[dict]) -> list[dict]:
    compacted: list[dict] = []
    final_index = len(turns) - 1
    for index, turn in enumerate(turns):
        item = dict(turn)
        item.pop("target_hit", None)
        item.pop("target_rank", None)
        if index != final_index:
            item.pop("agent_recommendations", None)
        compacted.append(item)
    return compacted


def session_metrics(hit_turn: int | None, hit_rank: int | None, max_turns: int) -> dict:
    hit = hit_turn is not None
    return {
        "hit": hit,
        "first_hit_turn": hit_turn,
        "mttc_turn": hit_turn if hit else max_turns + 1,
        "best_rank": hit_rank,
        "reciprocal_rank": 0.0 if hit_rank is None else round(1.0 / hit_rank, 6),
    }


def aggregate_metrics(sessions: list[dict], max_turns: int) -> dict:
    if not sessions:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
        }
    count = len(sessions)
    hit_rate = sum(1 for session in sessions if session["metrics"]["hit"]) / count
    mrr = sum(float(session["metrics"]["reciprocal_rank"]) for session in sessions) / count
    mttc = sum(float(session["metrics"].get("mttc_turn", max_turns + 1)) for session in sessions) / count
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def aggregate_scenario_metrics(sessions: list[dict], max_turns: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        scenario = session.get("scenario_type")
        if isinstance(scenario, str):
            grouped[scenario].append(session)
    return {
        scenario: aggregate_metrics(items, max_turns)
        for scenario, items in sorted(grouped.items())
    }


def run_one_conversation(agent: Any, product: dict, args: argparse.Namespace, index: int, plan: dict) -> dict:
    session_id = f"local_chat_{uuid.uuid4().hex}"
    target_asin = str(product.get("parent_asin") or "")
    agent.reset(session_id, {
        "purchase_frequency": "local simulator",
        "average_prior_rating": None,
        "rating_style": "unknown",
        "preference_tags": [],
        "summary": "Local LLM simulated customer.",
    })

    print(f"\n=== Conversation {index} ===")
    print(f"Scenario: {plan['scenario_type']}")
    if plan.get("override_turn"):
        print(f"Intent override turn: {plan['override_turn']}")
    print("Target product:")
    print(json.dumps(public_target_summary(product), indent=2, ensure_ascii=True))

    turns: list[dict] = []
    user_message = ""
    hit_turn: int | None = None
    hit_rank: int | None = None
    started = time.monotonic()

    for turn in range(1, args.max_turns + 1):
        answered_attribute = None
        if turn == 1:
            messages = _customer_prompt(
                product,
                turns,
                turn=turn,
                agent_message=None,
                ask_attribute=None,
                plan=plan,
            )
        else:
            previous = turns[-1]
            answered_attribute = previous.get("agent_ask_attribute")
            messages = _customer_prompt(
                product,
                turns,
                turn=turn,
                agent_message=previous.get("agent_message"),
                ask_attribute=previous.get("agent_ask_attribute"),
                plan=plan,
            )
        user_message = sanitize_customer_message(
            call_groq(
                messages,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            ),
            product,
        )
        print(f"\nCustomer LLM [turn {turn}]: {user_message}")

        response = agent.respond(session_id, user_message, turn, 10)
        ids = recommendation_ids(response)
        if target_asin in ids:
            hit_turn = turn
            hit_rank = ids.index(target_asin) + 1

        agent_message = str(response.get("message") or "") if isinstance(response, dict) else ""
        ask_attribute = response.get("ask_attribute") if isinstance(response, dict) else None
        print(f"Agent: {agent_message}")
        print(f"Agent ask_attribute: {ask_attribute}")
        print(f"Agent recommendations: {ids[:10]}")
        if hit_rank is not None:
            print(f"Target hit at rank {hit_rank} on turn {turn}.")

        turns.append({
            "turn": turn,
            "customer": user_message,
            "agent_message": agent_message,
            "agent_ask_attribute": ask_attribute,
            "agent_recommendations": ids[:10],
            "boundary_deflection": (
                plan["scenario_type"] == "boundary"
                and bool(answered_attribute)
                and "strong preference" in user_message.lower()
            ),
            "target_hit": hit_rank is not None,
            "target_rank": hit_rank,
        })

        if hit_rank is not None:
            break

    metrics = session_metrics(hit_turn, hit_rank, args.max_turns)
    return {
        "session_id": session_id,
        "scenario_type": plan["scenario_type"],
        "scenario_plan": plan,
        "target": public_target_summary(product),
        "metrics": metrics,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "turns": compact_turns_for_transcript(turns),
    }


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-")


def transcript_path(directory: str | Path, config: str | None, timestamp: datetime) -> Path:
    config_part = _safe_filename_part(config or "")
    stamp = timestamp.strftime("%Y%m%d%H%M%S")
    stem = f"chat_{config_part}_{stamp}" if config_part else f"chat_{stamp}"
    path = Path(directory) / f"{stem}.json"
    suffix = 2
    while path.exists():
        path = Path(directory) / f"{stem}_{suffix}.json"
        suffix += 1
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--count", type=int, default=5,
                        help="number of random target products to test")
    parser.add_argument("--target-asin", action="append", default=[],
                        help="handpick a target product; repeat for multiple products")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", default="mixed", choices=("mixed", *SCENARIOS),
                        help="force one scenario, or use the competition-style weighted mix")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config", default=None,
                        help="optional label used in the transcript filename")
    parser.add_argument("--transcript-dir", default="chat_transcripts")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    catalog_path = resolve_catalog_path(args.catalog)
    products = load_catalog(catalog_path)
    targets = choose_targets(products, args.target_asin, args.count, args.seed)
    scenarios = choose_scenarios(len(targets), args.scenario, args.seed)
    configure_catalog_for_agent(catalog_path)

    from starter.agent import Agent

    agent = Agent(str(catalog_path))
    timestamp = datetime.now().astimezone()
    run = {
        "run_id": uuid.uuid4().hex,
        "timestamp_local": timestamp.isoformat(timespec="seconds"),
        "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "catalog": str(catalog_path),
        "model": args.model,
        "configuration": args.config,
        "scenario_mode": args.scenario,
        "sessions": [],
    }

    for index, (product, scenario) in enumerate(zip(targets, scenarios), start=1):
        plan = scenario_plan(scenario, index, args.seed)
        run["sessions"].append(run_one_conversation(agent, product, args, index, plan))

    run["metrics"] = aggregate_metrics(run["sessions"], args.max_turns)
    run["scenario_metrics"] = aggregate_scenario_metrics(run["sessions"], args.max_turns)
    print("\nSimulator metrics:")
    print(json.dumps(run["metrics"], indent=2, ensure_ascii=True))
    print("Simulator scenario metrics:")
    print(json.dumps(run["scenario_metrics"], indent=2, ensure_ascii=True))

    if not args.no_save:
        path = transcript_path(args.transcript_dir, args.config, timestamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"\nSaved transcript: {path}")


if __name__ == "__main__":
    main()
