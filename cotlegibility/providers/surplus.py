"""Surplus Intelligence adapter: one OpenAI-compatible route to every model in the study.

Surplus (https://www.surplusintelligence.ai) is an order book for spare inference capacity. Requests go
to `https://api.surplusintelligence.ai/v1` with an `inf_...` buyer key and are filled by the cheapest
seller unless routing is pinned. What matters for a measurement study:

* Model identity is NOT attested. The docs say: "No cryptographic or semantic attestation ties a
  response to the advertised model" and "No quality verification. Quantization level, context
  truncation, injected system prompts, and altered sampling defaults are not measured." So we pin
  every request to first-party-hosted sellers (`anthropic`, `openai`, `bedrock`) via the `provider`
  allow-list, record the `x-si-*` route headers, and run identity canaries (see scripts/surplus_smoke.py).
* Reasoning controls are forwarded as sent (`reasoning_effort` or `reasoning: {effort}`), and the docs
  warn that on the direct-OpenAI seller some gpt-5.6 ids get `reasoning_effort` forced to "none" when
  tools are present. Every result therefore records `usage.completion_tokens_details.reasoning_tokens`.
* Tool definitions in OpenAI function format pass through unchanged; arguments come back as JSON strings.
* Unsupported parameters are silently dropped, so consult `/v1/models` `supported_parameters` first.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

SURPLUS_BASE_URL = "https://api.surplusintelligence.ai/v1"
SURPLUS_ANTHROPIC_BASE_URL = "https://api.surplusintelligence.ai/anthropic"   # Anthropic-shaped /v1/messages
LEGACY_THINKING_MODELS = {"claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"}   # budget_tokens era
ROUTE_HEADERS = ("x-si-served-by", "x-si-provider-family", "x-si-marketplace-attempts", "x-si-attempts",
                 "x-si-buyer-cost-micro", "x-request-id")

# canonical name used in our data -> (Surplus model id, preferred sellers in order, lab)
# Preferred sellers are first-party-hosted routes seen in /v1/prices on 2026-09-09; re-check before a run.
MODEL_TABLE: dict[str, tuple[str, list[str], str]] = {
    "gpt-5.4":          ("gpt-5.4",          ["bedrock", "openrouter"], "openai"),
    "gpt-5.5":          ("gpt-5.5",          ["bedrock"],               "openai"),
    "gpt-5.6-luna":     ("gpt-5.6-luna",     ["bedrock", "openrouter"], "openai"),
    "gpt-5.6-terra":    ("gpt-5.6-terra",    ["bedrock", "openrouter"], "openai"),
    "gpt-5.6-sol":      ("gpt-5.6-sol",      ["bedrock", "openrouter"], "openai"),
    "gpt-6-astra":      ("gpt-6-astra",      ["openai"],                "openai"),   # openrouter offer maps to -pro: avoid
    "claude-sonnet-4-5": ("claude-sonnet-4.5", ["anthropic", "bedrock"], "anthropic"),
    "claude-opus-4-5":  ("claude-opus-4.5",  ["anthropic"],             "anthropic"),
    "claude-sonnet-4-6": ("claude-sonnet-4.6", ["anthropic"],           "anthropic"),
    "claude-opus-4-6":  ("claude-opus-4.6",  ["anthropic"],             "anthropic"),
    "claude-opus-4-7":  ("claude-opus-4.7",  ["anthropic", "bedrock"],  "anthropic"),
    "claude-opus-4-8":  ("claude-opus-4.8",  ["anthropic", "bedrock"],  "anthropic"),
    "claude-sonnet-5":  ("claude-sonnet-5",  ["anthropic", "bedrock"],  "anthropic"),
    "claude-opus-5":    ("claude-opus-5",    ["anthropic"],             "anthropic"),
    "claude-fable-5":   ("claude-fable-5",   ["anthropic", "bedrock"],  "anthropic"),
    "claude-fable-5-1": ("claude-fable-5.1", ["anthropic"],             "anthropic"),
    "deepseek-v4-pro":  ("deepseek-v4-pro-0813", ["fireworks", "openrouter"], "deepseek"),
    "kimi-k3":          ("kimi-k3",          ["fireworks", "openrouter"], "moonshot"),
    "qwen3.8":          ("qwen3.8-2.4t-a95b", ["fireworks", "together"], "alibaba"),
    "gpt-oss-120b":     ("openai-gpt-oss-120b", ["fireworks", "together"], "openai-oss"),
}


def surplus_id(model: str) -> str:
    return MODEL_TABLE.get(model, (model, [], ""))[0]


def preferred_providers(model: str) -> list[str]:
    return MODEL_TABLE.get(model, (model, [], ""))[1]


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    """Public model catalogue with supported_parameters / supported_features per model (no key needed)."""
    import httpx

    rows = httpx.get(f"{SURPLUS_BASE_URL}/models", timeout=30).json()["data"]
    return {r["id"]: r for r in rows}


@lru_cache(maxsize=1)
def price_book() -> dict[str, list[dict]]:
    """Public order book: model -> list of {provider, providerModelId, pricing} (no key needed)."""
    import httpx

    book = httpx.get(f"{SURPLUS_BASE_URL}/prices", timeout=30).json()
    return {m["model"]: m.get("providers", []) for m in book["models"]}


@dataclass
class ChatResult:
    model: str
    surplus_model: str
    text: str
    tool_calls: list[dict]
    reasoning_text: str | None
    reasoning_tokens: int | None
    usage: dict
    route: dict
    served_model: str | None
    latency_s: float
    raw: dict = field(default_factory=dict)


def lab_of(model: str) -> str:
    return MODEL_TABLE.get(model, (model, [], "anthropic" if model.startswith("claude") else "openai"))[2]


def to_anthropic_tool(tool: dict) -> dict:
    """OpenAI function-tool format -> Anthropic tool format (name / description / input_schema)."""
    f = tool.get("function", tool)
    return {"name": f["name"], "description": f.get("description", ""), "input_schema": f.get("parameters") or {"type": "object"}}


class SurplusClient:
    """Dispatches Claude models to the Anthropic-shaped surface (so extended thinking is real) and everything
    else to chat completions. `call()` takes OpenAI-format tools and converts as needed."""

    def __init__(self, api_key: str | None = None, base_url: str = SURPLUS_BASE_URL, pin_providers: bool = True):
        from openai import OpenAI

        key = api_key or os.environ.get("SURPLUS_API_KEY")
        if not key:
            raise RuntimeError("SURPLUS_API_KEY is not set (buyer keys start with inf_)")
        self.key = key
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=2, timeout=600)
        self.pin_providers = pin_providers

    def call(self, model: str, messages: list[dict], tools: list[dict] | None = None, effort: str | None = "medium",
             max_tokens: int = 8000, providers: list[str] | None = None, extra: dict | None = None) -> ChatResult:
        if lab_of(model) == "anthropic":
            return self.messages(model, messages, tools=[to_anthropic_tool(t) for t in tools] if tools else None,
                                 effort=effort, max_tokens=max_tokens, providers=providers, extra=extra)
        return self.chat(model, messages, tools=tools, effort=effort, max_tokens=max_tokens, providers=providers, extra=extra)

    def messages(self, model: str, messages: list[dict], system: str | None = None, tools: list[dict] | None = None,
                 effort: str | None = "medium", max_tokens: int = 8000, providers: list[str] | None = None,
                 extra: dict | None = None) -> ChatResult:
        """Anthropic Messages API through Surplus (`/anthropic/v1/messages`). Beta headers are ignored by the
        router, so no fallbacks / display options; thinking is sent as adaptive (budget_tokens on 4.5-era)."""
        import anthropic

        sid = surplus_id(model)
        client = anthropic.Anthropic(api_key=self.key, base_url=SURPLUS_ANTHROPIC_BASE_URL, max_retries=2, timeout=600)
        body: dict = dict(model=sid, max_tokens=max_tokens, messages=messages)
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        if model in LEGACY_THINKING_MODELS:
            body["thinking"] = {"type": "enabled", "budget_tokens": min(8000, max_tokens - 1000)}
        else:
            body["thinking"] = {"type": "adaptive"}
            if effort:
                body["output_config"] = {"effort": effort}
        extra_body: dict = {}
        pins = providers if providers is not None else (preferred_providers(model) if self.pin_providers else None)
        if pins:
            extra_body["provider"] = pins
            extra_body["provider_order"] = pins
        if extra:
            extra_body.update(extra)
        t0 = time.time()
        raw = client.messages.with_raw_response.create(**body, extra_body=extra_body)
        latency = time.time() - t0
        resp = raw.parse()
        headers = {h: raw.headers.get(h) for h in ROUTE_HEADERS}
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"name": b.name, "arguments": b.input} for b in resp.content if b.type == "tool_use"]
        thinking = "\n".join(b.thinking for b in resp.content if b.type == "thinking" and getattr(b, "thinking", None))
        usage = resp.usage.model_dump() if resp.usage else {}
        return ChatResult(model=model, surplus_model=sid, text=text, tool_calls=calls, reasoning_text=thinking or None,
                          reasoning_tokens=None, usage=usage, route=headers, served_model=getattr(resp, "model", None),
                          latency_s=round(latency, 2), raw=resp.model_dump())

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None, effort: str | None = "medium",
             max_tokens: int = 8000, providers: list[str] | None = None, extra: dict | None = None) -> ChatResult:
        sid = surplus_id(model)
        row = catalog().get(sid, {})
        supported = set(row.get("supported_parameters") or [])
        body: dict = dict(model=sid, messages=messages, max_tokens=max_tokens)
        if tools:
            body["tools"] = tools
        extra_body: dict = {}
        if effort:
            if "reasoning_effort" in supported:
                extra_body["reasoning_effort"] = effort
            elif "reasoning" in supported:
                extra_body["reasoning"] = {"effort": effort}
        pins = providers if providers is not None else (preferred_providers(model) if self.pin_providers else None)
        if pins:
            extra_body["provider"] = pins
            extra_body["provider_order"] = pins
        if extra:
            extra_body.update(extra)
        t0 = time.time()
        raw = self.client.chat.completions.with_raw_response.create(**body, extra_body=extra_body)
        latency = time.time() - t0
        resp = raw.parse()
        headers = {h: raw.headers.get(h) for h in ROUTE_HEADERS}
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_unparsed": tc.function.arguments}
            calls.append({"name": tc.function.name, "arguments": args})
        extra_fields = getattr(msg, "model_extra", None) or {}
        reasoning_text = extra_fields.get("reasoning") or extra_fields.get("reasoning_content")
        if isinstance(reasoning_text, list):
            reasoning_text = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in reasoning_text)
        usage = resp.usage.model_dump() if resp.usage else {}
        rt = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        return ChatResult(model=model, surplus_model=sid, text=msg.content or "", tool_calls=calls,
                          reasoning_text=reasoning_text, reasoning_tokens=rt, usage=usage, route=headers,
                          served_model=getattr(resp, "model", None), latency_s=round(latency, 2), raw=resp.model_dump())


def spawn_tool_openai_format(description: str, schema: dict) -> dict:
    return {"type": "function", "function": {"name": "spawn_agent", "description": description, "parameters": schema, "strict": True}}
