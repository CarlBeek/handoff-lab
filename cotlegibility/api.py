"""Two direct provider adapters. Requests and complete responses are saved by callers."""
from __future__ import annotations

import time

ENDPOINTS = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}


def provider_for(model):
    return "anthropic" if model.startswith("claude") else "openai"


def request(model, messages, *, parameters=None, tool=None, output_schema=None):
    provider = provider_for(model)
    parameters = dict(parameters or {})
    reserved = {"model", "messages", "input", "tools", "text", "system", "store", "stream"}
    if reserved.intersection(parameters):
        raise ValueError(f"Parameters cannot override {sorted(reserved.intersection(parameters))}")
    start = time.monotonic()
    if provider == "openai":
        from openai import OpenAI
        body = {"model": model, "input": messages, "max_output_tokens": 8192, "store": False, **parameters}
        if tool:
            body["tools"] = [{"type": "function", "name": tool["name"], "description": tool["description"],
                              "parameters": tool["schema"], "strict": True}]
        if output_schema:
            body["text"] = {"format": {"type": "json_schema", "name": "intelligibility", "schema": output_schema, "strict": True}}
        client = OpenAI(base_url=ENDPOINTS[provider], max_retries=0, timeout=300)
        response = client.responses.create(**body)
        raw = response.model_dump(mode="json")
        text = response.output_text
        calls = [{"name": x["name"], "arguments": x["arguments"]} for x in raw.get("output", []) if x["type"] == "function_call"]
        stop = raw.get("status")
    else:
        import anthropic
        system = "\n\n".join(m["content"] for m in messages if m["role"] in {"system", "developer"})
        body = {"model": model, "messages": [m for m in messages if m["role"] not in {"system", "developer"}],
                "max_tokens": 8192, **parameters}
        if system:
            body["system"] = system
        if tool:
            body["tools"] = [{"name": tool["name"], "description": tool["description"], "input_schema": tool["schema"]}]
        # JSON is requested by the judge prompt and validated locally for both providers.
        client = anthropic.Anthropic(base_url=ENDPOINTS[provider], max_retries=0, timeout=300)
        with client.messages.stream(**body) as stream:
            response = stream.get_final_message()
        raw = response.model_dump(mode="json")
        text = "".join(b["text"] for b in raw.get("content", []) if b["type"] == "text")
        calls = [{"name": b["name"], "arguments": b["input"]} for b in raw.get("content", []) if b["type"] == "tool_use"]
        stop = raw.get("stop_reason")
    return {"provider": provider, "endpoint": ENDPOINTS[provider], "requested_model": model,
            "served_model": raw.get("model"), "request_id": getattr(response, "_request_id", None),
            "response_id": raw.get("id"), "usage": raw.get("usage"), "stop_reason": stop,
            "latency_s": round(time.monotonic() - start, 3), "text": text, "tool_calls": calls,
            "request": body, "raw": raw}
