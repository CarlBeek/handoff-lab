"""Surplus wire adapter: public quotes, pinned routes, and observable checks."""
from __future__ import annotations

from datetime import date, datetime, timezone
import math
import re
from urllib.parse import urlsplit

BASE = "https://api.surplusintelligence.ai"
ENDPOINTS = {"responses": BASE + "/v1/responses",
             "messages": BASE + "/anthropic/v1/messages"}
# Explicit local approval, not merely the marketplace's mutable trusted label.
PROVIDER_HOSTS = {"openai": "api.openai.com", "anthropic": "api.anthropic.com",
                  "openrouter": "openrouter.ai"}


def choose_route(client, model, pinned=None):
    """Prefer developer-host offers; an existing run never changes provider.

    The cheapest offer on that provider is a quote, not a price or seller lock.
    """
    result = client.get(f"{BASE}/api/markets/{model['model']}", timeout=20)
    result.raise_for_status()
    market = result.json()
    if market.get("model") != model["model"]:
        raise ValueError("Market returned a different model")
    providers = [pinned] if pinned else model["providers"]
    for provider in providers:
        offers = []
        for offer in market.get("offers", []):
            url = urlsplit(offer.get("seller_base_url", ""))
            prices = [offer.get(k) for k in ("effective_input_per_1m", "effective_output_per_1m")]
            if (offer.get("available") is True and offer.get("healthy") is True
                    and offer.get("trusted") is True and url.scheme == "https"
                    and url.hostname == PROVIDER_HOSTS[provider] and not url.username
                    and url.port in (None, 443)
                    and all(type(p) in (int, float) and math.isfinite(p) and p >= 0 for p in prices)):
                offers.append(offer)
        if offers:
            offer = min(offers, key=lambda o: o["effective_input_per_1m"] + o["effective_output_per_1m"])
            return {"provider": provider, "host": PROVIDER_HOSTS[provider],
                    "quoted_at": datetime.now(timezone.utc).isoformat(), "offer_id": offer["id"],
                    "input_usd_per_million": offer["effective_input_per_1m"] / 1_000_000,
                    "output_usd_per_million": offer["effective_output_per_1m"] / 1_000_000}
    raise ValueError(f"No healthy approved offer for {model['model']} on {providers}; no fallback sent")


def cost_micro(headers):
    value = headers.get("x-si-buyer-cost-micro", "")
    return int(value) if isinstance(value, str) and re.fullmatch(r"[0-9]+", value) else None


def saved_headers(headers):
    """Retain provenance/usage headers, never cookies or request credentials."""
    return {k.lower(): v for k, v in headers.items()
            if k.lower().startswith("x-si-") or k.lower() in {"x-request-id", "request-id"}}


def model_matches(served, allowed):
    if not isinstance(served, str):
        return False
    if served in allowed:
        return True
    for model in allowed:
        if served.startswith(model + "-"):
            suffix = served[len(model) + 1:]
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8}", suffix):
                try:
                    date.fromisoformat(suffix)
                    return True
                except ValueError:
                    pass
    return False


def audit(spec, response, headers):
    """Surplus does not expose a full outbound audit; these are observable checks."""
    problems = []
    if headers.get("x-si-served-by") != "marketplace":
        problems.append("missing_or_unexpected_serving_rail")
    if headers.get("x-si-provider-family") != spec["provider"]:
        problems.append("missing_or_unexpected_provider")
    if headers.get("x-si-truncated", "0") != "0":
        problems.append("gateway_truncated")
    changed = {s.strip() for s in headers.get("x-si-adapted-params", "").split(",") if s.strip()}
    if changed - {"cache_control", "prompt_cache_key", "stream_options"}:
        problems.append("adapted_generation_parameters")
    if cost_micro(headers) is None:
        problems.append("missing_or_invalid_cost")
    if not isinstance(response, dict):
        return problems + ["invalid_response"]
    if not model_matches(response.get("model"), spec["response_models"]):
        problems.append("missing_or_unexpected_model")
    echoed = response.get("reasoning")
    if isinstance(echoed, dict) and echoed.get("effort") is not None:
        requested = spec["parameters"].get("reasoning", {}).get("effort")
        if requested is not None and echoed["effort"] != requested:
            problems.append("reasoning_effort_mismatch")
    return problems


def send(client, spec, api_key):
    # Credentials go only to a constant Surplus endpoint, never an offer URL.
    headers = {"Authorization": f"Bearer {api_key}", "x-request-id": spec["id"]}
    if spec["api"] == "messages":
        headers["anthropic-version"] = "2023-06-01"
    reply = client.post(ENDPOINTS[spec["api"]], json=spec["request"], headers=headers)
    retained = saved_headers(reply.headers)
    try:
        response = reply.json()
    except ValueError:
        response = None
    result = {"http_status": reply.status_code, "response": response,
              "response_headers": retained, "cost_micro": cost_micro(retained),
              "request_id": retained.get("x-request-id") or retained.get("request-id")}
    if response is None:
        result["response_text"] = reply.text
    result["validation_errors"] = audit(spec, response, retained)
    result["state"] = "finished" if reply.is_success else "error"
    return result
