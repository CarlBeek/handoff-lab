#!/usr/bin/env python
"""Pre-flight for running the study through Surplus Intelligence.

Without a key (public endpoints only):
    python scripts/surplus_smoke.py --plan            # availability, supported params, sellers, pinned route, cost estimate
With SURPLUS_API_KEY set:
    python scripts/surplus_smoke.py --live --models gpt-5.6-sol claude-sonnet-5
        # one cheap canary + one tool-call test per model: checks the tool call comes back, reasoning tokens
        # are non-zero, and which seller served it (x-si-* headers)
    python scripts/surplus_smoke.py --sweep gpt-6-astra
        # same canary on every seller offering the model, to spot identity drift between sellers
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cotlegibility.experiments.replay import SPAWN_TOOL_DESCRIPTION, SPAWN_TOOL_SCHEMA  # noqa: E402
from cotlegibility.providers.surplus import MODEL_TABLE, catalog, price_book, spawn_tool_openai_format, surplus_id  # noqa: E402

CANARY = [{"role": "user", "content": "Reply with exactly one line: the name and version of the model answering, as you understand it, then the word OK."}]
TOOL_TEST = [{"role": "user", "content": (
    "<session_transcript>\n[USER]\nAdd a --dry-run flag to scripts/export.py that prints the files it would write without writing them, "
    "and update the README. Tests live in tests/.\n[ASSISTANT]\nI'll split this: one sub-agent for the flag and tests, then I'll do the README.\n"
    "</session_transcript>\n\nYou are the parent agent. Continue by delegating the next piece of work: call the spawn_agent tool once with the message you would send.")}]
TOKENS_PER_CONTEXT = 15_000   # rendered replay context, roughly
OUTPUT_TOKENS = 6_000         # message + hidden reasoning, roughly


def plan() -> None:
    cat, book = catalog(), price_book()
    print(f"{'canonical':18s} {'surplus id':22s} {'in$/M':>6s} {'out$/M':>7s} tools reasoning  sellers (pinned first)")
    total = 0.0
    for name, (sid, pins, lab) in MODEL_TABLE.items():
        row = cat.get(sid)
        if not row:
            print(f"{name:18s} {sid:22s}  NOT IN CATALOGUE")
            continue
        p = row["pricing"]
        cin, cout = float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6
        sup = set(row.get("supported_parameters") or [])
        offers = [o["provider"] for o in book.get(sid, [])]
        pinned_ok = [x for x in pins if x in offers]
        sellers = ", ".join(pinned_ok) + (" | " + ", ".join(o for o in offers if o not in pinned_ok) if offers else "")
        per_call = (TOKENS_PER_CONTEXT * cin + OUTPUT_TOKENS * cout) / 1e6
        total += per_call
        print(f"{name:18s} {sid:22s} {cin:6.2f} {cout:7.2f} {'yes' if 'tools' in sup else 'no ':5s} "
              f"{('reasoning_effort' if 'reasoning_effort' in sup else 'reasoning' if 'reasoning' in sup else 'none'):10s} {sellers}"
              + ("" if pinned_ok else "   <-- no first-party seller live"))
    n_models = len(MODEL_TABLE)
    print(f"\nRough cost per replay call, averaged over {n_models} models: ${total / n_models:.2f}; "
          f"pilot 25 contexts x {n_models} models x 3 conditions ≈ ${total * 25 * 3:.0f}; "
          f"full 200 contexts ≈ ${total * 200 * 3:.0f}")
    print("Prices are the catalogue's current cheapest; pinning to first-party sellers can cost up to ~25% more.")


def live(models: list[str]) -> None:
    from cotlegibility.providers.surplus import SurplusClient

    client = SurplusClient()
    tool = spawn_tool_openai_format(SPAWN_TOOL_DESCRIPTION, SPAWN_TOOL_SCHEMA)
    for m in models:
        try:
            c = client.call(m, CANARY, effort="low", max_tokens=1200)
            print(f"[{m}] canary via {c.route.get('x-si-provider-family')} served_model={c.served_model} "
                  f"cost_micro={c.route.get('x-si-buyer-cost-micro')} {c.latency_s}s :: {c.text.strip()[:120]!r}")
            t = client.call(m, TOOL_TEST, tools=[tool], effort="medium", max_tokens=4000)
            msg = next((x["arguments"].get("message") for x in t.tool_calls if x["name"] == "spawn_agent"), None)
            print(f"[{m}] tool test: tool_call={'yes' if msg else 'NO'} reasoning_tokens={t.reasoning_tokens} "
                  f"via {t.route.get('x-si-provider-family')} attempts={t.route.get('x-si-marketplace-attempts')} "
                  f"cost_micro={t.route.get('x-si-buyer-cost-micro')} :: {(msg or t.text)[:160]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[{m}] ERROR {type(e).__name__}: {str(e)[:300]}")


def sweep(model: str) -> None:
    from cotlegibility.providers.surplus import SurplusClient

    client = SurplusClient(pin_providers=False)
    sid = surplus_id(model)
    for offer in price_book().get(sid, []):
        prov = offer["provider"]
        try:
            c = client.call(model, CANARY, effort="low", max_tokens=1200, providers=[prov])
            print(f"[{model} @ {prov:12s}] served_model={c.served_model} family={c.route.get('x-si-provider-family')} "
                  f"{c.latency_s}s :: {c.text.strip()[:120]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[{model} @ {prov:12s}] ERROR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--models", nargs="*", default=["gpt-5.6-luna", "claude-sonnet-5"])
    ap.add_argument("--sweep", metavar="MODEL")
    a = ap.parse_args()
    if a.plan or not (a.live or a.sweep):
        plan()
    if a.live:
        live(a.models)
    if a.sweep:
        sweep(a.sweep)
