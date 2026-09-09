#!/usr/bin/env python
"""Fixed-context replay experiment.

    python scripts/replay_subagent.py --dry-run                       # extract contexts, print sizes, no API calls
    python scripts/replay_subagent.py --models gpt-5.5 claude-opus-5 --audience none human_reads --limit 5

Outputs out/replay_contexts.jsonl (redacted contexts) and out/replay_results.jsonl (one row per cell).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cotlegibility.experiments.replay import build_prompt, extract_contexts, run_cell  # noqa: E402
from cotlegibility.metrics.surface import surface_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--audience", nargs="*", default=["none"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ctxs = extract_contexts()
    # one context per session-spawn pair is fine, but the same session often spawns several near-identical
    # siblings; keep the first spawn per session for the pilot
    seen, uniq = set(), []
    for c in ctxs:
        if c.meta["session"] in seen:
            continue
        seen.add(c.meta["session"])
        uniq.append(c)
    with open(os.path.join(args.out, "replay_contexts.jsonl"), "w") as f:
        for c in ctxs:
            f.write(json.dumps(c.to_dict()) + "\n")
    print(f"{len(ctxs)} spawn contexts from {len(uniq)} sessions")
    for c in uniq[: args.limit]:
        m = surface_metrics(c.original_message)
        print(f"  {c.id} model={c.model_at_spawn} history_items={c.meta['history_items']} transcript_chars={len(c.transcript)} "
              f"original_msg_chars={len(c.original_message)} glued={m['glued_word_share']:.2f} func={m['function_word_rate']:.2f}")
    if args.dry_run or not args.models:
        print("dry run: prompt preview for first context:\n" + build_prompt(uniq[0], "human_reads")[:800] + "\n...")
        return
    with open(os.path.join(args.out, "replay_results.jsonl"), "a") as f:
        for c in uniq[: args.limit]:
            for model in args.models:
                for aud in args.audience:
                    res = run_cell(c, model, aud, args.effort)
                    msg = res.get("message") or ""
                    row = {"context_id": c.id, "model": model, "audience": aud, "effort": args.effort, "route": res.get("route"),
                           "served_model": res.get("served_model"), "reasoning_tokens": res.get("reasoning_tokens"),
                           "message": msg, "reasoning_summary": res.get("reasoning_summary"), **(surface_metrics(msg) if msg else {})}
                    f.write(json.dumps(row) + "\n")
                    rt = res.get("reasoning_tokens")
                    served = (res.get("route") or {}).get("x-si-provider-family") or res.get("route", "direct")
                    print(f"{c.id} {model:22s} {aud:14s} chars={len(msg):5d} glued={row.get('glued_word_share', float('nan')):.2f} "
                          f"func={row.get('function_word_rate', float('nan')):.2f} reasoning_tokens={rt} via={served}")


if __name__ == "__main__":
    main()
