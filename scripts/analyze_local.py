#!/usr/bin/env python
"""Run legibility metrics over everything observable on this machine and draw a first figure.

Sources: Codex CLI rollouts (~/.codex/sessions), Claude Code transcripts (~/.claude/projects),
and hand-transcribed exemplars in data/exemplars/. Writes out/metrics.csv, out/summary.csv,
out/fig_surface.png.

    python scripts/analyze_local.py [--ppl] [--min-chars 200]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cotlegibility.metrics.surface import surface_metrics  # noqa: E402
from cotlegibility.schema import Sample  # noqa: E402
from cotlegibility.sources import claude_code, codex  # noqa: E402

KEY_METRICS = ["glued_word_share", "function_word_rate", "ws_ratio", "dict_word_rate", "chars_per_token"]
ROW_ORDER = [  # (model, channel) rows shown in the figure, top to bottom
    ("human", "user_message"),
    ("gpt-5.5", "subagent_prompt"), ("gpt-5.5", "subagent_reply"), ("gpt-5.5", "assistant_message"),
    ("gpt-5.6-sol", "assistant_message"),
    ("claude-opus-5", "assistant_message"), ("claude-fable-5-1", "subagent_reply"), ("claude-fable-5-1", "reasoning_summary"),
]
PROVIDER_COLOR = {"openai": "#2a78d6", "anthropic": "#eb6834", "human": "#1baf7a", "exemplar": "#e34948"}


def load_exemplars(pattern: str = "data/exemplars/*.json"):
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        yield Sample(**{k: d.get(k) for k in ("id", "provider", "model", "channel", "text", "source", "timestamp", "meta")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl", action="store_true", help="also compute reference-LM bits/char (slow; sampled)")
    ap.add_argument("--ppl-max-per-group", type=int, default=30)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--out", default="out")
    ap.add_argument("--runs", help="harness manifest (out/harness_runs.jsonl): keep only samples from those run dirs and label them")
    args = ap.parse_args()
    runs = {}
    if args.runs and os.path.exists(args.runs):
        for line in open(args.runs):
            r = json.loads(line)
            runs[os.path.realpath(r["run_dir"])] = r
    os.makedirs(args.out, exist_ok=True)

    samples = list(codex.iter_samples()) + list(claude_code.iter_samples()) + list(load_exemplars())
    rows, seen = [], set()
    for s in samples:
        key = (s.provider, s.channel, hashlib.sha1(s.text.strip().encode()).hexdigest())
        if key in seen:      # forked sub-agent threads replay the parent's history verbatim
            continue
        seen.add(key)
        if len(s.text) < args.min_chars and s.source not in ("screenshot", "system_card"):
            continue
        if (s.meta or {}).get("encrypted"):
            continue   # Codex v2 spawn messages are ciphertext; counted separately, never scored
        d = s.to_dict()
        meta = d.pop("meta") or {}
        text = d.pop("text")
        run = None
        if runs:
            cwd = os.path.realpath(meta.get("cwd") or "")
            run = next((r for rd, r in runs.items() if cwd.startswith(rd)), None)
            if run is None and s.source not in ("screenshot", "system_card"):
                continue
        rows.append({**d, "is_exemplar": s.source in ("screenshot", "system_card"), "label": meta.get("label"), "sidechain": meta.get("sidechain"),
                     "harness": (run or {}).get("harness"), "task": (run or {}).get("task"), "condition": (run or {}).get("condition"),
                     **surface_metrics(text), "text_preview": text[:100].replace("\n", " "), "_text": text})
    df = pd.DataFrame(rows)

    if args.ppl:
        from cotlegibility.metrics.perplexity import bits_per_char
        idx = (df[~df.is_exemplar].groupby(["model", "channel"], group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), args.ppl_max_per_group), random_state=0)).index)
        idx = idx.union(df[df.is_exemplar].index)
        vals = {i: bits_per_char(df.at[i, "_text"])["ref_bits_per_char"] for i in idx}
        df["ref_bits_per_char"] = pd.Series(vals)
        KEY_METRICS.append("ref_bits_per_char")

    df.drop(columns=["_text"]).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    summary = (df.groupby(["provider", "model", "channel"])[KEY_METRICS].median().round(3)
                 .join(df.groupby(["provider", "model", "channel"]).size().rename("n")))
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    pd.set_option("display.width", 200)
    print(summary.to_string())
    make_figure(df, os.path.join(args.out, "fig_surface.png"))
    print("wrote", args.out)


def make_figure(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [m for m in KEY_METRICS if m in df.columns]
    labels = {"glued_word_share": "Share of words written\nwithout a preceding space", "function_word_rate": "Function-word rate",
              "ws_ratio": "Whitespace share of characters", "dict_word_rate": "Dictionary-word rate",
              "chars_per_token": "Characters per token\n(o200k)", "ref_bits_per_char": "Bits per character\nunder GPT-2 (lower = more predictable)"}
    rows = [(m, c) for m, c in ROW_ORDER if ((df.model == m) & (df.channel == c)).sum() >= 3]
    ex = df[df.is_exemplar].reset_index(drop=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 0.55 * (len(rows) + len(ex) + 1) + 1.2), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    yticks = list(range(len(rows) + len(ex)))
    ylabels = [f"{m}\n{c.replace('_', ' ')}" for m, c in rows] + [str(l) for l in ex.label]
    for ax, met in zip(axes, metrics):
        ax.set_facecolor("#fcfcfb")
        for y, (m, c) in enumerate(rows):
            g = df[(df.model == m) & (df.channel == c)][met].dropna()
            prov = df[(df.model == m)].provider.iloc[0]
            ax.scatter(g, y + rng.uniform(-0.18, 0.18, len(g)), s=9, color=PROVIDER_COLOR[prov], alpha=0.35, linewidths=0)
            ax.plot([g.median()] * 2, [y - 0.3, y + 0.3], color=PROVIDER_COLOR[prov], lw=2)
        for k, (_, e) in enumerate(ex.iterrows()):
            if pd.notna(e.get(met)):
                ax.scatter([e[met]], [len(rows) + k], s=42, color=PROVIDER_COLOR["exemplar"], zorder=3)
        ax.set_title(labels.get(met, met), fontsize=9, loc="left", color="#0b0b0b")
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8, color="#52514e")
        ax.invert_yaxis() if ax is axes[0] else None
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="x", labelsize=8, colors="#52514e")
        ax.grid(axis="x", color="#e5e4df", lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Legibility of agent text observed locally, by model and channel (dots = messages, bar = median)",
                 fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.text(0.01, 0.005, "Blue = OpenAI (Codex rollouts), orange = Anthropic (Claude Code transcripts), green = human-typed prompts, "
             "red = public exemplars (screenshots / system-card transcripts), one row each.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
