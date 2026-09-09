"""Portable local HTML report: distributions, denominators, and exact source passages."""
from __future__ import annotations

import base64
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .schema import digest, read_jsonl


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def group_key(row):
    meta = row.get("meta") or {}
    return (row["source"], row["model"], meta.get("served_model") or "unverified", row["channel"],
            meta.get("protocol", "observed"), json.dumps(meta.get("parameters", {}), sort_keys=True))


def group_label(key):
    source, model, served, channel, protocol, parameters = key
    config = f" · config {digest(parameters)[:6]}" if parameters != "{}" else ""
    served_note = f" · served {served}" if served not in {model, "unverified"} else ""
    return f"{model}{served_note}\n{source} · {channel} · {protocol}{config}"


def summarize(messages, passages):
    groups, by_message = defaultdict(list), defaultdict(list)
    for passage in passages:
        by_message[passage["sample_id"]].append(passage)
    for row in messages:
        groups[group_key(row)].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        chunks = [p for row in rows for p in by_message[row["id"]]]
        judged = [p["judge"] for p in chunks if p.get("judge_status") == "ok"]
        bpc = [r["bits_per_char"] for r in rows if r.get("bits_per_char") is not None]
        summary = dict(zip(("source", "model", "served_model", "channel", "protocol", "parameters"), key))
        summary.update(n_messages=len(rows), n_contexts=len({r["context_id"] for r in rows if r.get("context_id")}),
                       n_sessions=len({r["session_id"] for r in rows if r.get("session_id")}),
                       n_scorable=sum(r["analysis_status"] == "ok" for r in rows), n_reference=len(bpc),
                       n_judged=len(judged), n_passages=len(chunks),
                       n_judge_errors=sum(p.get("judge_status") == "error" for p in chunks),
                       n_judge_budget_skipped=sum(p.get("judge_status") == "budget_skipped" for p in chunks),
                       median_bpc=quantile(bpc, .5), p90_bpc=quantile(bpc, .9),
                       median_spacing_difference=quantile([r["spacing_difference"] for r in rows if r.get("spacing_difference") is not None], .5),
                       median_glued_word_share=quantile([r["glued_word_share"] for r in rows if r.get("glued_word_share") is not None], .5))
        for dim in ("words", "syntax", "meaning"):
            summary[f"{dim}_difficult_fraction"] = sum(j[dim]["grade"] in {"substantial", "unresolved"} for j in judged) / len(judged) if judged else None
        summary["context_missing_fraction"] = sum(j["context_missing"] for j in judged) / len(judged) if judged else None
        summaries.append(summary)
    return groups, summaries


def figure(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("bits_per_char", "Reference surprisal\nbits / character"),
               ("spacing_difference", "Original minus restored surprisal\nbits / original character"),
               ("glued_word_share", "Glued-word share\ndictionary-based estimate")]
    metrics = [(m, title) for m, title in metrics if any(r.get(m) is not None for rows in groups.values() for r in rows)]
    if not metrics:
        return False
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(3, .75 * len(groups) + 1.5)), squeeze=False, sharey=True)
    rng = random.Random(0)
    labels = []
    for y, (key, rows) in enumerate(sorted(groups.items())):
        label = f"{group_label(key)} · n={len(rows)}"
        labels.append(label)
        for ax, (metric, title) in zip(axes[0], metrics):
            values = [r[metric] for r in rows if r.get(metric) is not None]
            ax.scatter(values, [y + rng.uniform(-.14, .14) for _ in values], s=13, color="#246b87", alpha=.45)
            if values:
                ax.plot([median(values)] * 2, [y - .22, y + .22], color="#152f3b", lw=2)
            ax.set_title(title, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=.15)
    axes[0][0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0][0].invert_yaxis()
    fig.suptitle("Message distributions · dots are messages; bars are medians", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def marked(text, spans):
    out, end = [], 0
    for span in sorted(spans, key=lambda s: s["start"]):
        start, stop = span["start"], span["end"]
        if not 0 <= end <= start < stop <= len(text):
            continue
        out.extend([html.escape(text[end:start]), "<mark>", html.escape(text[start:stop]), "</mark>"])
        end = stop
    out.append(html.escape(text[end:]))
    return "".join(out)


def render(directory):
    directory = Path(directory)
    messages = list(read_jsonl(directory / "measured_messages.jsonl"))
    passages = list(read_jsonl(directory / "passages.jsonl"))
    manifest = next(read_jsonl(directory / "manifest.jsonl"))
    groups, summaries = summarize(messages, passages)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]) if summaries else ["source", "model", "n_messages"])
        writer.writeheader()
        writer.writerows(summaries)
    escape = lambda value: html.escape(str(value))
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    body = ["<h1>Intelligibility of agent text</h1>",
            "<p>Exploratory measurements of reference-model surprise, spacing, and reconstruction effort. "
            "These are proxies, not a human-comprehension scale or a probability of losing control.</p>",
            f"<p>{len(messages)} messages · {len(passages)} passages · {escape(manifest['created'])}</p>",
            f"<p>Coverage: {escape(dict(Counter(r['analysis_status'] for r in messages)))}. "
            "Short messages are retained. Empty, failed, and inaccessible messages do not receive zero scores.</p>",
            f"<p>Judge coverage: {escape(dict(Counter(p['judge_status'] for p in passages)))}. "
            f"Served judge models: {escape(dict(Counter(p.get('judge_served_model') or 'unknown' for p in passages if p.get('judge_status') == 'ok')))}. "
            "Check for judge-model changes before comparing cached results.</p>",
            "<p>Sources, channels, served models, and generation settings are grouped separately. "
            "Public examples and synthetic sanity cases do not estimate prevalence. "
            "Messages within a session are not independent replications.</p>"]
    if figure(groups, directory / "distributions.png"):
        encoded = base64.b64encode((directory / "distributions.png").read_bytes()).decode()
        body.append(f'<img alt="Distributions of message metrics" src="data:image/png;base64,{encoded}">')
    body.append("<h2>Coverage and measurements</h2><p>Judge difficulty is the fraction of judged passages rated substantial or unresolved. "
                "Passages from long messages contribute more observations. Unjudged passages are excluded from that denominator.</p><div class='scroll'><table><thead><tr>" +
                "".join(f"<th>{c}</th>" for c in ("Source / model / channel", "Messages", "Contexts / sessions", "Reference n", "Median BPC", "P90 BPC", "Judged / passages", "Words", "Syntax", "Meaning")) + "</tr></thead><tbody>")
    for s in summaries:
        label = group_label(tuple(s[k] for k in ("source", "model", "served_model", "channel", "protocol", "parameters")))
        values = [label, s["n_messages"], f"{s['n_contexts']} / {s['n_sessions']}", s["n_reference"], fmt(s["median_bpc"]), fmt(s["p90_bpc"]),
                  f"{s['n_judged']} / {s['n_passages']}", *[fmt(s[f"{dim}_difficult_fraction"]) for dim in ("words", "syntax", "meaning")]]
        body.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
    body.append("</tbody></table></div><h2>Inspect the messages</h2><p>Open a message to compare its exact source with the prose actually scored. "
                "Yellow marks show detected fused runs; recognized code and file paths are excluded from the reference score. "
                "Passages are scored independently, with approximate target size " + str(manifest["passage_chars"]) + " characters.</p>")
    by_message = defaultdict(list)
    for p in passages:
        by_message[p["sample_id"]].append(p)
    for row in sorted(messages, key=lambda r: (group_key(r), -(r.get("bits_per_char") or 0), r["id"])):
        body.append(f"<details><summary>{escape(row['model'])} · {escape(row['source'])} · {escape(row['channel'])} · "
                    f"BPC {fmt(row.get('bits_per_char'))} · {escape(row['analysis_status'])} · {escape(row['id'])}</summary>")
        body.append(f"<h3>Original message</h3><pre>{escape(row['text'])}</pre>")
        body.append(f"<p>Message BPC {fmt(row.get('bits_per_char'))}; spacing difference {fmt(row.get('spacing_difference'))}; "
                    f"excluded code/path characters {row['excluded_fraction']:.1%}.</p>")
        for p in by_message[row["id"]]:
            body.append(f"<h3>Passage {p['index'] + 1} · source characters {p['start']}–{p['end']} · BPC {fmt(p.get('bits_per_char'))}</h3>"
                        f"<pre>{marked(p['text'], p.get('source_spacing_spans', []))}</pre>"
                        f"<div class='columns'><div><h4>Scored prose</h4><pre>{escape(p['analysis_text'])}</pre></div>"
                        f"<div><h4>Spacing restored</h4><pre>{escape(p.get('restored', ''))}</pre></div></div>"
                        f"<p>Restored BPC {fmt(p.get('restored_bits_per_char'))}; difference on original denominator {fmt(p.get('spacing_difference'))}. "
                        f"Judge: {escape(p['judge_status'])}.</p>")
            if p.get("judge"):
                body.append(f"<pre>{escape(json.dumps(p['judge'], indent=2, ensure_ascii=False))}</pre>")
            if p.get("judge_error"):
                body.append(f"<p>{escape(p['judge_error'])}</p>")
        body.append(f"<details><summary>Provenance</summary><pre>{escape(json.dumps({k: row.get(k) for k in ('timestamp', 'context_id', 'session_id', 'meta')}, indent=2, ensure_ascii=False))}</pre></details></details>")
    body.append(f"<details><summary>Reproduction manifest</summary><pre>{escape(json.dumps(manifest, indent=2))}</pre></details>")
    css = "body{font:16px/1.55 system-ui,sans-serif;color:#25343b;max-width:1400px;margin:32px auto;padding:0 24px}h1,h2,h3{line-height:1.25}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f6;padding:16px;font:14px/1.6 ui-monospace,monospace}details{border-top:1px solid #d5dfe2;padding:14px 0}summary{cursor:pointer}mark{background:#ffe39c}.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #d5dfe2}th{white-space:nowrap}"
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent-text intelligibility</title><style>' + css + "</style><body>" + "\n".join(body) + "</body></html>"
    (directory / "report.html").write_text(document, encoding="utf-8")
    print(f"Report: {directory / 'report.html'}")
