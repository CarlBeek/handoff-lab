"""Analyze frozen AI/human handoffs locally; never initiate collection or paid calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from notebooks import surprisal
from scripts.collect import AUDIENCE_PROTOCOL, AUDIENCES, digest, read_json
from scripts.handoffs import load_run

VERSION = "audience-analysis-v1"
ASTRA = "gpt-6-astra"
COLUMNS = ["id", "task_id", "model_id", "audience", "context_hash", "repo", "protocol", "split",
           "api", "parameters", "provider", "served_model", "collected_at", "schedule",
           "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
           "reasoning_tokens", "cost_usd", "status", "validation_errors", "text", "prose",
           "bits_per_char", "total_bits", "tokens", "characters", "reference_device"]


def score_rows(rows, device="auto", scorer=None):
    scorer = scorer or surprisal.bits_per_char
    scored = []
    for original in rows:
        row = dict(original)
        if row["status"] == "ok":
            row["prose"] = surprisal.prose_only(row["text"])
            if not row["prose"]:
                row.update(status="no_prose", total_bits=0.0, tokens=0, characters=0)
            else:
                # Local scoring failures stop analysis, never silently exclude observations.
                result = scorer(row["prose"], device=device)
                if (not all(math.isfinite(result[k]) and result[k] >= 0
                            for k in ("bits_per_char", "total_bits", "tokens", "characters"))
                        or result["characters"] != len(row["prose"]) or result["tokens"] < 1
                        or not math.isclose(result["bits_per_char"],
                                            result["total_bits"] / result["characters"], rel_tol=1e-9)):
                    raise ValueError("Invalid or inconsistent reference scores")
                row.update({k: result[k] for k in ("bits_per_char", "total_bits", "tokens", "characters")})
                row["reference_device"] = result.get("device")
        scored.append(row)
    return scored


def summarize(rows, models, resamples=10_000, seed=0, astra_id=ASTRA):
    """Task-weighted effects and contrasts, all using one complete panel and draw matrix."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ids = [m["id"] for m in models]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Need a nonempty, unique model panel")
    samples = pd.DataFrame(rows, columns=COLUMNS)
    if not samples.empty:
        if set(samples["protocol"]) != {AUDIENCE_PROTOCOL} or samples["split"].nunique(dropna=False) != 1:
            raise ValueError("Do not combine protocols or pilot/main splits")
        if not set(samples["split"]) <= {"pilot", "main"}:
            raise ValueError("Unknown split")
        if not set(samples["model_id"]) <= set(ids) or not set(samples["audience"]) <= set(AUDIENCES):
            raise ValueError("Unknown model or audience")
        if samples.duplicated(["task_id", "model_id", "audience"]).any() or samples["id"].duplicated().any():
            raise ValueError("Expected one observation per task/model/audience")
        if (samples.groupby("task_id")["context_hash"].nunique(dropna=False) > 1).any():
            raise ValueError("A task ID refers to different contexts")
        for model_id, group in samples.groupby("model_id"):
            if any(group[k].dropna().nunique() > 1 for k in ("parameters", "served_model", "provider", "api")):
                raise ValueError(f"Mixed settings or served snapshots: {model_id}")
    valid = samples.loc[samples["status"].eq("ok")]
    if not valid.empty and (not np.isfinite(valid["bits_per_char"].to_numpy(dtype=float)).all()
                            or (valid["bits_per_char"] < 0).any()):
        raise ValueError("Invalid reference scores")
    columns = pd.MultiIndex.from_product([ids, AUDIENCES], names=["model_id", "audience"])
    matrix = valid.pivot(index="task_id", columns=["model_id", "audience"], values="bits_per_char")
    matrix = matrix.reindex(columns=columns).dropna().sort_index()
    tasks = sorted(samples["task_id"].unique())
    task_ids = matrix.index.tolist()
    cells = {(r["task_id"], r["model_id"], r["audience"]): r for r in rows}
    exclusions = [{"task_id": task, "missing_cells": [
        {"model_id": model_id, "audience": audience,
         "status": cells.get((task, model_id, audience), {}).get("status", "not_attempted")}
        for model_id in ids for audience in AUDIENCES
        if cells.get((task, model_id, audience), {}).get("status") != "ok"]}
        for task in tasks if task not in task_ids]
    points, contrasts, task_effects = [], [], {}
    if task_ids:
        bpc = matrix.to_numpy(dtype=float).reshape(len(task_ids), len(ids), 2)
        effects = bpc[:, :, 0] - bpc[:, :, 1]
        means = effects.mean(axis=0)
        draws = None
        if len(task_ids) > 1:
            indices = np.random.default_rng(seed).integers(0, len(task_ids), size=(resamples, len(task_ids)))
            draws = effects[indices].mean(axis=1)

        def interval(values):
            if values is None:
                return {"ci_low": None, "ci_high": None}
            low, high = np.quantile(values, [.025, .975])
            return {"ci_low": float(low), "ci_high": float(high)}

        for j, model in enumerate(models):
            points.append({"model_id": model["id"], "label": model["label"],
                "mean_audience_effect": float(means[j]),
                "mean_bpc_AI": float(bpc[:, j, 0].mean()),
                "mean_bpc_human": float(bpc[:, j, 1].mean()), "n_tasks": len(task_ids),
                **interval(draws[:, j] if draws is not None else None)})
        if astra_id in ids:
            a = ids.index(astra_id)
            for j, model_id in enumerate(ids):
                if j != a:
                    contrasts.append({"astra_id": astra_id, "comparator_id": model_id,
                        "mean_difference": float(means[a] - means[j]), "n_tasks": len(task_ids),
                        **interval(draws[:, a] - draws[:, j] if draws is not None else None)})
        task_effects = {task: {model_id: float(effects[i, j]) for j, model_id in enumerate(ids)}
                        for i, task in enumerate(task_ids)}
    coverage = []
    for model_id in ids:
        for audience in AUDIENCES:
            statuses = Counter(cells.get((t, model_id, audience), {}).get("status", "not_attempted") for t in tasks)
            coverage.append({"model_id": model_id, "audience": audience, "planned": len(tasks),
                             "valid_scores": statuses.get("ok", 0), "statuses": dict(statuses)})
    return {"points": points, "astra_contrasts": contrasts, "astra_id": astra_id,
            "astra_comparison_status": "available" if contrasts else
                "no_matched_tasks" if not task_ids else "Astra or comparators not selected",
            "task_ids": task_ids, "task_effects": task_effects,
            "n_planned_tasks": len(tasks), "n_matched_tasks": len(task_ids),
            "excluded_tasks": exclusions, "coverage": coverage,
            "resamples": resamples, "seed": seed, "confidence_level": .95,
            "ci_method": "paired task percentile bootstrap; shared draws for all effects and contrasts",
            "contrast_task_set": "complete panel: both audiences for every selected model",
            "multiplicity": "Nominal individual 95% intervals; no multiple-comparison adjustment"}


def paired_messages(rows, summary, models):
    """Keep every complete pair; select low/middle/high signed effects on primary tasks."""
    cells = {(r["task_id"], r["model_id"], r["audience"]): r for r in rows if r["status"] == "ok"}
    pairs, representatives = [], []
    for model in models:
        group = []
        for task in sorted({r["task_id"] for r in rows}):
            ai, human = (cells.get((task, model["id"], a)) for a in AUDIENCES)
            if ai is None or human is None:
                continue
            pair = {"task_id": task, "model_id": model["id"],
                    "in_primary_panel": task in summary["task_ids"],
                    "audience_effect": ai["bits_per_char"] - human["bits_per_char"],
                    "AI": {k: ai[k] for k in ("id", "text", "prose", "bits_per_char")},
                    "human": {k: human[k] for k in ("id", "text", "prose", "bits_per_char")}}
            pairs.append(pair)
            if pair["in_primary_panel"]:
                group.append(pair)
        ordered = sorted(group, key=lambda p: (p["audience_effect"], p["task_id"]))
        if ordered:
            representatives.extend(ordered[i] for i in sorted({0, len(ordered)//2, len(ordered)-1}))
    return pairs, representatives


def plot_effects(summary, models, split, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, max(4, len(models) * .55 + 2)))
    points = {p["model_id"]: p for p in summary["points"]}
    for i, model in enumerate(models):
        point = points.get(model["id"])
        if point:
            color = "#326a88" if model["id"] == summary["astra_id"] else "#666666"
            if point["ci_low"] is not None:
                ax.plot([point["ci_low"], point["ci_high"]], [i, i], color=color, marker="|", markersize=9)
            ax.plot(point["mean_audience_effect"], i, "o", color=color)
    ax.set_yticks(range(len(models)), [m["label"] for m in models])
    ax.set_ylim(len(models) - .5, -.5)
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set(title=f"Audience effect on handoff predictability · {split}",
           xlabel="Mean GPT-2 BPC difference: AI recipient − human recipient")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    if not points:
        ax.text(.5, .5, "No complete matched tasks yet", transform=ax.transAxes, ha="center")
    note = f"95% paired task-bootstrap intervals · {summary['n_matched_tasks']} complete matched tasks"
    if split == "pilot" or summary["n_matched_tasks"] < 10:
        note += "\nPilot / small sample: provisional intervals."
    note += "\nBPC measures GPT-2 predictability; it does not measure comprehension or efficiency."
    fig.text(.5, .02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def analyze(run, output=None, *, model_ids=None, task_limit=None, device="auto",
            resamples=10_000, seed=0, annotations=None):
    manifest, rows = load_run(run, model_ids, task_limit, expected_protocols={AUDIENCE_PROTOCOL})
    models = manifest["models"] if manifest else read_json(ROOT / "data/models.json")
    split = manifest["split"] if manifest else Path(run).name
    output = Path(output) if output else ROOT / "out" / f"audience-{split}"
    if output.resolve() == Path(run).resolve() or Path(run).resolve() in output.resolve().parents:
        raise ValueError("Write analysis exports outside the raw run directory")
    rows = score_rows(rows, device)
    summary = summarize(rows, models, resamples, seed)
    pairs, representatives = paired_messages(rows, summary, models)
    summary.update(protocol=AUDIENCE_PROTOCOL, split=split, models=models, analysis_version=VERSION,
                   selection={"model_ids": model_ids, "task_limit": task_limit},
                   run_hash=digest(manifest) if manifest else None, samples_hash=digest(rows))
    summary["reference"] = {"model": surprisal.MODEL, "revision": surprisal.REVISION,
        "scorer_version": surprisal.VERSION, "stride": surprisal.STRIDE,
        "device_requested": device, "devices_used": sorted({r["reference_device"] for r in rows if r["reference_device"]}),
        "prose_exclusion_regex": surprisal.NONPROSE.pattern,
        "code_sha256": hashlib.sha256(Path(surprisal.__file__).read_bytes()).hexdigest()}
    summary["software"] = {name: version(name) for name in ("torch", "transformers", "numpy", "pandas", "matplotlib")}
    summary["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["known_cost_usd"] = sum(r["cost_usd"] or 0 for r in rows)
    summary["unknown_cost_attempts"] = sum(r["status"] != "not_attempted" and r["cost_usd"] is None for r in rows)
    from scripts.annotate import annotation_summary, read_jsonl
    summary["annotations"] = annotation_summary(rows, read_jsonl(annotations) if annotations else [], models)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "samples.jsonl", rows)
    write_jsonl(output / "pairs.jsonl", pairs)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_effects(summary, models, split, output / "audience_effect.png")
    parts = ["# Representative paired handoffs\n",
             "Automatically selected low, middle, and high signed audience effects per model, "
             "using only primary-panel tasks (ties: task ID). These are illustrative, not frequency estimates.\n"]
    if not representatives:
        parts.append("No complete primary-panel pairs yet.\n")
    for pair in representatives:
        parts.append(f"## {pair['model_id']} · {pair['task_id']}\n\nAI − human BPC: {pair['audience_effect']:.6f}\n")
        for audience in AUDIENCES:
            text = pair[audience]["text"]
            fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", text)), default=0))
            parts.append(f"### {audience} recipient\n\n{fence}text\n{text}\n{fence}\n")
    (output / "paired_messages.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"{split}: {summary['n_matched_tasks']} / {summary['n_planned_tasks']} complete matched tasks")
    for point in summary["points"]:
        print(json.dumps(point))
    for contrast in summary["astra_contrasts"]:
        print(json.dumps(contrast))
    print(f"Coverage, exclusions, matched task IDs and annotation frequencies: {output / 'summary.json'}")
    print(f"Representative original pairs: {output / 'paired_messages.md'}")
    if split == "pilot" or summary["n_matched_tasks"] < 10:
        print("Pilot / small sample: intervals are provisional; one task has no interval.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "data/raw/audience/pilot")
    parser.add_argument("--out", type=Path, dest="output")
    parser.add_argument("--model", action="append", dest="model_ids")
    parser.add_argument("--tasks", type=int, dest="task_limit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--annotations", type=Path, help="Optional imported labels.jsonl; missing labels stay missing")
    args = parser.parse_args(argv)
    try:
        analyze(**vars(args))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
