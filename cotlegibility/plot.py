"""One Matplotlib timeline: mean reference surprisal and cluster-bootstrap CIs."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from .report import quantile
from .schema import digest, read_jsonl, write_jsonl


def bootstrap(values, resamples=10_000, seed=0):
    """Percentile CI for a mean. Same seed/order gives paired resampling across models."""
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    means = [mean(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return quantile(means, .025), quantile(means, .975)


def summarize(rows, models, *, source="controlled", channel="subagent_prompt", unit="context", resamples=10_000, seed=0):
    if unit not in {"context", "session"}:
        raise ValueError("Unit must be context or session")
    catalog = {}
    for model in models:
        if model["id"] in catalog:
            raise ValueError(f"Duplicate release metadata: {model['id']}")
        if model.get("release_date"):
            date.fromisoformat(model["release_date"])
            if source != "synthetic_sanity" and not model.get("release_source", "").startswith("https://"):
                raise ValueError(f"A dated model needs a source URL: {model['id']}")
        catalog[model["id"]] = model
    grouped, coverage, excluded, contexts = defaultdict(list), Counter(), Counter(), defaultdict(set)
    seen = {}
    for row in rows:
        if row["id"] in seen:
            if seen[row["id"]] != row:
                raise ValueError(f"Conflicting measured event: {row['id']}")
            continue
        seen[row["id"]] = row
        if row["source"] != source or row["channel"] != channel:
            excluded["other_source_or_channel"] += 1
            continue
        coverage[row["model"]] += 1
        model = catalog.get(row["model"])
        if not model or not model.get("release_date"):
            excluded["missing_release_metadata"] += 1
            continue
        bpc = row.get("bits_per_char")
        if row.get("analysis_status") != "ok" or not isinstance(bpc, (int, float)) or not math.isfinite(bpc) or bpc < 0:
            excluded["unscored_or_failed"] += 1
            continue
        if not row.get(f"{unit}_id"):
            excluded[f"missing_{unit}_id"] += 1
            continue
        grouped[row["model"]].append(row)
        if unit == "context" and row.get("meta", {}).get("context_hash"):
            contexts[row["context_id"]].add(row["meta"]["context_hash"])
    if any(len(hashes) > 1 for hashes in contexts.values()):
        raise ValueError("A context ID refers to different task contents; use unique context IDs")
    # Never silently merge generation configurations or returned snapshots into one point.
    protocol = set()
    for model, observations in grouped.items():
        signatures = {digest({k: (r.get("meta") or {}).get(k) for k in ("parameters", "served_model", "protocol")}) for r in observations}
        if len(signatures) != 1:
            raise ValueError(f"Mixed settings or served snapshots for {model}; select one configuration before plotting")
        protocol.update((r.get("meta") or {}).get("protocol", "observed") for r in observations)
    if len(protocol) > 1:
        raise ValueError("Mixed collection protocols; plot one protocol at a time")
    by_unit = {}
    for model, observations in grouped.items():
        units = defaultdict(list)
        for row in observations:
            units[row[f"{unit}_id"]].append(row)
        by_unit[model] = units
    # Context comparisons use the intersection across every dated, requested model.
    # A model with zero valid observations prevents a misleading partial comparison.
    requested = [m for m in catalog if catalog[m].get("release_date")]
    shared = set.intersection(*(set(by_unit.get(m, {})) for m in requested)) if requested else set()
    points, warnings = [], []
    for model in requested:
        units = by_unit.get(model, {})
        selected = sorted(shared if unit == "context" else units)
        if not selected:
            continue
        values = [mean(r["bits_per_char"] for r in units[u]) for u in selected]
        lo, hi = bootstrap(values, resamples, seed)
        used = [r for u in selected for r in units[u]]
        timestamps = sorted(r["timestamp"] for r in used if r.get("timestamp"))
        points.append({**catalog[model], "mean_bpc": mean(values), "ci_low": lo, "ci_high": hi,
            "n_units": len(values), "n_messages": len(used), "n_available_units": len(units),
            "n_selected_messages": coverage[model], "unit_scores": dict(zip(selected, values)),
            "sample_ids": [r["id"] for r in used], "parameters": used[0].get("meta", {}).get("parameters", {}),
            "served_model": used[0].get("meta", {}).get("served_model"),
            "collected_from": timestamps[0] if timestamps else None, "collected_to": timestamps[-1] if timestamps else None})
    if unit == "context":
        warnings.append("Only tasks with a scored observation for every dated model are included; missingness can still bias the comparison.")
    else:
        warnings.append("Observational comparison: sessions and workloads are not matched across models. Settings may be unrecorded.")
    if any(p["n_units"] < 10 for p in points):
        warnings.append("Pilot-sized sample: fewer than 10 independent units for at least one model. Interpret intervals cautiously.")
    if any(p["n_units"] == 1 for p in points):
        warnings.append("A single independent unit cannot support a bootstrap interval; its point has no error bar.")
    if not points:
        warnings.append("No comparable model scores yet. Supply scored messages, sourced release dates, and the required context/session IDs.")
    if source in {"synthetic_sanity", "screenshot", "system_card"}:
        warnings.append("Selected or synthetic examples do not estimate model-wide prevalence or a historical trend.")
    return {"points": sorted(points, key=lambda p: (p["release_date"], p["id"])), "warnings": warnings,
            "coverage": dict(coverage), "excluded": dict(excluded), "source": source, "channel": channel,
            "unit": unit, "resamples": resamples, "seed": seed, "metric": "task-averaged BPC" if unit == "context" else "session-averaged BPC",
            "ci_method": "paired task-cluster percentile bootstrap" if unit == "context" else "independent session-cluster percentile bootstrap",
            "confidence_level": .95, "release_metadata": models}


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    color = "#326a88"
    for index, point in enumerate(summary["points"]):
        value, lo, hi = point["mean_bpc"], point["ci_low"], point["ci_high"]
        when = date.fromisoformat(point["release_date"])
        if lo is not None:
            # Center the whisker on the interval, independently of the point estimate.
            ax.errorbar(when, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="none", color=color, capsize=5, lw=1.5)
        ax.plot(when, value, "o", color=color, ms=7)
        ax.annotate(f"{point.get('label', point['id'])}\nn={point['n_units']}", (when, value),
                    xytext=(9, 12 if index % 2 == 0 else -27), textcoords="offset points", fontsize=10)
    dates = [date.fromisoformat(p["release_date"]) for p in summary["points"]]
    if not dates:
        dates = [date.fromisoformat(m["release_date"]) for m in summary["release_metadata"] if m.get("release_date")]
        ax.text(.5, .5, "No comparable model scores yet", transform=ax.transAxes, ha="center", fontsize=12)
    if dates:
        padding = timedelta(days=max(14, (max(dates) - min(dates)).days * .15))
        ax.set_xlim(min(dates) - padding, max(dates) + padding)
    endpoints = [v for p in summary["points"] for v in (p["mean_bpc"], p["ci_low"], p["ci_high"]) if v is not None]
    if endpoints:
        pad = max(.15, (max(endpoints) - min(endpoints)) * .25)
        ax.set_ylim(max(0, min(endpoints) - pad), max(endpoints) + pad)
    synthetic = summary["source"] == "synthetic_sanity"
    ax.set_title("Synthetic preview — not model results" if synthetic else "Agent-text surprisal over model releases", pad=18)
    ax.set_xlabel("Illustrative dates (not model releases)" if synthetic else "Model public-release date", labelpad=12)
    ax.set_ylabel("Mean surprisal (bits / character)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    unit = "task" if summary["unit"] == "context" else "session"
    note = f"95% {unit}-bootstrap CIs · n = {unit}s · higher = more unexpected text"
    if summary["unit"] == "session":
        note += "\nObservational: workloads are not matched across models."
    elif any(p["n_units"] < 10 for p in summary["points"]):
        note += "\nPilot sample: fewer than 10 tasks; intervals are provisional."
    fig.text(.5, .02, note, ha="center", va="bottom", fontsize=9, color="#4a5962")
    fig.tight_layout(rect=(0, .10, 1, 1))
    return fig


def render(directory, models, output=None, **kwargs):
    directory = Path(directory)
    manifest = next(read_jsonl(directory / "manifest.jsonl"))
    if not manifest.get("reference"):
        raise ValueError("This analysis has no reference scores; rerun analyze without --surface-only")
    rows = list(read_jsonl(directory / "measured_messages.jsonl"))
    summary = summarize(rows, models, **kwargs)
    summary.update(reference=manifest["reference"], measured_data_hash=digest(rows), analysis_manifest=manifest)
    output = Path(output) if output else directory / "surprisal.png"
    if output.suffix.lower() != ".png":
        raise ValueError("Plot output must be a .png file")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(summary)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    write_jsonl(output.with_suffix(".jsonl"), [summary])
    print(f"Chart: {output} ({len(summary['points'])} model points)")
    for warning in summary["warnings"]:
        print(f"  {warning}")
    return summary
