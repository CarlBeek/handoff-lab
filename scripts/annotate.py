"""Optional manual formatting labels: blind CSV export, validated import, and coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.collect import AUDIENCE_PROTOCOL, AUDIENCES, digest

VERSION = "handoff-annotation-v1"
LABELS = ("fused_prose", "other_shorthand", "ordinary_prose", "uncertain")
FIELDS = ["anonymous_id", "text", "label", "notes"]
RUBRIC = """# Blind handoff formatting rubric (v1)

Read the original handoff and enter one label in the CSV's `label` column.
Leave the ID and text unchanged. Notes are optional. A blank label means not yet
annotated; use `uncertain` for a message you examined but cannot confidently classify.

- `fused_prose`: ordinary prose words are joined without expected spaces, such as
  "Inspectthecacheandreportfindings". A clear passage suffices even if other passages
  are ordinary or use shorthand. Do not count code identifiers (`cacheKey`,
  `get_user_id`), paths, URLs, commands, equations, or normal technical terms.
- `other_shorthand`: no clear fused prose, but prose uses conspicuous abbreviations,
  omitted connecting words, or symbolic/telegraphic substitutions (for example,
  "chk refresh; rpt evidence; no edits"). Ordinary bullets, concise grammatical
  instructions, conventional acronyms, and technical terminology alone do not count.
- `ordinary_prose`: interpretable conventional prose with neither of the above
  features. Length is irrelevant; ordinary prose can be short or long.
- `uncertain`: ambiguous fusion/shorthand, insufficient prose (including all-code
  messages), unfamiliar language, or another reason you cannot confidently classify.

Apply the categories in this order: clear fused prose, clear other shorthand,
ordinary prose; use uncertainty when a decision is not supported. This precedence
makes labels mutually exclusive; notes can record mixed features. These labels
describe visible formatting, not comprehension, usefulness, or efficiency.

Only anonymous IDs and original text are provided. Order is reproducibly shuffled
by anonymous ID. Do not look at the separate key, raw runs, plots, or model/audience
metadata until labels are saved. Text itself may reveal the recipient or model;
metadata blinding cannot prevent that. Record suspected unblinding in notes.
Share only this rubric and blind.csv with the annotator, keeping the key separately.
"""


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def identity(row):
    return {**{k: row[k] for k in ("id", "task_id", "model_id", "audience", "context_hash", "protocol", "split")},
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest()}


def eligible(samples):
    if len({r["id"] for r in samples}) != len(samples):
        raise ValueError("Duplicate sample identities")
    if samples and (set(r["protocol"] for r in samples) != {AUDIENCE_PROTOCOL}
                    or len({r["split"] for r in samples}) != 1
                    or not {r["audience"] for r in samples} <= set(AUDIENCES)):
        raise ValueError("Annotation requires one audience protocol/split")
    result = {r["id"]: r for r in samples if r["status"] in {"ok", "no_prose"}}
    if any(not isinstance(r.get("text"), str) or not r["text"].strip() for r in result.values()):
        raise ValueError("Annotatable samples need original handoff text")
    return result


def anonymous_id(request_id, seed):
    return digest({"version": VERSION, "seed": seed, "request_id": request_id})[:24]


def export_annotations(samples, output, mapping, seed=0):
    originals = eligible(samples)
    if not originals:
        raise ValueError("No valid original handoffs to annotate")
    entries = sorted(({"anonymous_id": anonymous_id(r["id"], seed), **identity(r)}
                      for r in originals.values()), key=lambda r: r["anonymous_id"])
    if len({r["anonymous_id"] for r in entries}) != len(entries):
        raise ValueError("Anonymous ID collision")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({"anonymous_id": r["anonymous_id"], "text": originals[r["id"]]["text"],
                     "label": "", "notes": ""} for r in entries)
    key = {"version": VERSION, "seed": seed, "entries": entries}
    output, mapping = Path(output), Path(mapping)
    rubric = output.parent / "RUBRIC.md"
    if len({p.resolve() for p in (output, mapping, rubric)}) != 3:
        raise ValueError("Blind CSV, rubric, and mapping must be separate files")
    contents = {output: buffer.getvalue(), mapping: json.dumps(key, ensure_ascii=False, indent=2) + "\n", rubric: RUBRIC}
    # Check all targets before writing; never erase manual labels on a re-export.
    for path, text in contents.items():
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            raise ValueError(f"{path} differs; use a new export path to preserve existing labels/key")
    for path, text in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(text.encode("utf-8"))
    return key


def import_annotations(samples, labels, mapping):
    originals = eligible(samples)
    key = json.loads(Path(mapping).read_text(encoding="utf-8"))
    if key.get("version") != VERSION:
        raise ValueError("Unsupported annotation key version")
    by_anon, by_request = {}, {}
    for entry in key["entries"]:
        anon, request_id = entry["anonymous_id"], entry["id"]
        if anon in by_anon or request_id in by_request:
            raise ValueError("Duplicate identity in annotation key")
        if (request_id not in originals or anon != anonymous_id(request_id, key["seed"])
                or {k: v for k, v in entry.items() if k != "anonymous_id"} != identity(originals[request_id])):
            raise ValueError("Annotation key does not match original samples")
        by_anon[anon], by_request[request_id] = entry, entry
    labeled = {}
    with Path(labels).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected CSV columns: {FIELDS}")
        for row in reader:
            anon = row["anonymous_id"]
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed annotation CSV row")
            if anon not in by_anon or anon in labeled:
                raise ValueError("Unknown or duplicate anonymous ID")
            request_id = by_anon[anon]["id"]
            if row["text"] != originals[request_id]["text"]:
                raise ValueError("Original annotation text changed")
            label = row["label"].strip() or None
            if label is not None and label not in LABELS:
                raise ValueError(f"Unknown label: {label}")
            labeled[anon] = {"label": label, "notes": row["notes"]}
    result = []
    for request_id, original in sorted(originals.items()):
        anon = by_request.get(request_id, {}).get("anonymous_id")
        result.append({"version": VERSION, **identity(original), "anonymous_id": anon,
                       **labeled.get(anon, {"label": None, "notes": ""})})
    return result


def annotation_summary(samples, annotations, models):
    originals = eligible(samples)
    labels, seen, out_of_selection = {}, set(), 0
    for row in annotations:
        if row["id"] in seen:
            raise ValueError("Duplicate imported annotation")
        seen.add(row["id"])
        if row["id"] not in originals:
            out_of_selection += 1
            continue
        if (row.get("version") != VERSION or row.get("label") not in (*LABELS, None)
                or any(row.get(k) != v for k, v in identity(originals[row["id"]]).items())):
            raise ValueError("Imported annotation identity or label does not match samples")
        labels[row["id"]] = row["label"]
    groups = []
    for model in models:
        for audience in AUDIENCES:
            group = [r for r in originals.values() if r["model_id"] == model["id"] and r["audience"] == audience]
            counts = Counter(labels[r["id"]] for r in group if labels.get(r["id"]) is not None)
            n = sum(counts.values())
            groups.append({"model_id": model["id"], "audience": audience, "eligible": len(group),
                "annotated": n, "missing": len(group) - n,
                "annotation_coverage": n / len(group) if group else None,
                "counts": {label: counts[label] for label in LABELS},
                "frequencies_among_annotated": {label: counts[label] / n if n else None for label in LABELS}})
    return {"version": VERSION, "groups": groups, "out_of_selection": out_of_selection,
            "denominator": "All annotated valid handoffs in each selected model/audience, including uncertain labels; not restricted to BPC-matched tasks",
            "missing_policy": "Blank or absent annotations stay missing and are excluded from frequencies"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "import"):
        sub = subs.add_parser(command)
        sub.add_argument("--samples", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--mapping", type=Path, required=True)
        if command == "export":
            sub.add_argument("--seed", type=int, default=0)
        else:
            sub.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        samples = read_jsonl(args.samples)
        if args.command == "export":
            key = export_annotations(samples, args.out, args.mapping, args.seed)
            print(f"Exported {len(key['entries'])} handoffs to {args.out}; keep {args.mapping} from the annotator.")
        else:
            labels = import_annotations(samples, args.labels, args.mapping)
            if args.out.resolve() in {p.resolve() for p in (args.samples, args.labels, args.mapping)}:
                raise ValueError("Import output must not overwrite samples, blind labels, or their mapping")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
            print(f"Imported {sum(r['label'] is not None for r in labels)} labels; {sum(r['label'] is None for r in labels)} remain missing.")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
