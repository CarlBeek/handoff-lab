"""Download one pinned public dataset and freeze a small, repository-balanced sample."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
DATASET = "princeton-nlp/SWE-bench_Lite_bm25_13K"
REVISION = "8d4bee9605fd8d72f14373a7f64d070c92f575c8"
PARQUET = "data/test-00000-of-00001.parquet"
VERSION = "swe-handoff-context-v1"
# Explicit column allowlist: never load answers, hints, patches or grading metadata.
COLUMNS = ["instance_id", "repo", "base_commit", "problem_statement", "text"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def packet(row):
    """Preserve issue/code verbatim; discard the upstream patch-generation wrapper."""
    issue = row["problem_statement"]
    code = re.search(r"\n<code>\n(.*)\n</code>\n", row["text"], flags=re.S)
    if not isinstance(issue, str) or not issue.strip() or not code or not code[1].strip():
        raise ValueError("Missing issue or delimited retrieved code")
    if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
        raise ValueError("Missing repository revision")
    result = {
        "id": row["instance_id"],
        "source": {"dataset": DATASET, "revision": REVISION, "split": "test",
                   "instance_id": row["instance_id"], "repo": row["repo"],
                   "base_commit": row["base_commit"]},
        "issue": issue, "code": code[1], "preparation": VERSION,
    }
    result["context_hash"] = digest(result)
    return result


def sample_tasks(rows, pilot=5, main=50, seed=20260909):
    if pilot < 1 or main < 1:
        raise ValueError("Pilot and main sizes must be positive")
    candidates, excluded, seen = [], [], set()
    for row in sorted(rows, key=lambda row: row["instance_id"]):
        if row["instance_id"] in seen:
            raise ValueError(f"Duplicate task: {row['instance_id']}")
        seen.add(row["instance_id"])
        try:
            candidates.append(packet(row))
        except ValueError as exc:
            excluded.append({"id": row["instance_id"], "reason": str(exc)})
    if len(candidates) < pilot + main:
        raise ValueError(f"Need {pilot + main} eligible tasks; found {len(candidates)}")
    groups = defaultdict(list)
    for context in candidates:
        groups[context["source"]["repo"]].append(context)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    for split, count in (("pilot", pilot), ("main", main)):
        remaining = count
        while remaining:
            repos = sorted(repo for repo, group in groups.items() if group)
            rng.shuffle(repos)
            for repo in repos:
                selected.append({**groups[repo].pop(), "split": split})
                remaining -= 1
                if not remaining:
                    break
    manifest = {
        "dataset": DATASET, "revision": REVISION, "parquet": PARQUET,
        "preparation": VERSION, "seed": seed,
        "sampling": "Shuffled repository round-robin without replacement, pilot then main",
        "eligible": len(candidates), "excluded": excluded,
        "candidate_hash": digest(candidates), "contexts_hash": digest(selected),
        "tasks": [{"id": c["id"], "split": c["split"], "repo": c["source"]["repo"],
                   "context_hash": c["context_hash"]} for c in selected],
    }
    return selected, manifest


def prepare(output=ROOT / "data", pilot=5, main=50, seed=20260909):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=COLUMNS).to_pylist()
    contexts, manifest = sample_tasks(rows, pilot, main, seed)
    contents = {
        "contexts.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in contexts),
        "task_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }
    output = Path(output)
    # Check every target before writing anything. A frozen selection is never overwritten.
    for name, content in contents.items():
        target = output / name
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError(f"{target} differs; use another --out directory for a new study")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        target = output / name
        if not target.exists():
            with target.open("x", encoding="utf-8") as fh:
                fh.write(content)
    for split in ("pilot", "main"):
        chosen = [c for c in contexts if c["split"] == split]
        print(f"{split}: {len(chosen)} tasks, repositories {dict(Counter(c['source']['repo'] for c in chosen))}")
    print(f"Frozen contexts: {output / 'contexts.jsonl'}")
    return contexts, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--main", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    if args.pilot < 1 or args.main < 1:
        parser.error("--pilot and --main must be positive")
    prepare(args.out, args.pilot, args.main, args.seed)


if __name__ == "__main__":
    main()
