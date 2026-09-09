#!/usr/bin/env python
"""Drive Claude Code and Codex CLI headlessly on the same tasks so both log real parent -> sub-agent messages.

Each run gets a fresh copy of the synthetic target repo. Both harnesses persist their own transcripts
(~/.claude/projects, ~/.codex/sessions); this script only records a manifest (out/harness_runs.jsonl) with
the run directory, model, task and condition so the parsers can join on `cwd`.

    python scripts/run_harness.py --dry-run
    python scripts/run_harness.py --harness claude --models claude-sonnet-5 --tasks probe
    python scripts/run_harness.py --harness codex --models gpt-5.6-luna --tasks probe
    python scripts/run_harness.py --harness both --models claude-opus-5 gpt-5.6-sol --tasks all --condition default human_reads

Conditions: default (no instruction) or human_reads (OpenAI's own wording written into CLAUDE.md / AGENTS.md).
Claude Code runs with --dangerously-skip-permissions inside the throwaway repo; Codex runs with -s workspace-write.
Both use the harness's default reasoning effort so the data reflects the product as shipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.make_target_repo import make  # noqa: E402

HUMAN_READS = ("Messages that you send to other agents and your final answer may be read by a human, so ensure they "
               "are legible. Always put proper spaces between words and/or numbers.\n")
CLAUDE_MODELS = {"claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-7",
                 "claude-opus-4-8", "claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-fable-5-1"}


def harness_for(model: str) -> str:
    return "claude" if model.startswith("claude") else "codex"


def build_cmd(harness: str, model: str, prompt: str, run_dir: str, max_turns: int) -> list[str]:
    if harness == "claude":
        return ["claude", "-p", prompt, "--model", model, "--output-format", "json",
                "--dangerously-skip-permissions", "--max-turns", str(max_turns)]
    return ["codex", "exec", "-m", model, "-C", run_dir, "-s", "workspace-write", "--skip-git-repo-check", "--json",
            "-c", "model_reasoning_summary=detailed", prompt]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=["claude", "codex", "both"], default="both")
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-5", "gpt-5.6-luna"])
    ap.add_argument("--tasks", nargs="+", default=["probe"])
    ap.add_argument("--condition", nargs="+", default=["default"], choices=["default", "human_reads"])
    ap.add_argument("--workdir", default=os.path.abspath("out/harness_runs"))
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "tasks.json")))}
    task_ids = list(tasks) if args.tasks == ["all"] else args.tasks
    manifest = os.path.join(os.path.dirname(args.workdir.rstrip("/")), "harness_runs.jsonl")
    os.makedirs(args.workdir, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

    for model in args.models:
        harness = harness_for(model)
        if args.harness != "both" and harness != args.harness:
            continue
        for tid in task_ids:
            for cond in args.condition:
                stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
                run_dir = os.path.join(args.workdir, f"{harness}__{re.sub(r'[^a-z0-9.-]', '-', model)}__{tid}__{cond}__{stamp}")
                cmd = build_cmd(harness, model, tasks[tid]["prompt"], run_dir, args.max_turns)
                if args.dry_run:
                    print("DRY", run_dir, "\n   ", " ".join(shlex.quote(c) for c in cmd)[:300])
                    continue
                make(run_dir)
                if cond == "human_reads":
                    for fname in ("CLAUDE.md", "AGENTS.md"):
                        with open(os.path.join(run_dir, fname), "w") as fh:
                            fh.write(HUMAN_READS)
                    subprocess.run(["git", "add", "-A"], cwd=run_dir, check=True)
                    subprocess.run(["git", "-c", "user.email=h@x", "-c", "user.name=h", "commit", "-q", "-m", "instruction"], cwd=run_dir, check=True)
                print(f"RUN {harness} {model} {tid} {cond} -> {run_dir}", flush=True)
                t0 = time.time()
                try:
                    proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=args.timeout)
                    rc, out, err = proc.returncode, proc.stdout, proc.stderr
                except subprocess.TimeoutExpired as e:
                    rc, out, err = -1, (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
                elapsed = round(time.time() - t0, 1)
                with open(os.path.join(run_dir, "harness_stdout.txt"), "w") as fh:
                    fh.write(out)
                with open(os.path.join(run_dir, "harness_stderr.txt"), "w") as fh:
                    fh.write(err)
                session_id = None
                cost = None
                if harness == "claude":
                    try:
                        j = json.loads(out.strip().splitlines()[-1])
                        session_id, cost = j.get("session_id"), j.get("total_cost_usd")
                    except (ValueError, IndexError):
                        pass
                else:
                    m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', out) or re.search(r'"session_id"\s*:\s*"([^"]+)"', out)
                    session_id = m.group(1) if m else None
                rec = dict(harness=harness, model=model, task=tid, condition=cond, run_dir=run_dir, started=stamp,
                           elapsed_s=elapsed, returncode=rc, session_id=session_id, reported_cost_usd=cost,
                           min_subagents=tasks[tid]["min_subagents"])
                with open(manifest, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"    done rc={rc} {elapsed}s session={session_id} cost={cost}", flush=True)


if __name__ == "__main__":
    main()
