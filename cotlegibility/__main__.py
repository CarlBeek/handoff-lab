"""Import, collect, analyze, report, and plot. API work is always explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import Sample, read_jsonl, unique_samples, write_jsonl


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="Normalize explicit log directories or existing Sample JSONL")
    imp.add_argument("--codex", type=Path)
    imp.add_argument("--claude", type=Path)
    imp.add_argument("--exemplars", type=Path)
    imp.add_argument("--jsonl", type=Path, action="append", default=[])
    imp.add_argument("--out", type=Path, required=True)
    coll = commands.add_parser("collect", help="Generate matched handoffs using direct provider APIs")
    coll.add_argument("--contexts", type=Path, default=Path("data/contexts.jsonl"))
    coll.add_argument("--models", type=Path, default=Path("models.json"))
    coll.add_argument("--replicates", type=positive, default=1)
    coll.add_argument("--max-calls", type=positive, default=10)
    coll.add_argument("--out", type=Path, default=Path("out/collection"))
    coll.add_argument("--dry-run", action="store_true")
    analysis = commands.add_parser("analyze", help="Score messages and build a portable local report")
    analysis.add_argument("input", type=Path)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--surface-only", action="store_true", help="Skip GPT-2, useful for parser/format checks")
    analysis.add_argument("--passage-chars", type=positive, default=600)
    analysis.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    analysis.add_argument("--judge-model", help="Optional fixed model ID; makes API requests")
    analysis.add_argument("--judge-parameters", type=json.loads, default={})
    analysis.add_argument("--judge-limit", type=int, default=30, help="Maximum NEW judge requests; 0 uses cached results only")
    report = commands.add_parser("report", help="Rebuild HTML/CSV/figure from saved measurements, without model calls")
    report.add_argument("directory", type=Path)
    plot = commands.add_parser("plot", help="Matplotlib timeline of mean BPC with 95%% cluster-bootstrap CIs")
    plot.add_argument("directory", type=Path, help="An analysis directory containing measured_messages.jsonl and manifest.jsonl")
    plot.add_argument("--models", type=Path, default=Path("data/model_releases.json"))
    plot.add_argument("--out", type=Path, help="PNG path; defaults to DIRECTORY/surprisal.png")
    plot.add_argument("--source", default="controlled", help="One source; sources are never pooled")
    plot.add_argument("--channel", default="subagent_prompt")
    plot.add_argument("--unit", choices=("context", "session"), default="context", help="Matched tasks, or observational session clusters")
    plot.add_argument("--resamples", type=positive, default=10000)
    plot.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            if not any((args.codex, args.claude, args.exemplars, args.jsonl)):
                parser.error("import requires at least one explicit source")
            samples = []
            for name in ("codex", "claude", "exemplars"):
                path = getattr(args, name)
                if path and not path.expanduser().is_dir():
                    parser.error(f"Missing {name} directory: {path}")
            if args.codex:
                from .sources.codex import iter_samples
                samples.extend(iter_samples(args.codex))
            if args.claude:
                from .sources.claude_code import iter_samples
                samples.extend(iter_samples(args.claude))
            if args.exemplars:
                samples.extend(Sample(**json.loads(p.read_text())) for p in sorted(args.exemplars.expanduser().glob("*.json")))
            for path in args.jsonl:
                samples.extend(Sample(**row) for row in read_jsonl(path))
            unique = list(unique_samples(samples))
            write_jsonl(args.out, (s.to_dict() for s in unique))
            print(f"Imported {len(unique)} messages; removed {len(samples) - len(unique)} duplicate event records")
        elif args.command == "collect":
            from .collect import collect
            collect(list(read_jsonl(args.contexts)), json.loads(args.models.read_text()), args.out,
                    replicates=args.replicates, limit=args.max_calls, dry_run=args.dry_run)
        elif args.command == "analyze":
            if args.passage_chars < 50 or args.judge_limit < 0 or not isinstance(args.judge_parameters, dict):
                parser.error("passage-chars must be >=50, judge-limit >=0, and judge-parameters a JSON object")
            from .analyze import analyze
            from .report import render
            samples = list(unique_samples(Sample(**row) for row in read_jsonl(args.input)))
            analyze(samples, args.out, reference=not args.surface_only, passage_chars=args.passage_chars, device=args.device,
                    judge_model=args.judge_model, judge_parameters=args.judge_parameters, judge_limit=args.judge_limit)
            render(args.out)
        elif args.command == "plot":
            from .plot import render
            render(args.directory, json.loads(args.models.read_text()), args.out, source=args.source, channel=args.channel,
                   unit=args.unit, resamples=args.resamples, seed=args.seed)
        else:
            from .report import render
            render(args.directory)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
