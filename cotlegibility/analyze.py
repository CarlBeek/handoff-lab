"""Original messages -> cached measurements and passage annotations. No generation here."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .metrics import judge as reader
from .metrics.perplexity import MODEL, REVISION, VERSION as REF_VERSION, bits_per_char
from .metrics.surface import NONPROSE, VERSION as SPACING_VERSION, passages, prose_only, spacing
from .schema import digest, read_jsonl, write_jsonl


def environment():
    versions = {}
    for name in ("cotlegibility", "wordfreq", "torch", "transformers", "tokenizers", "openai", "anthropic"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions


def cached(directory, spec, compute):
    path = Path(directory) / f"{digest(spec)}.jsonl"
    if path.exists():
        return next(read_jsonl(path))["result"]
    result = compute()
    write_jsonl(path, [{"spec": spec, "result": result}])
    return result


def prose_segment(text, exclusions, start, end):
    """Remove complete-document code/path matches, including those crossing a passage boundary."""
    parts, cursor = [], start
    for match in exclusions:
        if match.end() <= start or match.start() >= end:
            continue
        parts.extend([text[cursor:max(cursor, match.start())], " "])
        cursor = min(end, match.end())
    parts.append(text[cursor:end])
    return "".join(parts).strip()


def measure(text, reference, cache, spec):
    result = spacing(text)
    result["analysis_chars"] = len(text)
    result["whitespace_share"] = sum(c.isspace() for c in text) / len(text) if text else None
    if reference and text:
        def score(value):
            return cached(cache / "reference", {**spec, "text": value}, lambda: bits_per_char(value, device=spec["device"]))
        original, restored = score(text), score(result["restored"])
        result.update(bits_per_char=original["bits_per_char"], bits_per_token=original["bits_per_token"],
                      reference_tokens=original["tokens"], total_bits=original["total_bits"],
                      restored_bits_per_char=restored["bits_per_char"], restored_total_bits=restored["total_bits"],
                      spacing_difference=(original["total_bits"] - restored["total_bits"]) / len(text))
    return result


def analyze(samples, output, *, reference=True, passage_chars=600, device="auto", judge_model=None,
            judge_parameters=None, judge_limit=30):
    output = Path(output)
    cache = output / "cache"
    versions = environment()
    if reference and device == "auto":
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ref_spec = {"model": MODEL, "revision": REVISION, "version": REF_VERSION, "device": device, "versions": versions}
    messages, chunks, judge_calls = [], [], 0
    judge_queue = defaultdict(list)
    for sample in sorted(samples, key=lambda s: digest(s.id)):
        row, text = sample.to_dict(), sample.text
        exclusions = list(NONPROSE.finditer(text))
        masked = prose_only(text)
        analysis_text = prose_segment(text, exclusions, 0, len(text))
        row.update(analysis_text=analysis_text, chars=len(text),
                   excluded_fraction=sum(m.end() - m.start() for m in exclusions) / len(text) if text else 0,
                   analysis_status=sample.status if sample.status != "ok" else "ok" if analysis_text else "no_prose")
        if row["analysis_status"] == "ok":
            row.update(measure(analysis_text, reference, cache, ref_spec))
            for index, (start, end, original) in enumerate(passages(text, passage_chars)):
                prose = prose_segment(text, exclusions, start, end)
                item = {"sample_id": sample.id, "index": index, "start": start, "end": end,
                        "text": original, "analysis_text": prose, "analysis_status": "ok" if prose else "no_prose"}
                if prose:
                    item.update(measure(prose, reference, cache, ref_spec))
                    # Evidence/highlights use offsets within ORIGINAL passage, not cleaned prose.
                    item["source_spacing_spans"] = spacing(masked[start:end])["spacing_spans"]
                item["judge_status"] = "not_requested"
                if judge_model and prose:
                    kwargs = {"passage": original, "model": judge_model,
                              "context": sample.meta.get("reader_context", ""),
                              "before": text[max(0, start - 300):start], "after": text[end:end + 150],
                              "parameters": judge_parameters}
                    judge_spec = {"version": reader.VERSION, "prompt": reader.PROMPT, "schema": reader.SCHEMA,
                                  "versions": versions, **kwargs}
                    group = digest([sample.source, sample.model, sample.channel, sample.meta.get("parameters")])
                    judge_queue[group].append((item, judge_spec, kwargs))
                chunks.append(item)
        messages.append(row)
        print(f"Analyzed {sample.id[:32]}: {row['analysis_status']}", flush=True)
    # Round-robin source/model/channel/settings groups; hash-order passages within each.
    # This is a balanced convenience sample, not a prevalence-weighted random sample.
    queues = [deque(sorted(items, key=lambda x: digest([x[0]["sample_id"], x[0]["index"]])))
              for _, items in sorted(judge_queue.items())]
    while any(queues):
        for queue in queues:
            if not queue:
                continue
            item, judge_spec, kwargs = queue.popleft()
            path = cache / "judge" / f"{digest(judge_spec)}.jsonl"
            if not path.exists() and judge_calls >= judge_limit:
                item["judge_status"] = "budget_skipped"
                continue
            judge_calls += int(not path.exists())
            try:
                result = cached(cache / "judge", judge_spec, lambda: reader.judge(**kwargs))
            except Exception as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                write_jsonl(path, [{"spec": judge_spec, "result": result}])
            item.update(judge_status=result["status"], judge=result.get("result"),
                        judge_error=result.get("error"), judge_raw_path=str(path),
                        judge_served_model=result.get("response", {}).get("served_model"))
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "dataset_hash": digest([s.to_dict() for s in samples]),
                "messages": len(messages), "passages": len(chunks), "reference": ref_spec if reference else None,
                "spacing_version": SPACING_VERSION, "passage_chars": passage_chars, "judge_model": judge_model,
                "judge_version": reader.VERSION if judge_model else None, "judge_parameters": judge_parameters,
                "judge_limit": judge_limit, "new_judge_calls": judge_calls, "versions": versions}
    write_jsonl(output / "measured_messages.jsonl", messages)
    write_jsonl(output / "passages.jsonl", chunks)
    write_jsonl(output / "manifest.jsonl", [manifest])
    return messages, chunks, manifest
