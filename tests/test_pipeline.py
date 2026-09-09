import json
import pytest

from cotlegibility.__main__ import main
from cotlegibility.analyze import analyze
from cotlegibility.collect import cells, collect
from cotlegibility.report import group_key, group_label, render
from cotlegibility.schema import Sample, read_jsonl, unique_samples, write_jsonl


def sample(id="one", **kwargs):
    return Sample(**{"id": id, "provider": "test", "model": "m1", "channel": "subagent_prompt",
                     "source": "fixture", "text": "Inspectthecacheupdatelogic.", **kwargs})


def test_event_dedup_not_text_dedup():
    a, b = sample(), sample("two", model="m2")
    assert list(unique_samples([a, a, b])) == [a, b]
    with pytest.raises(ValueError, match="Conflicting"):
        list(unique_samples([a, sample(text="different")]))


def test_analysis_retains_missing_short_and_all_code(tmp_path):
    original = "Inspectthecache `privateIdentifier`\n<script>alert(1)</script>"
    rows, chunks, manifest = analyze([sample(text=original), sample("short", text="OK"),
        sample("code", text="```python\nx=1\n```"), sample("hidden", text="", status="unobservable")], tmp_path, reference=False, passage_chars=50)
    by_id = {r["id"]: r for r in rows}
    assert by_id["one"]["text"] == original
    assert by_id["short"]["analysis_status"] == "ok"
    assert by_id["code"]["analysis_status"] == "no_prose"
    assert by_id["hidden"]["analysis_status"] == "unobservable"
    assert all("bits_per_char" not in r for r in rows)
    render(tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert "<script>alert" not in page and "&lt;script&gt;alert" in page
    assert "Original message" in page and "unobservable" in page
    assert (tmp_path / "summary.csv").exists()


def test_group_labels_distinguish_settings():
    a, b = sample().to_dict(), sample(meta={"parameters": {"reasoning": {"effort": "medium"}}}).to_dict()
    assert group_key(a) != group_key(b)
    assert group_label(group_key(a)) != group_label(group_key(b))


def test_empty_report_replaces_previous_summary(tmp_path):
    analyze([sample()], tmp_path, reference=False)
    render(tmp_path)
    analyze([], tmp_path, reference=False)
    render(tmp_path)
    assert "fixture" not in (tmp_path / "summary.csv").read_text()
    assert "0 messages" in (tmp_path / "report.html").read_text()


def test_collect_budget_resume_and_raw_capture(monkeypatch, tmp_path):
    import cotlegibility.collect as collection
    requests = []

    def request(model, messages, **kwargs):
        requests.append((model, messages, kwargs))
        return {"served_model": model + "-snapshot", "stop_reason": "completed", "raw": {"id": "r1"},
                "tool_calls": [{"name": "spawn_agent", "arguments": json.dumps({"message": "Inspect the cache."})}]}

    monkeypatch.setattr(collection, "request", request)
    contexts = [{"id": "cache", "text": "Review the cache"}]
    models = [{"id": "test1"}, {"id": "test2"}]
    collect(contexts, models, tmp_path, replicates=2, limit=1, dry_run=True)
    assert not requests and not list(tmp_path.iterdir())
    collect(contexts, models, tmp_path, replicates=2, limit=1)
    rows = list(read_jsonl(tmp_path / "messages.jsonl"))
    assert len(rows) == 4 and sum(r["status"] == "missing" for r in rows) == 3
    collect(contexts, models, tmp_path, replicates=2, limit=10)
    collect(contexts, models, tmp_path, replicates=2, limit=10)
    rows = list(read_jsonl(tmp_path / "messages.jsonl"))
    assert len(requests) == 4 and len({r["id"] for r in rows}) == 4
    assert all(r["status"] == "ok" for r in rows)
    raw = next(read_jsonl(rows[0]["meta"]["raw_path"]))
    assert raw["cell"]["messages"] == requests[0][1]
    assert raw["response"]["raw"] == {"id": "r1"}


def test_failed_collection_is_visible_and_cached(monkeypatch, tmp_path):
    import cotlegibility.collect as collection
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(collection, "request", fail)
    for _ in range(2):
        collect([{"id": "a", "text": "A task"}], [{"id": "test"}], tmp_path)
    assert len(calls) == 1
    assert next(read_jsonl(tmp_path / "messages.jsonl"))["status"] == "error"
    with pytest.raises(ValueError, match="Duplicate model"):
        list(cells([{"id": "a", "text": "task"}], [{"id": "x"}, {"id": "x"}], 1))


def test_judge_budget_balances_groups_and_reuses_cache(monkeypatch, tmp_path):
    from cotlegibility.metrics import judge
    requests = []

    def reader(**kwargs):
        requests.append(kwargs)
        result = {**{d: {"grade": "none", "evidence": [], "reason": "Clear"} for d in judge.DIMENSIONS}, "context_missing": False}
        return {"status": "ok", "result": result, "response": {"served_model": "reader-snapshot"}}

    monkeypatch.setattr(judge, "judge", reader)
    messages = [sample("long", text="One message with many ordinary words. " * 20),
                sample("other", model="m2", text="Another message with words. " * 20)]
    _, chunks, manifest = analyze(messages, tmp_path, reference=False, passage_chars=50, judge_model="reader", judge_limit=2)
    assert {p["sample_id"] for p in chunks if p["judge_status"] == "ok"} == {"long", "other"}
    assert manifest["new_judge_calls"] == 2
    _, chunks, manifest = analyze(messages, tmp_path, reference=False, passage_chars=50, judge_model="reader", judge_limit=0)
    assert len(requests) == 2 and manifest["new_judge_calls"] == 0
    assert sum(p["judge_status"] == "ok" for p in chunks) == 2
    assert all("m1" not in q and "m2" not in q for q in requests)


def test_cli_local_only(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [sample().to_dict()])
    normalized = tmp_path / "messages.jsonl"
    main(["import", "--jsonl", str(source), "--out", str(normalized)])
    report = tmp_path / "report"
    main(["analyze", str(normalized), "--out", str(report), "--surface-only"])
    main(["report", str(report)])
    assert (report / "report.html").exists()
