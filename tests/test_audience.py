from collections import Counter
import csv
import json

import numpy as np
import pytest

from scripts import analyze_audience as analysis, annotate, collect, handoffs, surplus
from test_collection import contexts, fake_client, make_model, response


def panel():
    return [make_model("gpt-6-astra"), make_model("gpt-5.6-sol"), make_model("claude-fable-5.1", "messages")]


def reply(**body):
    instruction = body.get("system") or body["input"][0]["content"]
    message = "Inspectthecache.\nReturnevidence." if "experienced AI software" in instruction else "Inspect the cache.\nReturn evidence."
    if "messages" in body:
        return {"model": body["model"], "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "name": "send_handoff", "input": {"message": message}}]}
    raw = response(model=body["model"], message=message).model_dump()
    raw["output"][0]["name"] = "send_handoff"
    return raw


def run(output, tasks=None, models=None, **kwargs):
    return collect.collect(tasks or contexts(), models or panel(), output,
                           audience_comparison=True, **kwargs)


def fake_score(text, **kwargs):
    bpc = 2.0 if "Inspectthecache" in text else 1.0
    return {"bits_per_char": bpc, "total_bits": bpc * len(text), "tokens": len(text.split()),
            "characters": len(text), "device": "fixture"}


@pytest.mark.parametrize("api", ["responses", "messages"])
def test_prompt_diff_is_only_audience_word(api):
    requests = collect.plan(contexts(1), [make_model(api=api)], "pilot", audience_comparison=True)
    by_audience = {r["audience"]: r for r in requests}
    ai, human = (by_audience[a] for a in collect.AUDIENCES)
    left = json.dumps(ai["request"]).replace("experienced AI software", "experienced human software")
    assert left == json.dumps(human["request"])
    assert ai["id"] != human["id"]
    assert ai["parameters"] == human["parameters"] and ai["context_hash"] == human["context_hash"]
    for spec in requests:
        assert spec["id"] == collect.digest({k: v for k, v in spec.items() if k != "id"})
        body = spec["request"]
        assert body["tools"][0]["name"] == "send_handoff"
        assert body["tools"][0]["description"] == collect.AUDIENCE_DESCRIPTION
        assert not {"previous_response_id", "conversation", "session_id"} & body.keys()
        if api == "responses":
            assert len(body["input"]) == 2 and body["tool_choice"] == "auto"
            assert body["store"] is False and body["parallel_tool_calls"] is False
        else:
            assert len(body["messages"]) == 1
            assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_schedule_balances_tasks_models_and_is_reproducible():
    requests = collect.plan(contexts(5), panel(), "pilot", audience_comparison=True)
    assert len(requests) == 30
    assert requests == collect.plan(contexts(5), panel()[::-1], "pilot", audience_comparison=True)
    assert [r["schedule"]["task_index"] for r in requests] == sorted(r["schedule"]["task_index"] for r in requests)
    first = [r for r in requests if r["schedule"]["audience_position"] == 0]
    assert sorted(Counter(r["audience"] for r in first).values()) == [7, 8]
    for task in contexts(5):
        assert sorted(Counter(r["audience"] for r in first if r["task_id"] == task["id"]).values()) == [1, 2]
    for model in panel():
        assert sorted(Counter(r["audience"] for r in first if r["model_id"] == model["id"]).values()) == [2, 3]


def test_offline_preview_complete_design_and_no_writes(tmp_path, capsys):
    report = run(tmp_path / "raw", tasks=contexts(5))
    assert sum(r["pending"] for r in report) == 30
    assert all(r["pending_by_audience"] == {"AI": 5, "human": 5} for r in report)
    assert not (tmp_path / "raw").exists()
    text = capsys.readouterr().out
    assert '"design": "5 tasks x 3 models x 2 audiences"' in text
    assert '"pending_requests": 30' in text
    assert '"tool_choice": "auto"' in text


def test_cli_keeps_catalog_subset_and_uses_separate_default(monkeypatch):
    captured = []
    monkeypatch.setattr(collect, "collect", lambda c, m, out, **kw: captured.append((m, out, kw)))
    collect.main(["--audience-comparison", "--model", "gpt-6-astra", "--model", "gpt-5.6-sol",
                  "--model", "claude-fable-5.1"])
    models, output, options = captured[-1]
    assert {m["id"] for m in models} == {m["id"] for m in panel()}
    assert output == collect.ROOT / "data/raw/audience/pilot" and options["audience_comparison"]
    collect.main([])
    assert len(captured[-1][0]) == 8
    assert captured[-1][1] == collect.ROOT / "data/raw/pilot"


def test_partial_resume_schedule_and_new_models_preserve_old_bytes(tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), reply(**body))[1])
    run(tmp_path, tasks=contexts(5), execute=True, max_calls=1, client=client)
    assert len(list(tmp_path.glob("*/run.json"))) == 3  # Includes unattempted models.
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    manifest, rows = handoffs.load_run(tmp_path)
    assert len(rows) == 30 and sum(r["status"] == "not_attempted" for r in rows) == 29
    run(tmp_path, tasks=contexts(5), models=panel()[::-1], execute=True, max_calls=29, client=client)
    assert len(seen) == 30 and all(p.read_bytes() == text for p, text in original.items())
    expected = collect.plan(contexts(5), panel(), "pilot", audience_comparison=True)
    assert seen == [r["request"] for r in expected]
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    run(tmp_path, tasks=contexts(5), models=[make_model("aaa-new-model")], execute=True, max_calls=10, client=client)
    assert len(seen) == 40 and all(p.read_bytes() == text for p, text in original.items())
    assert collect.read_json(tmp_path / "aaa-new-model/run.json")["model_slot"] == 3
    run(tmp_path, tasks=contexts(5), execute=True, max_calls=30, client=client)
    assert len(seen) == 40


def test_task_prefix_extension_keeps_both_conditions(tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), reply(**body))[1])
    tasks = contexts(4, "main")
    models = panel()[:1]
    collect.collect(tasks, models, tmp_path, audience_comparison=True, split="main", tasks=2,
                    execute=True, max_calls=10, client=client)
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    collect.collect(tasks, models, tmp_path, audience_comparison=True, split="main", tasks=4,
                    execute=True, max_calls=10, client=client)
    assert len(seen) == 8 and all(p.read_bytes() == text for p, text in original.items())


def test_interrupted_manifest_preparation_keeps_initial_schedule(tmp_path, monkeypatch):
    write = collect.write_json
    def interrupt(path, value):
        write(path, value)
        if path.name == "run.json":
            raise KeyboardInterrupt
    monkeypatch.setattr(collect, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, execute=True, max_calls=18, client=fake_client(reply))
    assert not list(tmp_path.rglob("request-*.json"))
    assert collect.read_json(next(tmp_path.glob("*/run.json")))["model_slot"] == 0
    monkeypatch.setattr(collect, "write_json", write)
    run(tmp_path, models=panel()[::-1], execute=True, max_calls=18, client=fake_client(reply))
    expected = collect.plan(contexts(), panel(), "pilot", audience_comparison=True)
    actual = handoffs.load_run(tmp_path)[0]["requests"]
    assert sorted(actual, key=collect.schedule_key) == expected


def test_old_and_new_protocols_are_separate_and_old_remains_readable(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    collect.collect(contexts(), [make_model()], old, execute=True, max_calls=1, client=fake_client())
    assert handoffs.load_run(old)[0]["protocol"] == collect.PROTOCOL
    with pytest.raises(ValueError, match="Frozen study changed"):
        run(old)
    run(new, execute=True, max_calls=1, client=fake_client(reply))
    with pytest.raises(ValueError, match="Frozen study changed"):
        collect.collect(contexts(), panel(), new)
    with pytest.raises(ValueError, match="Wrong protocol"):
        analysis.analyze(old, tmp_path / "analysis")
    with pytest.raises(ValueError, match="Wrong protocol"):
        handoffs.load_run(new, expected_protocols={collect.PROTOCOL, "swe-handoff-v1"})


@pytest.mark.parametrize("field,value", [("audience", "human"), ("schedule", {})])
def test_condition_and_schedule_tampering_rejected_on_resume_and_read(tmp_path, field, value):
    run(tmp_path, models=panel()[:1], execute=True, max_calls=1, client=fake_client(reply))
    path = tmp_path / "gpt-6-astra/run.json"
    manifest = collect.read_json(path)
    manifest["requests"][0][field] = value
    collect.write_json(path, manifest)
    with pytest.raises(ValueError, match="Saved plan changed"):
        run(tmp_path, models=panel()[:1])
    with pytest.raises(ValueError, match="condition or schedule"):
        handoffs.load_run(tmp_path)


@pytest.mark.parametrize("api", ["responses", "messages"])
@pytest.mark.parametrize("bad", ["missing", "extra", "wrong_tool", "blank", "malformed", "incomplete"])
def test_invalid_responses_stop_and_remain_missing(tmp_path, api, bad):
    seen = []
    def broken(**body):
        seen.append(body)
        raw = reply(**body)
        key = "output" if api == "responses" else "content"
        if bad == "missing":
            raw[key] = []
        elif bad == "extra":
            raw[key] *= 2
        elif bad == "wrong_tool":
            raw[key][0]["name"] = "spawn_agent"
        elif bad in {"blank", "malformed"}:
            args = {"message": " "} if bad == "blank" else {"other": "not a message"}
            raw[key][0]["arguments" if api == "responses" else "input"] = json.dumps(args) if api == "responses" else args
        else:
            raw["status" if api == "responses" else "stop_reason"] = "incomplete" if api == "responses" else "max_tokens"
        return raw
    run(tmp_path, tasks=contexts(1), models=[make_model(api=api)], execute=True, max_calls=2, client=fake_client(broken))
    assert len(seen) == 1
    manifest, rows = handoffs.load_run(tmp_path)
    expected = "incomplete_response" if bad == "incomplete" else "invalid_handoff"
    assert [r["status"] for r in rows] == [expected, "not_attempted"]
    assert all(r["bits_per_char"] is None for r in rows)
    run(tmp_path, tasks=contexts(1), models=[make_model(api=api)], execute=True, max_calls=2, client=fake_client(reply))
    assert [r["status"] for r in handoffs.load_run(tmp_path)[1]] == [expected, "ok"]


def test_interruption_does_not_retry_or_change_the_other_condition(tmp_path):
    def interrupt(**body):
        record = collect.read_json(next(tmp_path.rglob("request-*.json")))
        assert record["state"] == "started" and record["audience"] == "AI"
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, models=panel()[:1], tasks=contexts(1), execute=True, max_calls=2, client=fake_client(interrupt))
    original = next(tmp_path.rglob("request-*.json")).read_bytes()
    run(tmp_path, models=panel()[:1], tasks=contexts(1), execute=True, max_calls=2, client=fake_client(reply))
    rows = handoffs.load_run(tmp_path)[1]
    assert [r["status"] for r in rows] == ["started", "ok"]
    assert any(p.read_bytes() == original for p in tmp_path.rglob("request-*.json"))


def measured():
    effects = [[2., 0., -1.], [4., 3., 4.], [8., 6., 6.]]
    rows = []
    for i, base in enumerate([100., 2., 50.]):
        for j, model in enumerate(panel()):
            for audience in collect.AUDIENCES:
                bpc = base + effects[i][j] if audience == "AI" else base
                rows.append({"id": f"{i}-{j}-{audience}", "task_id": str(i), "model_id": model["id"],
                    "audience": audience, "context_hash": str(i), "protocol": collect.AUDIENCE_PROTOCOL,
                    "split": "pilot", "parameters": "fixed", "served_model": model["id"],
                    "provider": "fixture", "api": model["api"], "status": "ok", "bits_per_char": bpc,
                    "text": 'Review `cacheKey` and src/cache.py.\nKeep spaces, commas, and Unicode: café.',
                    "prose": "Review and keep spaces."})
    return rows, np.array(effects)


def test_audience_effects_and_contrasts_use_same_whole_task_draws():
    rows, effects = measured()
    summary = analysis.summarize(rows, panel(), resamples=777, seed=29)
    assert summary == analysis.summarize(rows[::-1], panel(), resamples=777, seed=29)
    assert summary["task_ids"] == ["0", "1", "2"]
    indices = np.random.default_rng(29).integers(0, 3, size=(777, 3))
    draws = effects[indices].mean(axis=1)
    for j, point in enumerate(summary["points"]):
        assert point["mean_audience_effect"] == pytest.approx(effects[:, j].mean())
        assert [point["ci_low"], point["ci_high"]] == pytest.approx(np.quantile(draws[:, j], [.025, .975]))
    for j, contrast in enumerate(summary["astra_contrasts"], 1):
        assert contrast["mean_difference"] == pytest.approx((effects[:, 0] - effects[:, j]).mean())
        assert [contrast["ci_low"], contrast["ci_high"]] == pytest.approx(np.quantile(draws[:, 0] - draws[:, j], [.025, .975]))


def test_constant_paired_contrast_has_zero_width_interval():
    rows, _ = measured()
    for row in rows:
        if row["audience"] == "AI":
            row["bits_per_char"] = next(r["bits_per_char"] for r in rows
                if r["task_id"] == row["task_id"] and r["model_id"] == row["model_id"] and r["audience"] == "human")
            row["bits_per_char"] += int(row["task_id"]) * 10 + (2 if row["model_id"] == analysis.ASTRA else 1)
    for contrast in analysis.summarize(rows, panel())["astra_contrasts"]:
        assert contrast["mean_difference"] == pytest.approx(1)
        assert contrast["ci_low"] == pytest.approx(1) and contrast["ci_high"] == pytest.approx(1)


def test_complete_panel_exclusions_zero_one_task_and_missing_model():
    rows, _ = measured()
    rows[0].update(status="incomplete_response", bits_per_char=None)
    summary = analysis.summarize(rows, panel())
    assert summary["task_ids"] == ["1", "2"]
    assert summary["excluded_tasks"] == [{"task_id": "0", "missing_cells": [
        {"model_id": analysis.ASTRA, "audience": "AI", "status": "incomplete_response"}]}]
    assert all(p["n_tasks"] == 2 for p in summary["points"] + summary["astra_contrasts"])
    one = analysis.summarize(rows[6:12], panel())
    assert all(p["ci_low"] is None for p in one["points"] + one["astra_contrasts"])
    assert analysis.summarize(rows[:-1], panel())["task_ids"] == ["1"]
    assert analysis.summarize(rows, panel() + [make_model("missing-model")])["points"] == []
    assert analysis.summarize([], panel())["points"] == []


@pytest.mark.parametrize("field,value", [("protocol", collect.PROTOCOL), ("split", "main"),
    ("audience", None), ("parameters", "changed"), ("served_model", "changed"),
    ("context_hash", "changed"), ("provider", "changed"), ("bits_per_char", float("nan")),
    ("bits_per_char", float("inf")), ("bits_per_char", -1)])
def test_mixed_or_invalid_analysis_samples_fail(field, value):
    rows, _ = measured()
    rows[0][field] = value
    with pytest.raises(ValueError):
        analysis.summarize(rows, panel())


def test_scoring_totals_no_prose_and_failures():
    rows = [{"status": "ok", "text": "Inspectthecache `identifier`.", "bits_per_char": None},
            {"status": "ok", "text": "```python\nx = 1\n```", "bits_per_char": None},
            {"status": "error", "text": None, "bits_per_char": None}]
    scored = analysis.score_rows(rows, scorer=fake_score)
    assert scored[0]["bits_per_char"] == scored[0]["total_bits"] / scored[0]["characters"]
    assert scored[0]["tokens"] > 0 and scored[0]["text"] == rows[0]["text"]
    assert scored[1]["status"] == "no_prose" and scored[1]["bits_per_char"] is None
    assert scored[1]["tokens"] == scored[1]["characters"] == 0
    assert scored[2] == rows[2] and rows[0]["bits_per_char"] is None


def rewrite_csv(path, transform):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = transform(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=annotate.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_blinded_roundtrip_preserves_identities_and_missing_labels(tmp_path):
    rows, _ = measured()
    rows[0]["text"] += "\r\nOriginal CRLF too."
    blind, mapping = tmp_path / "blind/blind.csv", tmp_path / "key.json"
    key = annotate.export_annotations(rows, blind, mapping, seed=71)
    original_bytes = blind.read_bytes()
    assert key == annotate.export_annotations(rows[::-1], blind, mapping, seed=71)
    assert blind.read_bytes() == original_bytes
    assert set(next(csv.reader(blind.open()))) == set(annotate.FIELDS)
    assert all(model["id"] not in blind.read_text() for model in panel())
    assert all(r["id"] not in blind.read_text() for r in rows)
    def label_partially(exported):
        exported[0]["label"] = "fused_prose"
        exported[1]["label"] = "uncertain"
        return exported[:-1]  # Omitted rows and blank labels both remain missing.
    rewrite_csv(blind, label_partially)
    imported = annotate.import_annotations(rows, blind, mapping)
    assert len(imported) == len(rows)
    assert Counter(r["label"] for r in imported) == {"fused_prose": 1, "uncertain": 1, None: 16}
    assert next(r for r in imported if r["anonymous_id"] == key["entries"][0]["anonymous_id"])["id"] == key["entries"][0]["id"]
    summary = annotate.annotation_summary(rows, imported, panel())
    assert sum(g["annotated"] for g in summary["groups"]) == 2
    assert sum(g["missing"] for g in summary["groups"]) == 16
    assert sum(g["counts"]["ordinary_prose"] for g in summary["groups"]) == 0
    assert all(all(f is None for f in g["frequencies_among_annotated"].values())
               for g in summary["groups"] if g["annotated"] == 0)
    with pytest.raises(ValueError, match="preserve existing"):
        annotate.export_annotations(rows, blind, mapping, seed=71)


@pytest.mark.parametrize("bad", ["text", "unknown_id", "duplicate", "label", "key"])
def test_annotation_mismatches_fail(tmp_path, bad):
    rows, _ = measured()
    blind, mapping = tmp_path / "blind.csv", tmp_path / "key.json"
    annotate.export_annotations(rows, blind, mapping)
    def mutate(exported):
        if bad == "text":
            exported[0]["text"] += "edited"
        elif bad == "unknown_id":
            exported[0]["anonymous_id"] = "unknown"
        elif bad == "duplicate":
            exported.append(exported[0])
        elif bad == "label":
            exported[0]["label"] = "negative"
        return exported
    rewrite_csv(blind, mutate)
    if bad == "key":
        key = collect.read_json(mapping)
        key["entries"][0]["audience"] = "wrong"
        collect.write_json(mapping, key)
    with pytest.raises(ValueError):
        annotate.import_annotations(rows, blind, mapping)


def test_analysis_cli_and_annotation_cli_are_offline_and_preserve_raw(tmp_path, monkeypatch):
    raw, output = tmp_path / "raw", tmp_path / "analysis"
    run(raw, execute=True, max_calls=18, client=fake_client(reply))
    original = {p: p.read_bytes() for p in raw.rglob("*.json")}
    def deny(*args, **kwargs):
        raise AssertionError("Analysis attempted collection")
    monkeypatch.setattr(surplus, "send", deny)
    monkeypatch.setattr(surplus, "choose_route", deny)
    monkeypatch.setattr(analysis.surprisal, "bits_per_char", fake_score)
    analysis.main(["--run", str(raw), "--out", str(output)])
    summary = collect.read_json(output / "summary.json")
    assert summary["n_matched_tasks"] == 3 and len(summary["astra_contrasts"]) == 2
    assert summary["known_cost_usd"] == pytest.approx(.018)
    assert (output / "audience_effect.png").stat().st_size > 1000
    samples = annotate.read_jsonl(output / "samples.jsonl")
    assert all(r["total_bits"] > 0 and r["tokens"] > 0 and r["characters"] == len(r["prose"]) for r in samples)
    pairs = annotate.read_jsonl(output / "pairs.jsonl")
    assert len(pairs) == 9 and "Inspectthecache." in (output / "paired_messages.md").read_text()
    blind, mapping, imported = tmp_path / "blind/blind.csv", tmp_path / "key.json", output / "labels.jsonl"
    common = ["--samples", str(output / "samples.jsonl"), "--mapping", str(mapping)]
    annotate.main(["export", *common, "--out", str(blind)])
    annotate.main(["import", *common, "--labels", str(blind), "--out", str(imported)])
    analysis.main(["--run", str(raw), "--out", str(output), "--annotations", str(imported)])
    assert sum(g["missing"] for g in collect.read_json(output / "summary.json")["annotations"]["groups"]) == 18
    assert all(p.read_bytes() == text for p, text in original.items())
    print(f"Synthetic audience figure: {output / 'audience_effect.png'}")


def test_empty_analysis_does_not_load_reference_or_invent_results(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Empty analysis tried to load or score a model")
    monkeypatch.setattr(analysis.surprisal, "load", deny)
    monkeypatch.setattr(analysis.surprisal, "bits_per_char", deny)
    summary = analysis.analyze(tmp_path / "pilot", tmp_path / "out")
    assert summary["points"] == [] and summary["astra_contrasts"] == []
    assert summary["n_matched_tasks"] == 0
    assert (tmp_path / "out/samples.jsonl").read_text() == ""
