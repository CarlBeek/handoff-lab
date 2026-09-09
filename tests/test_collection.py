import json
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from scripts import collect, prepare_tasks


def source_row(i=0, repo="org/project"):
    return {
        "instance_id": f"{repo.replace('/', '__')}-{i}", "repo": repo, "base_commit": "a" * 40,
        "problem_statement": f"Investigate failure {i}.",
        "text": "Patch instruction\n<code>\n[start of example.py]\n1 x = 1\n[end of example.py]\n</code>\nGenerate a patch. <patch>FORMAT EXAMPLE</patch>\n",
        "patch": "SECRET SOLUTION", "test_patch": "SECRET TEST", "hints_text": "SECRET HINT",
    }


def contexts():
    return [{**prepare_tasks.packet(source_row(i)), "split": "pilot"} for i in range(3)]


def models():
    return [{"id": "fixture-model", "label": "Fixture",
             "release_date": "2026-01-01", "release_source": "https://example.org/release",
             "parameters": {"reasoning": {"effort": "medium"}, "max_output_tokens": 100}}]


def fake_client(action):
    return SimpleNamespace(responses=SimpleNamespace(create=action))


def response(model="fixture-model", message="Inspect the cache update ordering."):
    raw = {"id": "resp_fixture", "object": "response", "created_at": 0, "status": "completed",
           "model": model, "output": [{"type": "function_call", "id": "fc_fixture",
           "call_id": "call_fixture", "name": "spawn_agent", "arguments": json.dumps({"message": message})}],
           "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}
    return SimpleNamespace(model_dump=lambda **kw: raw, _request_id="req_fixture")


def test_packet_excludes_answers_and_wrapper():
    result = prepare_tasks.packet(source_row())
    serialized = json.dumps(result)
    assert "SECRET" not in serialized and "FORMAT EXAMPLE" not in serialized
    assert "Patch instruction" not in serialized and "Generate a patch" not in serialized
    assert result["issue"] == source_row()["problem_statement"]
    assert result["code"].startswith("[start of example.py]\n1 x = 1")
    assert result["source"]["revision"] == prepare_tasks.REVISION
    assert not {"patch", "test_patch", "hints_text"}.intersection(prepare_tasks.COLUMNS)


def test_sampling_is_disjoint_balanced_and_order_independent():
    rows = [source_row(i, f"org/repo{r}") for r in range(6) for i in range(20)]
    a, manifest = prepare_tasks.sample_tasks(rows)
    b, again = prepare_tasks.sample_tasks(list(reversed(rows)))
    assert a == b and manifest == again
    assert len(a) == len({c["id"] for c in a}) == 55
    for split, expected in (("pilot", 5), ("main", 50)):
        chosen = [c for c in a if c["split"] == split]
        counts = prepare_tasks.Counter(c["source"]["repo"] for c in chosen)
        assert len(chosen) == expected
        assert max(counts.values()) - min(counts.values()) <= 1
    assert a != prepare_tasks.sample_tasks(rows, seed=99)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_tasks.sample_tasks(rows + rows[:1])
    with pytest.raises(ValueError, match="Need"):
        prepare_tasks.sample_tasks(rows[:3])


def test_ineligible_rows_are_reported():
    rows = [source_row(i) for i in range(4)]
    rows[-1]["text"] = "No code"
    tasks, manifest = prepare_tasks.sample_tasks(rows, pilot=1, main=2)
    assert len(tasks) == 3
    assert manifest["excluded"] == [{"id": rows[-1]["instance_id"], "reason": "Missing issue or delimited retrieved code"}]


def test_prepare_is_idempotent_and_will_not_overwrite(monkeypatch, tmp_path):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist([source_row(i) for i in range(5)]), path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: path)
    output = tmp_path / "prepared"
    a = prepare_tasks.prepare(output, pilot=1, main=2)
    assert prepare_tasks.prepare(output, pilot=1, main=2) == a
    original = (output / "contexts.jsonl").read_bytes()
    with pytest.raises(ValueError, match="differs"):
        prepare_tasks.prepare(output, pilot=1, main=2, seed=3)
    assert (output / "contexts.jsonl").read_bytes() == original


def test_plan_has_matched_input_and_restricted_tool():
    catalog = models() + [{**models()[0], "id": "second-fixture"}]
    requests = collect.plan(contexts(), catalog, "pilot")
    assert len(requests) == 6
    assert requests[0]["request"]["input"] == requests[1]["request"]["input"]
    assert requests[0]["request"]["tools"][0]["strict"] is True
    assert requests[0]["request"]["parallel_tool_calls"] is False
    assert requests[0]["request"]["tool_choice"]["name"] == "spawn_agent"
    with pytest.raises(ValueError, match="Duplicate model"):
        collect.plan(contexts(), models() * 2, "pilot")
    edited = contexts()
    edited[0]["issue"] += " Edited"
    with pytest.raises(ValueError, match="hash mismatch"):
        collect.plan(edited, models(), "pilot")
    bad = models()
    bad[0]["parameters"]["tools"] = []
    with pytest.raises(ValueError):
        collect.plan(contexts(), bad, "pilot")


def test_preview_and_budget_resume(monkeypatch, tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), response())[1])
    directory = tmp_path / "run"
    collect.collect(contexts(), models(), directory, client=client)
    assert not directory.exists() and not seen
    with pytest.raises(ValueError, match="explicit positive"):
        collect.collect(contexts(), models(), directory, execute=True, client=client)
    collect.collect(contexts(), models(), directory, execute=True, max_calls=1, client=client)
    assert len(seen) == 1
    collect.collect(contexts(), models(), directory, execute=True, max_calls=9, client=client)
    collect.collect(contexts(), models(), directory, execute=True, max_calls=9, client=client)
    assert len(seen) == 3
    raw = json.loads(next(directory.glob("request-*.json")).read_text())
    assert raw["request"] in seen and raw["request_id"] == "req_fixture"
    changed = models()
    changed[0]["parameters"]["max_output_tokens"] = 200
    with pytest.raises(ValueError, match="configuration changed"):
        collect.collect(contexts(), changed, directory)


def test_interrupt_is_saved_before_call_and_never_retried(tmp_path):
    seen = []
    def interrupt(**body):
        seen.append(body)
        assert json.loads(next(tmp_path.glob("request-*.json")).read_text())["state"] == "started"
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        collect.collect(contexts()[:1], models(), tmp_path, execute=True, max_calls=1,
                        client=fake_client(interrupt))
    collect.collect(contexts()[:1], models(), tmp_path, execute=True, max_calls=1,
                    client=fake_client(interrupt))
    assert len(seen) == 1


def test_real_sdk_wire_and_no_retry_on_error(tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=response().model_dump(), headers={"x-request-id": "req_wire"})
        return httpx.Response(429, json={"error": {"message": "fixture rate limit", "type": "rate_limit"}})
    client = OpenAI(api_key="test-only", base_url=collect.ENDPOINT, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3, client=client)
    assert len(calls) == 2  # No SDK retry; stop before charging a third request.
    records = [json.loads(p.read_text()) for p in tmp_path.glob("request-*.json")]
    assert {r["state"] for r in records} == {"finished", "error"}
    assert next(r for r in records if r["state"] == "finished")["request_id"] == "req_wire"
    assert next(r for r in records if r["state"] == "error")["error"]["status_code"] == 429


def test_conflicting_saved_record_and_concurrent_run_fail(tmp_path):
    import fcntl
    client = fake_client(lambda **body: response())
    with (tmp_path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another collector"):
            collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=client)
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=client)
    path = next(tmp_path.glob("request-*.json"))
    value = json.loads(path.read_text())
    value["context_hash"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Conflicting"):
        collect.collect(contexts(), models(), tmp_path)
