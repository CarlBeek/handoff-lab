import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import collect, prepare_tasks, surplus


def source_row(i=0, repo="org/project"):
    return {
        "instance_id": f"{repo.replace('/', '__')}-{i}", "repo": repo, "base_commit": "a" * 40,
        "problem_statement": f"Investigate failure {i}.",
        "text": "Patch instruction\n<code>\n[start of example.py]\n1 x = 1\n[end of example.py]\n</code>\nGenerate a patch. <patch>FORMAT EXAMPLE</patch>\n",
        "patch": "SECRET SOLUTION", "test_patch": "SECRET TEST", "hints_text": "SECRET HINT",
    }


def contexts(n=3, split="pilot"):
    return [{**prepare_tasks.packet(source_row(i)), "split": split} for i in range(n)]


def make_model(id="fixture-model", api="responses"):
    return {"id": id, "model": id, "label": id, "vendor": "Fixture", "api": api,
            "release_date": "2026-01-01", "release_source": "https://example.org/release",
            "providers": ["openai" if api == "responses" else "anthropic", "openrouter"],
            "response_models": [id],
            "parameters": ({"reasoning": {"effort": "medium"}, "max_output_tokens": 100}
                           if api == "responses" else {"thinking": {"type": "adaptive"},
                           "output_config": {"effort": "medium"}, "max_tokens": 100})}


def models():
    return [make_model()]


def wire_headers(provider="openai"):
    return {"x-request-id": "req_fixture", "x-si-served-by": "marketplace",
            "x-si-provider-family": provider, "x-si-buyer-cost-micro": "1000"}


def offer(provider, price=1_000_000, **extra):
    return {"id": provider + "-offer", "seller_base_url": "https://" + surplus.PROVIDER_HOSTS[provider],
            "available": True, "healthy": True, "trusted": True,
            "effective_input_per_1m": price, "effective_output_per_1m": price * 5, **extra}


def fake_client(action=None, *, offers=None, headers=None, status=200):
    def handler(request):
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"model": request.url.path.rsplit("/", 1)[1],
                "offers": offers if offers is not None else [offer(p) for p in surplus.PROVIDER_HOSTS]})
        assert request.url.host == "api.surplusintelligence.ai"
        assert request.headers["authorization"] == "Bearer test-only"
        body = json.loads(request.content)
        raw = action(**body) if action else response(model=body["model"])
        raw = raw.model_dump() if hasattr(raw, "model_dump") else raw
        retained = wire_headers(body["provider"])
        retained.update(headers or {})
        retained = {k: v for k, v in retained.items() if v is not None}
        return httpx.Response(status, json=raw, headers=retained)
    return httpx.Client(transport=httpx.MockTransport(handler))


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


def test_plan_matched_inputs_and_native_tools():
    catalog = models() + [make_model("claude-fixture", "messages")]
    requests = collect.plan(contexts(), catalog, "pilot")
    a, b = requests[:2]
    assert a["request"]["input"][1]["content"] == b["request"]["messages"][0]["content"]
    assert a["request"]["input"][0]["content"] == b["request"]["system"] == collect.INSTRUCTION
    assert a["request"]["tools"][0]["parameters"] == b["request"]["tools"][0]["input_schema"]
    assert a["request"]["tool_choice"] == "auto"
    assert b["request"]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert a["request"]["provider"] == "openai" and b["request"]["provider"] == "anthropic"
    assert b["request"]["output_config"] == {"effort": "medium"}
    assert all(r["request"]["stream"] is False for r in requests)
    assert all("store" not in r["request"] for r in requests if r["api"] == "messages")
    with pytest.raises(ValueError, match="duplicate"):
        collect.plan(contexts(), models() * 2, "pilot")
    edited = contexts()
    edited[0]["issue"] += " Edited"
    with pytest.raises(ValueError, match="hash mismatch"):
        collect.plan(edited, models(), "pilot")


def test_preview_offline_and_quote_has_no_paid_calls_or_writes(tmp_path):
    def deny(**body):
        raise AssertionError("Paid call from preview")
    report = collect.collect(contexts(), models(), tmp_path / "raw", client=fake_client(deny))
    assert not report[0]["route_resolved"]
    quoted = collect.collect(contexts(), models(), tmp_path / "raw", quote=True, client=fake_client(deny))
    assert quoted[0]["route_resolved"] and quoted[0]["estimated_cost_at_output_cap_usd"] > 0
    assert not (tmp_path / "raw").exists()


def test_partial_resume_and_adding_model_preserve_old_bytes(tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), response(model=body["model"]))[1])
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=client)
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=9, client=client)
    assert len(seen) == 3
    assert all(p.read_bytes() == content for p, content in original.items())
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    catalog = models() + [make_model("sixth-model")]
    collect.collect(contexts(), catalog, tmp_path, execute=True, max_calls=9, client=client)
    assert len(seen) == 6 and all(b["model"] == "sixth-model" for b in seen[3:])
    assert all(p.read_bytes() == content for p, content in original.items())
    collect.collect(contexts(), catalog, tmp_path, execute=True, max_calls=9, client=client)
    assert len(seen) == 6


def test_25_then_50_targets_are_idempotent_and_balanced(tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), response(model=body["model"]))[1])
    catalog = models() + [make_model("second-model")]
    tasks = contexts(50, "main")
    collect.collect(tasks, catalog, tmp_path, split="main", execute=True, max_calls=50, client=client)
    assert len(seen) == 50
    original = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    collect.collect(tasks, catalog, tmp_path, split="main", execute=True, max_calls=50, client=client)
    assert len(seen) == 50
    collect.collect(tasks, catalog, tmp_path, split="main", tasks=50, execute=True, max_calls=50, client=client)
    assert len(seen) == 100
    assert all(p.read_bytes() == content for p, content in original.items())
    assert len({p.name for p in tmp_path.rglob("request-*.json")}) == 100


def test_provider_priority_fallback_then_pin(tmp_path):
    model = models()[0]
    route = surplus.choose_route(fake_client(offers=[offer("openrouter", 1), offer("openai", 100)]), model)
    assert route["provider"] == "openai"  # Not simply cheapest.
    fallback = fake_client(offers=[offer("openai", healthy=False), offer("openrouter")])
    collect.collect(contexts(), [model], tmp_path, execute=True, max_calls=1, client=fallback)
    manifest = json.loads((tmp_path / model["id"] / "run.json").read_text())
    assert manifest["route"]["provider"] == "openrouter"
    seen = []
    client = fake_client(lambda **body: (seen.append(body), response())[1])
    collect.collect(contexts(), [model], tmp_path, execute=True, max_calls=1, client=client)
    assert seen[0]["provider"] == "openrouter"
    with pytest.raises(ValueError, match="No healthy approved offer"):
        collect.collect(contexts(), [model], tmp_path, execute=True, max_calls=1,
                        client=fake_client(offers=[offer("openai")]))


@pytest.mark.parametrize("bad", [
    {"trusted": False}, {"healthy": False}, {"available": False},
    {"seller_base_url": "https://api.openai.com.evil.example/v1"},
    {"seller_base_url": "http://api.openai.com/v1"},
    {"seller_base_url": "https://user@api.openai.com/v1"},
    {"seller_base_url": "https://api.openai.com:444/v1"},
    {"effective_input_per_1m": -1}, {"effective_output_per_1m": "NaN"},
])
def test_unapproved_offers_fail_closed(bad):
    with pytest.raises(ValueError, match="No healthy approved"):
        surplus.choose_route(fake_client(offers=[offer("openai", **bad)]), models()[0])


def test_config_change_and_task_change_refused(tmp_path):
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=fake_client())
    changed = models()
    changed[0]["parameters"]["max_output_tokens"] = 200
    with pytest.raises(ValueError, match="configuration changed"):
        collect.collect(contexts(), changed, tmp_path)
    with pytest.raises(ValueError, match="Frozen study changed"):
        collect.collect(contexts()[::-1], models(), tmp_path)
    with pytest.raises(ValueError, match="Frozen study changed"):
        collect.collect(contexts(4), models(), tmp_path)


def test_interrupt_saved_before_send_and_never_retried(tmp_path):
    def interrupt(**body):
        saved = json.loads(next(tmp_path.rglob("request-*.json")).read_text())
        assert saved["state"] == "started"
        raise KeyboardInterrupt
    client = fake_client(interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect.collect(contexts()[:1], models(), tmp_path, execute=True, max_calls=1, client=client)
    collect.collect(contexts()[:1], models(), tmp_path, execute=True, max_calls=1, client=client)


def test_http_error_no_retry_and_saved_raw_headers(tmp_path):
    calls = []
    client = fake_client(lambda **body: (calls.append(body), {"error": {"message": "rate limited"}})[1],
                        status=429, headers={"set-cookie": "private"})
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3, client=client)
    assert len(calls) == 1
    record = json.loads(next(tmp_path.rglob("request-*.json")).read_text())
    assert record["state"] == "error" and record["http_status"] == 429
    assert record["response"]["error"]["message"] == "rate limited"
    assert record["cost_micro"] == 1000
    assert "set-cookie" not in record["response_headers"]
    assert "test-only" not in json.dumps(record)


@pytest.mark.parametrize("headers,problem", [
    ({"x-si-provider-family": "venice"}, "missing_or_unexpected_provider"),
    ({"x-si-provider-family": None}, "missing_or_unexpected_provider"),
    ({"x-si-served-by": "fallback_key"}, "missing_or_unexpected_serving_rail"),
    ({"x-si-truncated": "1"}, "gateway_truncated"),
    ({"x-si-adapted-params": "reasoning"}, "adapted_generation_parameters"),
    ({"x-si-buyer-cost-micro": None}, "missing_or_invalid_cost"),
])
def test_provenance_violation_saves_and_stops(tmp_path, headers, problem):
    calls = []
    client = fake_client(lambda **body: (calls.append(body), response())[1], headers=headers)
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3, client=client)
    record = json.loads(next(tmp_path.rglob("request-*.json")).read_text())
    assert len(calls) == 1 and problem in record["validation_errors"]


def test_snapshot_change_stops_and_cannot_be_pooled(tmp_path):
    seen = []
    def reply(**body):
        seen.append(body)
        return response(model=f"fixture-model-2026-09-0{len(seen)}")
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3, client=fake_client(reply))
    assert len(seen) == 2
    records = [json.loads(p.read_text()) for p in tmp_path.rglob("request-*.json")]
    assert any("served_snapshot_changed" in r["validation_errors"] for r in records)
    with pytest.raises(ValueError, match="Mixed served snapshots"):
        collect.collect(contexts(), models(), tmp_path)


def test_spend_stop_and_unknown_cost(tmp_path):
    seen = []
    client = fake_client(lambda **body: (seen.append(body), response())[1],
                         headers={"x-si-buyer-cost-micro": "100000"})
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3,
                    stop_after_usd=.15, client=client)
    assert len(seen) == 2  # Clearly documented one-request overshoot, not a hard budget.


def test_concurrent_run_and_conflicting_record_fail(tmp_path):
    import fcntl
    with (tmp_path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another collector"):
            collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=fake_client())
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=fake_client())
    path = next(tmp_path.rglob("request-*.json"))
    saved = json.loads(path.read_text())
    saved["context_hash"] = "changed"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Conflicting"):
        collect.collect(contexts(), models(), tmp_path)


def test_execute_guards_and_cli_selection(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="explicit positive"):
        collect.collect(contexts(), models(), tmp_path, execute=True)
    monkeypatch.delenv("SURPLUS_API_KEY")
    with pytest.raises(ValueError, match="SURPLUS_API_KEY"):
        collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="--tasks"):
            collect.collect(contexts(), models(), tmp_path, tasks=bad)
    captured = []
    monkeypatch.setattr(collect, "collect", lambda c, m, *a, **kw: captured.extend(m))
    collect.main(["--model", "claude-fable-5.1"])
    assert [m["id"] for m in captured] == ["claude-fable-5.1"]
    with pytest.raises(SystemExit):
        collect.main(["--model", "typo"])


def test_catalog_has_requested_panel():
    catalog = collect.read_json(collect.ROOT / "data/models.json")
    collect.validate_models(catalog)
    assert len(catalog) == 8
    ids = {m["id"] for m in catalog}
    assert {"claude-fable-5", "claude-fable-5.1"} <= ids
    assert not any("4.5" in m for m in ids)
    assert not any("sonnet" in m.lower() for m in ids)


def test_native_messages_wire_and_response(tmp_path):
    model = make_model("claude-fixture", "messages")
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"model": model["model"], "offers": [offer("anthropic")]})
        assert str(request.url) == surplus.ENDPOINTS["messages"]
        assert request.headers["authorization"] == "Bearer test-only"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["provider"] == "anthropic" and body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium"}
        return httpx.Response(200, json={"model": model["model"], "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "spawn_agent", "input": {"message": "Inspect."}}]},
            headers=wire_headers("anthropic"))
    client = httpx.Client(transport=httpx.MockTransport(handler))
    collect.collect(contexts(), [model], tmp_path, execute=True, max_calls=1, client=client)
    record = collect.read_json(next(tmp_path.rglob("request-*.json")))
    assert record["state"] == "finished" and record["validation_errors"] == []
    assert record["response"]["content"][0]["input"] == {"message": "Inspect."}


def test_transport_error_is_saved_redacted_and_not_retried(tmp_path):
    calls = []
    def fail(**body):
        calls.append(body)
        raise httpx.ReadTimeout("Uncertain call with key test-only")
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3, client=fake_client(fail))
    record = collect.read_json(next(tmp_path.rglob("request-*.json")))
    assert len(calls) == 1 and record["state"] == "error"
    assert record["error"]["type"] == "ReadTimeout" and "test-only" not in json.dumps(record)


def test_non_json_failure_preserved_without_retry(tmp_path):
    calls = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"model": models()[0]["model"], "offers": [offer("openai")]})
        calls.append(request)
        return httpx.Response(502, text="upstream unavailable")
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=3,
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    record = collect.read_json(next(tmp_path.rglob("request-*.json")))
    assert len(calls) == 1 and record["response"] is None
    assert record["response_text"] == "upstream unavailable" and record["cost_micro"] is None


def test_reported_effort_changes_and_model_alias_validation():
    spec = collect.plan(contexts(), models(), "pilot")[0]
    raw = response().model_dump()
    raw["reasoning"] = {"effort": "low"}
    assert "reasoning_effort_mismatch" in surplus.audit(spec, raw, wire_headers())
    allowed = ["claude-fable-5.1", "claude-fable-5-1"]
    assert surplus.model_matches("claude-fable-5-1-20260901", allowed)
    assert surplus.model_matches("claude-fable-5.1-2026-09-01", allowed)
    assert not surplus.model_matches("claude-fable-5.1-2026-99-01", allowed)
    assert not surplus.model_matches("claude-fable-5.1-distilled", allowed)


def test_catalog_rejects_case_collisions_and_missing_labels():
    with pytest.raises(ValueError, match="duplicate"):
        collect.validate_models([make_model("model-A"), make_model("model-a")])
    model = make_model()
    model.pop("vendor")
    with pytest.raises(ValueError, match="label and vendor"):
        collect.validate_models([model])
