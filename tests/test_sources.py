import json

from cotlegibility.schema import unique_samples, write_jsonl
from cotlegibility.sources import claude_code, codex


def event(kind, payload, timestamp="2026-09-09T00:00:00Z"):
    return {"type": kind, "payload": payload, "timestamp": timestamp}


def test_codex_child_provenance_and_inherited_events(tmp_path):
    copied = event("response_item", {"type": "message", "id": "copy", "role": "assistant", "content": [{"text": "Parent message"}]})
    write_jsonl(tmp_path / "parent.jsonl", [event("session_meta", {"id": "parent"}),
        event("turn_context", {"model": "parent-model"}), copied])
    write_jsonl(tmp_path / "child.jsonl", [event("session_meta", {"id": "child", "source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent"}}}}),
        copied, event("turn_context", {"model": "child-model"}),
        event("response_item", {"type": "message", "id": "child-reply", "role": "assistant", "channel": "final", "content": [{"text": "Child reply"}]})])
    samples = list(codex.iter_samples(tmp_path))
    assert len(samples) == 2
    reply = next(s for s in samples if s.channel == "subagent_reply")
    assert reply.model == "child-model" and reply.meta["parent_thread_id"] == "parent"
    assert sum(s.text == "Parent message" for s in samples) == 1


def test_codex_missing_child_is_not_parent_model(tmp_path):
    records = [event("session_meta", {"id": "parent"}), event("turn_context", {"model": "parent-model"})]
    for call_id in ("wait1", "wait2"):
        records += [event("response_item", {"type": "function_call", "name": "wait_agent", "call_id": call_id, "arguments": "{}"}),
            event("response_item", {"type": "function_call_output", "call_id": call_id, "output": json.dumps({"status": {"missing-child": {"completed": "Finished review"}}})})]
    write_jsonl(tmp_path / "parent.jsonl", records)
    samples = list(unique_samples(codex.iter_samples(tmp_path)))
    assert len(samples) == 1 and samples[0].model == "unknown"
    assert samples[0].meta["parent_model"] == "parent-model"


def test_codex_observation_gaps_and_bad_arguments(tmp_path):
    records = [event("session_meta", {"id": "p"}), event("turn_context", {"model": "m"})]
    for i, message in enumerate(("gAAAAciphertext", ["not", "text"], "")):
        records.append(event("response_item", {"type": "function_call", "name": "collaboration.spawn_agent", "call_id": str(i),
                                              "arguments": json.dumps({"message": message})}))
    records += [event("response_item", {"type": "reasoning", "id": "hidden", "encrypted_content": "cipher", "summary": []}),
                event("response_item", {"type": "reasoning", "id": "summary", "summary": [{"text": "Visible summary"}]}),
                event("inter_agent_communication", {"id": "com", "encrypted_content": "cipher"})]
    write_jsonl(tmp_path / "p.jsonl", records)
    samples = list(codex.iter_samples(tmp_path))
    assert [s.status for s in samples[:3]] == ["unobservable", "error", "missing"]
    assert all(not s.text for s in samples[:4])
    assert samples[3].channel == "reasoning_unavailable"
    assert samples[4].channel == "reasoning_summary"
    assert samples[5].status == "unobservable"


def test_codex_narrow_import_skips_imported_claude_threads(tmp_path):
    root = tmp_path / "sessions" / "day"
    write_jsonl(tmp_path / "external_agent_session_imports.json", [{"records": [{"imported_thread_id": "foreign"}]}])
    for session in ("native", "foreign"):
        write_jsonl(root / f"{session}.jsonl", [event("session_meta", {"id": session}),
            event("response_item", {"type": "function_call", "call_id": "send", "name": "send_message", "arguments": '{"message":"Follow up"}'})])
    samples = list(codex.iter_samples(root))
    assert len(samples) == 1 and samples[0].session_id == "native"
    assert samples[0].channel == "interagent_message"


def test_claude_snapshots_and_child_channels(tmp_path):
    records = []
    for text, stop in (("Partial", None), ("Full reply", "end_turn")):
        records.append({"type": "assistant", "sessionId": "s", "isSidechain": True,
            "message": {"id": "message1", "model": "child-model", "stop_reason": stop, "content": [{"type": "text", "text": text}]}})
    records.append({"type": "assistant", "sessionId": "s", "message": {"id": "message2", "model": "parent-model", "content": [
        {"type": "thinking", "thinking": "A visible thought"}, {"type": "tool_use", "name": "Agent", "input": {"prompt": "Review this code"}}]}})
    write_jsonl(tmp_path / "session.jsonl", records)
    samples = list(claude_code.iter_samples(tmp_path))
    assert len(samples) == 3
    assert samples[0].text == "Full reply" and samples[0].channel == "subagent_reply"
    assert samples[0].model == "child-model"
    assert [s.channel for s in samples[1:]] == ["reasoning_summary", "subagent_prompt"]


def test_claude_redacted_and_missing_blocks_are_visible(tmp_path):
    write_jsonl(tmp_path / "session.jsonl", [{"type": "assistant", "message": {"id": "m", "model": "claude-test", "content": [
        {"type": "redacted_thinking", "data": "ciphertext"}, {"type": "tool_use", "name": "Agent", "input": {}}]}}])
    samples = list(claude_code.iter_samples(tmp_path))
    assert [(s.channel, s.status) for s in samples] == [("reasoning_unavailable", "unobservable"), ("subagent_prompt", "missing")]
