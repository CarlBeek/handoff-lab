"""Import explicit Codex rollout events without treating child text as human-facing."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..schema import Sample, digest, read_jsonl


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def _parent(meta):
    source = meta.get("source")
    spawn = ((source.get("subagent") or {}).get("thread_spawn") or {}) if isinstance(source, dict) else {}
    return meta.get("parent_thread_id") or spawn.get("parent_thread_id")


def _event(record):
    p = record.get("payload") or {}
    return p.get("id") or p.get("call_id") or digest([record.get("timestamp"), p])


def iter_samples(root="~/.codex/sessions"):
    root = Path(root).expanduser()
    files, metadata = {}, {}
    imports = next((p / "external_agent_session_imports.json" for p in (root, *root.parents)
                    if (p / "external_agent_session_imports.json").exists()), None)
    imported = set()
    if imports:
        imported = {r.get("imported_thread_id") for r in json.loads(imports.read_text()).get("records", [])}
    for path in sorted(root.rglob("*.jsonl")):
        meta = next((r.get("payload", {}) for r in read_jsonl(path) if r.get("type") == "session_meta"), {})
        session = meta.get("id") or str(path)
        if session not in imported:
            files[session], metadata[session] = path, meta

    @lru_cache(maxsize=None)
    def events(session):
        return {_event(r) for r in read_jsonl(files[session]) if r.get("type") == "response_item"}

    for session, path in files.items():
        meta, model, calls = metadata[session], "unknown", {}
        parent, ancestors, inherited = _parent(meta), set(), set()
        ancestor = parent
        while ancestor in files and ancestor not in ancestors:
            ancestors.add(ancestor)
            inherited.update(events(ancestor))
            ancestor = _parent(metadata[ancestor])
        for i, record in enumerate(read_jsonl(path)):
            kind, p = record.get("type"), record.get("payload") or {}
            if kind == "turn_context":
                model = p.get("model") or model
            if kind not in {"response_item", "inter_agent_communication"}:
                continue
            if kind == "response_item" and _event(record) in inherited:
                continue
            base = dict(id=f"codex:{session}:{_event(record)}", provider="openai", model=model,
                        source="codex_rollout", session_id=session, timestamp=record.get("timestamp"),
                        meta={"path": str(path), "line": i + 1, "cwd": meta.get("cwd"),
                              "cli_version": meta.get("cli_version"), "parent_thread_id": parent})
            if kind == "inter_agent_communication":
                yield Sample(channel="interagent_message", text=p.get("content") or "",
                             status="unobservable" if p.get("encrypted_content") else "ok", **base)
            elif p.get("type") == "function_call":
                name = (p.get("name") or "").rsplit(".", 1)[-1]
                calls[p.get("call_id")] = name
                if name not in {"spawn_agent", "send_input", "send_message", "message_agent", "followup_task"}:
                    continue
                channel = "subagent_prompt" if name == "spawn_agent" else "interagent_message"
                try:
                    args = json.loads(p.get("arguments") or "{}")
                    message = args.get("message") or args.get("input") or args.get("prompt") or ""
                    if not isinstance(message, str):
                        raise ValueError("Agent message must be text")
                except (ValueError, AttributeError):
                    yield Sample(channel=channel, text="", status="error", **base)
                    continue
                encrypted = message.startswith("gAAAA")
                base["meta"].update(tool=name, fork_context=args.get("fork_context"))
                yield Sample(channel=channel, text="" if encrypted else message,
                             status="unobservable" if encrypted else "ok" if message else "missing", **base)
            elif p.get("type") == "function_call_output" and calls.get(p.get("call_id")) == "wait_agent":
                try:
                    statuses = json.loads(p.get("output") or "{}").get("status") or {}
                except (ValueError, AttributeError):
                    continue
                for child, status in statuses.items():
                    message = status.get("completed") if isinstance(status, dict) else None
                    if message and child not in files:
                        # Never attribute an unavailable child's answer to the parent's model.
                        reply = {**base, "id": f"codex:{child}:completion:{digest(message)}", "model": "unknown",
                                 "session_id": child, "meta": {"parent_model": model, "parent_thread_id": session}}
                        reply["timestamp"] = None
                        yield Sample(channel="subagent_reply", text=message, **reply)
            elif p.get("type") == "reasoning":
                if not p.get("summary") and p.get("encrypted_content"):
                    yield Sample(channel="reasoning_unavailable", text="", status="unobservable", **base)
                for j, block in enumerate(p.get("summary") or []):
                    if block.get("text"):
                        yield Sample(channel="reasoning_summary", text=block["text"], **{**base, "id": base["id"] + f":{j}"})
            elif p.get("type") == "message":
                message = _text(p.get("content"))
                if p.get("role") == "assistant" and message.strip():
                    channel = "assistant_message" if not parent else "subagent_reply" if p.get("channel") == "final" else "subagent_message"
                    yield Sample(channel=channel, text=message, **base)
