"""OpenAI Codex CLI/app session rollouts -> Samples.

Codex writes one JSONL "rollout" per thread under ~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl.
Records: {"timestamp", "type", "payload"}. Types used here:
  session_meta  - id, cli_version, source ('cli' | 'vscode' | {'subagent': {'thread_spawn': {parent_thread_id, agent_nickname, ...}}})
  turn_context  - model for the following turn (e.g. 'gpt-5.5', 'gpt-5.6-sol')
  response_item - payload.type in: message (role user/assistant), function_call (name, arguments, call_id),
                  function_call_output (call_id, output), reasoning (summary[], encrypted_content)
Multi-agent tools: spawn_agent {agent_type?, fork_context?, message} -> {"agent_id", "nickname"};
wait_agent {targets[]} -> {"status": {agent_id: {"completed": <child final message>}}}; close_agent.
Sub-agent threads are separate rollouts whose session_meta.source names the parent thread.
Raw reasoning is encrypted; only `summary` texts are readable.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator

from ..schema import Sample

SPAWN_TOOLS = {"spawn_agent"}
SEND_TOOLS = {"send_input", "send_message", "message_agent"}   # follow-up parent -> child channels, if present
WAIT_TOOLS = {"wait_agent"}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text"))
    return ""


def imported_thread_ids(codex_home: str = "~/.codex") -> set[str]:
    """Codex Desktop can import Claude Code sessions; those rollouts are Claude output, not OpenAI output.
    The Claude Code parser is the authoritative source for them, so we skip them here."""
    p = os.path.join(os.path.expanduser(codex_home), "external_agent_session_imports.json")
    if not os.path.exists(p):
        return set()
    try:
        return {r.get("imported_thread_id") for r in json.load(open(p)).get("records", [])}
    except (json.JSONDecodeError, AttributeError):
        return set()


def iter_samples(root: str = "~/.codex/sessions", skip_imported: bool = True) -> Iterator[Sample]:
    imported = imported_thread_ids() if skip_imported else set()
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", "*.jsonl"), recursive=True)):
        if any(tid and tid in path for tid in imported):
            continue
        model = "unknown"
        session = None
        cli = None
        parent = None
        nickname = None
        cwd = None
        calls: dict[str, tuple[str, dict]] = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t, p, ts = r.get("type"), r.get("payload") or {}, r.get("timestamp")
                if t == "session_meta":
                    session = p.get("id") or session
                    cli = p.get("cli_version") or cli
                    cwd = p.get("cwd") or cwd
                    src = p.get("source")
                    if isinstance(src, dict) and "subagent" in src:
                        spawn = (src["subagent"] or {}).get("thread_spawn") or {}
                        parent, nickname = spawn.get("parent_thread_id"), spawn.get("agent_nickname")
                    continue
                if t == "turn_context":
                    model = p.get("model") or model
                    continue
                if t != "response_item":
                    continue
                pt = p.get("type")
                base = dict(provider="openai", model=model, source="codex_rollout", timestamp=ts, session_id=session,
                            meta={"cli_version": cli, "path": path, "parent_thread_id": parent, "nickname": nickname, "cwd": cwd})
                if pt == "function_call":
                    name = p.get("name")
                    try:
                        args = json.loads(p.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls[p.get("call_id")] = (name, args)
                    if name in SPAWN_TOOLS or name in SEND_TOOLS:
                        text = args.get("message") or args.get("input") or args.get("prompt") or ""
                        if text:
                            base["meta"].update(tool=name, agent_type=args.get("agent_type"), fork_context=args.get("fork_context"),
                                                encrypted=text.startswith("gAAAA"))   # MultiAgent v2 ciphertext (since Codex 0.153.x here)
                            yield Sample(id=f"codex:{session}:{i}", channel="subagent_prompt", text=text, **base)
                elif pt == "function_call_output":
                    name, _ = calls.get(p.get("call_id"), (None, {}))
                    if name in WAIT_TOOLS:
                        try:
                            out = json.loads(p.get("output") or "{}")
                        except json.JSONDecodeError:
                            continue
                        for agent_id, st in (out.get("status") or {}).items():
                            text = st.get("completed") if isinstance(st, dict) else None
                            if text:
                                base["meta"].update(child_agent_id=agent_id)
                                yield Sample(id=f"codex:{session}:{i}:{agent_id[:8]}", channel="subagent_reply", text=text, **base)
                elif pt == "reasoning":
                    for j, s in enumerate(p.get("summary") or []):
                        text = s.get("text") if isinstance(s, dict) else None
                        if text:
                            yield Sample(id=f"codex:{session}:{i}:{j}", channel="reasoning_summary", text=text, **base)
                elif pt == "message":
                    text = _text_of(p.get("content"))
                    if not text.strip():
                        continue
                    if p.get("role") == "assistant":
                        yield Sample(id=f"codex:{session}:{i}", channel="assistant_message", text=text, **base)
                    elif p.get("role") == "user" and not text.lstrip().startswith("<"):
                        # Human-typed text; skip injected <environment_context>/<user_instructions> blocks.
                        base.update(provider="human", model="human")
                        yield Sample(id=f"codex:{session}:{i}", channel="user_message", text=text, **base)
