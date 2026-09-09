"""Import Claude Code events; preserve producing-model and parent/child provenance."""
from pathlib import Path

from ..schema import Sample, read_jsonl


def iter_samples(root="~/.claude/projects"):
    for path in sorted(Path(root).expanduser().rglob("*.jsonl")):
        # Streaming snapshots reuse message IDs: retain the final snapshot of each block.
        samples = {}
        for i, record in enumerate(read_jsonl(path)):
            if record.get("type") != "assistant":
                continue
            msg = record.get("message") or {}
            content = msg.get("content") or []
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue
            child = record.get("isSidechain") or "subagents" in path.parts
            session = record.get("sessionId") or path.stem
            for j, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text, channel, status = "", None, "ok"
                if block.get("type") == "tool_use":
                    name, inp = block.get("name"), block.get("input") or {}
                    if name in {"Agent", "Task"}:
                        text, channel = inp.get("prompt") or "", "subagent_prompt"
                    elif name == "SendMessage":
                        text = inp.get("content") or inp.get("message") or ""
                        channel = "interagent_message"
                elif block.get("type") == "thinking":
                    text, channel = block.get("thinking") or "", "reasoning_summary"
                elif block.get("type") == "redacted_thinking":
                    channel, status = "reasoning_unavailable", "unobservable"
                elif block.get("type") == "text":
                    text = block.get("text") or ""
                    channel = ("subagent_reply" if msg.get("stop_reason") == "end_turn" else "subagent_message") if child else "assistant_message"
                if channel:
                    if not isinstance(text, str):
                        text, status = "", "error"
                    if not text.strip() and status == "ok":
                        status = "missing"
                    sid = f"claude:{session}:{path.stem}:{msg.get('id') or record.get('uuid') or i}:{j}"
                    samples[sid] = Sample(id=sid, provider="anthropic", model=msg.get("model") or "unknown",
                        channel=channel, text=text, status=status, source="claude_code_transcript", timestamp=record.get("timestamp"),
                        session_id=session, meta={"path": str(path), "line": i + 1, "cwd": record.get("cwd"),
                                                 "sidechain": bool(child), "agent_id": record.get("agentId")})
        yield from samples.values()
