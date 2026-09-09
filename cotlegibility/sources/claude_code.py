"""Claude Code session transcripts -> Samples.

Claude Code writes one JSONL per session under ~/.claude/projects/<encoded-cwd>/<session>.jsonl.
Assistant records carry `message.model` and a content list of blocks: `text`, `thinking`,
`tool_use` (the `Agent` tool's `prompt` input is the parent -> sub-agent message).
Sub-agent turns are flagged `isSidechain: true` in the same directory.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator

from ..schema import Sample

SUBAGENT_TOOLS = {"Agent", "Task"}


def iter_samples(root: str = "~/.claude/projects") -> Iterator[Sample]:
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", "*.jsonl"), recursive=True)):
        session = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") not in ("assistant", "user"):
                    continue
                msg = r.get("message") or {}
                model = msg.get("model") or "unknown"
                ts = r.get("timestamp")
                sidechain = bool(r.get("isSidechain"))
                content = msg.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if not isinstance(content, list):
                    continue
                for j, b in enumerate(content):
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    base = dict(id=f"cc:{session}:{i}:{j}", provider="anthropic", model=model,
                                source="claude_code_transcript", timestamp=ts, session_id=session,
                                meta={"sidechain": sidechain, "path": path, "cwd": r.get("cwd")})
                    if r["type"] == "assistant" and bt == "tool_use" and b.get("name") in SUBAGENT_TOOLS:
                        inp = b.get("input") or {}
                        text = inp.get("prompt") or ""
                        if text:
                            base["meta"].update(subagent_type=inp.get("subagent_type"), description=inp.get("description"))
                            yield Sample(channel="subagent_prompt", text=text, **base)
                    elif r["type"] == "assistant" and bt == "thinking" and (b.get("thinking") or "").strip():
                        # Claude 4.6+ never returns raw thinking; anything visible is a summary.
                        yield Sample(channel="reasoning_summary", text=b["thinking"], **base)
                    elif r["type"] == "assistant" and bt == "text" and (b.get("text") or "").strip():
                        yield Sample(channel="subagent_reply" if sidechain else "assistant_message", text=b["text"], **base)
                    elif r["type"] == "user" and bt == "text" and (b.get("text") or "").strip() and not sidechain:
                        # Human-authored text (skip tool results and system-injected content).
                        t = b["text"]
                        if t.startswith("<") or "system-reminder" in t[:200]:
                            continue
                        base.update(provider="human", model="human")
                        yield Sample(channel="user_message", text=t, **base)
