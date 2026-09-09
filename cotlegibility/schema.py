from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Channels are the *kind of text* we measure. Keep this list small and stable; it is the main
# grouping variable in every plot.
CHANNELS = (
    "subagent_prompt",     # parent agent -> child agent (spawn message, follow-up instruction)
    "subagent_reply",      # child agent -> parent agent (final report / wait output)
    "reasoning_raw",       # raw chain of thought (only open-weight models, or published excerpts)
    "reasoning_summary",   # provider-generated summary of reasoning (OpenAI summary / Claude display=summarized)
    "assistant_message",   # text addressed to the human user
    "user_message",        # human-written text (a human-legibility reference distribution)
)


@dataclass
class Sample:
    """One unit of text to be measured."""

    id: str
    provider: str            # 'openai' | 'anthropic' | 'google' | 'open' | 'human'
    model: str               # API model id or best-known name, e.g. 'gpt-5.5', 'claude-opus-5'
    channel: str             # one of CHANNELS
    text: str
    source: str              # 'codex_rollout' | 'claude_code_transcript' | 'api_replay' | 'screenshot' | ...
    timestamp: str | None = None   # ISO-8601 when the text was generated
    session_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
