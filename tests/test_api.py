import sys
from types import SimpleNamespace

import pytest

from cotlegibility.api import request
from cotlegibility.collect import TOOL
from cotlegibility.metrics.judge import SCHEMA


def test_openai_adapter_records_exact_request_and_response(monkeypatch):
    seen = {}
    raw = {"id": "r1", "model": "served-snapshot", "status": "completed", "usage": {"input_tokens": 10},
           "output": [{"type": "function_call", "name": "spawn_agent", "arguments": '{"message":"Review"}'}]}

    def create(**body):
        seen["body"] = body
        return SimpleNamespace(model_dump=lambda **kw: raw, output_text="", _request_id="req1")

    def client(**kwargs):
        seen["client"] = kwargs
        return SimpleNamespace(responses=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    result = request("producer", [{"role": "user", "content": "Delegate"}], parameters={"max_output_tokens": 100}, tool=TOOL)
    assert result["request"] == seen["body"] and result["raw"] == raw
    assert seen["client"]["max_retries"] == 0 and seen["client"]["base_url"] == "https://api.openai.com/v1"
    assert seen["body"]["store"] is False and seen["body"]["tools"][0]["strict"] is True
    assert result["served_model"] == "served-snapshot" and result["request_id"] == "req1"
    request("reader", [{"role": "user", "content": "Read"}], output_schema=SCHEMA)
    assert seen["body"]["text"]["format"]["schema"] == SCHEMA


def test_anthropic_adapter_streams_and_keeps_model_settings(monkeypatch):
    seen = {}
    raw = {"id": "r1", "model": "claude-snapshot", "stop_reason": "tool_use", "content": [
        {"type": "text", "text": "Delegating"}, {"type": "tool_use", "name": "spawn_agent", "input": {"message": "Review"}}]}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_final_message(self):
            return SimpleNamespace(model_dump=lambda **kw: raw)

    def stream(**body):
        seen.update(body)
        return Stream()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kw: SimpleNamespace(messages=SimpleNamespace(stream=stream))))
    result = request("claude-test", [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Task"}],
                     parameters={"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}, tool=TOOL)
    assert seen["system"] == "Rules" and len(seen["messages"]) == 1
    assert seen["thinking"] == {"type": "adaptive"}
    assert result["tool_calls"][0]["arguments"]["message"] == "Review"
    assert result["raw"] == raw


def test_parameters_cannot_replace_protocol():
    with pytest.raises(ValueError, match="override"):
        request("model", [], parameters={"model": "different"})
