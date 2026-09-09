"""Fixed-context collection: one visible handoff, no child execution or experimental factors."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .api import ENDPOINTS, provider_for, request
from .schema import Sample, digest, read_jsonl, write_jsonl

INSTRUCTION = "You are coordinating the work described below. Delegate the specified subtask by calling spawn_agent exactly once."
TOOL = {"name": "spawn_agent", "description": "Delegate work to a child agent. It sees the repository and your message, but not this conversation.",
        "schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"], "additionalProperties": False}}


def cells(contexts, models, replicates):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not isinstance(models, list) or not models:
        raise ValueError("Models must be a nonempty list of {id, parameters} objects")
    seen_models = set()
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"].strip():
            raise ValueError("Each model needs a nonempty string id")
        if not isinstance(model.get("parameters", {}), dict):
            raise ValueError("Model parameters must be an object")
        key = digest(model)
        if key in seen_models:
            raise ValueError(f"Duplicate model configuration: {model['id']}")
        seen_models.add(key)
    seen = set()
    for context in contexts:
        if not isinstance(context.get("id"), str) or not context["id"].strip():
            raise ValueError("Each context needs a nonempty string id")
        if context["id"] in seen:
            raise ValueError(f"Duplicate context ID: {context['id']}")
        seen.add(context["id"])
        if not context.get("text", "").strip():
            raise ValueError("Each context needs text")
        for model in models:
            for replicate in range(replicates):
                prompt = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": context["text"]}]
                cell = {"context": context, "model": model, "replicate": replicate, "messages": prompt, "tool": TOOL,
                        "endpoint": ENDPOINTS[provider_for(model["id"])], "protocol": "handoff-v2"}
                yield digest(cell), cell


def collect(contexts, models, output, replicates=1, limit=10, dry_run=False):
    output = Path(output)
    planned = list(cells(contexts, models, replicates))
    pending = [(key, cell) for key, cell in planned if not (output / "raw" / f"{key}.jsonl").exists()]
    print(f"{len(planned)} cells; {len(pending)} uncached; at most {limit} new requests")
    if dry_run:
        for _, cell in pending[:limit]:
            print(f"  {cell['model']['id']} / {cell['context']['id']} / replicate {cell['replicate']}")
        return
    for key, cell in pending[:limit]:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = request(cell["model"]["id"], cell["messages"], parameters=cell["model"].get("parameters"), tool=TOOL)
            status = "ok"
        except Exception as exc:
            response, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
        write_jsonl(output / "raw" / f"{key}.jsonl", [{"cell": cell, "timestamp": timestamp, "status": status, "response": response}])
        print(f"{cell['model']['id']} / {cell['context']['id']}: {status}", flush=True)
    samples = []
    for key, cell in planned:
        path = output / "raw" / f"{key}.jsonl"
        record = next(read_jsonl(path)) if path.exists() else {
            "response": {"error": "Not collected within the request budget"}, "status": "missing", "timestamp": None}
        response, text = record["response"], ""
        status = record["status"]
        calls = [c for c in response.get("tool_calls", []) if c["name"] == "spawn_agent"]
        if status == "ok":
            try:
                if len(calls) != 1:
                    raise ValueError(f"Expected one spawn call, received {len(calls)}")
                arguments = calls[0]["arguments"]
                text = (json.loads(arguments) if isinstance(arguments, str) else arguments)["message"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty or non-string handoff")
                if response.get("stop_reason") in {"incomplete", "max_tokens", "refusal", "failed"}:
                    status = "error"
            except (ValueError, KeyError, TypeError) as exc:
                status, text = "missing", ""
                response["parse_error"] = str(exc)
        samples.append(Sample(id=key, provider=provider_for(cell["model"]["id"]), model=cell["model"]["id"],
            channel="subagent_prompt", source="controlled", text=text, timestamp=record["timestamp"], context_id=cell["context"]["id"], status=status,
            meta={"context_hash": digest(cell["context"]), "protocol": cell["protocol"], "replicate": cell["replicate"],
                  "parameters": cell["model"].get("parameters", {}), "raw_path": str(path) if path.exists() else None,
                  "collection_status": "attempted" if path.exists() else "not_collected",
                  "reader_context": cell["context"].get("reader_context", ""),
                  **{k: response.get(k) for k in ("served_model", "endpoint", "response_id", "request_id", "usage", "stop_reason", "error", "parse_error")}}).to_dict())
    write_jsonl(output / "messages.jsonl", samples)
