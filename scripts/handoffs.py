"""Read-only manifest validation and handoff extraction, shared by both analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_run(directory, model_ids=None, task_limit=None, *, expected_protocols=None):
    from scripts import collect

    directory = Path(directory)
    if task_limit is not None and (type(task_limit) is not int or task_limit < 1):
        raise ValueError("task_limit must be positive")
    if model_ids is not None and (not model_ids or len(set(model_ids)) != len(model_ids)):
        raise ValueError("Select a nonempty, unique list of model IDs")
    study_path = directory / "study.json"
    study = json.loads(study_path.read_text(encoding="utf-8")) if study_path.exists() else None
    if study and expected_protocols is not None and study["protocol"] not in expected_protocols:
        raise ValueError("Wrong protocol for this analysis; keep audience and legacy runs separate")
    if study and (directory / "run.json").exists():
        raise ValueError("Keep legacy and per-model layouts in separate directories")
    paths = sorted(directory.glob("*/run.json")) if study else [directory / "run.json"]
    orphans = [p for p in directory.glob("*/request-*.json") if not (p.parent / "run.json").exists()]
    if orphans or (not study and list(directory.glob("*/run.json"))):
        raise ValueError("Raw model runs exist without a manifest")
    aggregate = {"study": study, "models": [], "requests": [], "model_runs": []}
    rows = []
    slots = set()
    for path in paths:
        if not path.exists():
            if list(path.parent.glob("request-*.json")):
                raise ValueError("Raw requests exist without a run manifest")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if expected_protocols is not None and manifest["protocol"] not in expected_protocols:
            raise ValueError("Wrong protocol for this analysis; keep audience and legacy runs separate")
        if manifest["protocol"] == collect.AUDIENCE_PROTOCOL and not study:
            raise ValueError("Audience runs require a separate study manifest")
        models_here = [manifest["model"]] if study else manifest["models"]
        if study:
            study_hash = hashlib.sha256(json.dumps(study, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if manifest["study_hash"] != study_hash or manifest["protocol"] != study["protocol"] or manifest["split"] != study["split"]:
                raise ValueError("Model run belongs to a different study")
            if path.parent.name != models_here[0]["id"]:
                raise ValueError("Model directory disagrees with manifest")
            expected_tasks = [(t["id"], t["context_hash"]) for t in study["tasks"]]
            actual_tasks = [(r["task_id"], r["context_hash"]) for r in manifest["requests"]]
            if study["protocol"] == collect.AUDIENCE_PROTOCOL:
                validate_audience_plan(study, manifest)
                slot = manifest["model_slot"]
                if slot in slots:
                    raise ValueError("Duplicate audience model schedule slot")
                slots.add(slot)
                expected_tasks = [task for task in expected_tasks for _ in collect.AUDIENCES]
            if actual_tasks != expected_tasks:
                raise ValueError("Model run has different frozen tasks")
            if any(r["model_id"] != models_here[0]["id"] or r["study_hash"] != study_hash
                   or r["provider"] != manifest["route"]["provider"]
                   or r["parameters"] != models_here[0]["parameters"]
                   or r["api"] != models_here[0]["api"]
                   or r["protocol"] != study["protocol"] or r["split"] != study["split"]
                   for r in manifest["requests"]):
                raise ValueError("Request disagrees with model/study manifest")
        selected_models = [m for m in models_here if model_ids is None or m["id"] in model_ids]
        if not selected_models:
            continue
        expected = manifest["requests"]
        ids = {r["id"] for r in expected}
        if len(ids) != len(expected):
            raise ValueError("Duplicate requests in run manifest")
        extras = {p.stem.removeprefix("request-") for p in path.parent.glob("request-*.json")} - ids
        if extras:
            raise ValueError("Unexpected requests in this run directory")
        selected_ids = {m["id"] for m in selected_models}
        tasks_here = list(dict.fromkeys(r["task_id"] for r in expected))
        selected_tasks = set(tasks_here[:task_limit])
        aggregate["models"].extend(selected_models)
        aggregate["model_runs"].append(manifest)
        for spec in expected:
            if spec["model_id"] not in selected_ids or spec["task_id"] not in selected_tasks:
                continue
            identity = {k: v for k, v in spec.items() if k != "id"}
            if hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != spec["id"]:
                raise ValueError("Request hash disagrees with manifest")
            record_path = path.parent / f"request-{spec['id']}.json"
            record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
            if record_path.exists() and any(record.get(k) != v for k, v in spec.items()):
                raise ValueError(f"Saved request disagrees with manifest: {spec['id']}")
            rows.append(extract_handoff(spec, record))
            aggregate["requests"].append(spec)
    if model_ids is not None and set(model_ids) != {m["id"] for m in aggregate["models"]}:
        raise ValueError("Selected model has no saved run")
    if not aggregate["models"]:
        return None, []
    aggregate["split"] = aggregate["model_runs"][0]["split"]
    aggregate["protocol"] = aggregate["model_runs"][0]["protocol"]
    if any(m["protocol"] != aggregate["protocol"] or m["split"] != aggregate["split"]
           for m in aggregate["model_runs"]):
        raise ValueError("Do not combine protocols or pilot/main splits")
    return aggregate, rows


def validate_audience_plan(study, manifest):
    """Reject missing/duplicated conditions, edited prompts, and changed schedules."""
    from scripts import collect

    if (study.get("audiences") != collect.AUDIENCES or study.get("schedule") != collect.SCHEDULE
            or study.get("instruction") != collect.AUDIENCE_INSTRUCTION
            or study.get("tool_name") != "send_handoff"
            or study.get("tool_description") != collect.AUDIENCE_DESCRIPTION
            or study.get("tool_schema") != collect.SCHEMA or study.get("tool_choice") != "auto"
            or manifest.get("audiences") != collect.AUDIENCES):
        raise ValueError("Audience protocol definition changed")
    slot = manifest.get("model_slot")
    if type(slot) is not int or slot < 0:
        raise ValueError("Invalid audience model schedule slot")
    requests = manifest["requests"]
    if len(requests) != len(study["tasks"]) * 2:
        raise ValueError("Expected two audience conditions for every frozen task")
    model, provider = manifest["model"], manifest["route"]["provider"]
    collect.validate_models([model])
    if provider not in model["providers"]:
        raise ValueError("Unapproved saved route")
    for i, task in enumerate(study["tasks"]):
        pair = requests[2*i:2*i+2]
        try:
            evidence_key = "input" if model["api"] == "responses" else "messages"
            evidence = pair[0]["request"][evidence_key][-1]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Missing saved task evidence") from None
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Missing saved task evidence")
        for position, (spec, audience) in enumerate(zip(pair, collect.audience_order(i, slot))):
            if (spec.get("audience") != audience or spec["task_id"] != task["id"]
                    or spec["context_hash"] != task["context_hash"] or spec["source"] != task["source"]
                    or spec.get("schedule") != {"task_index": i, "model_slot": slot, "audience_position": position}
                    or spec["request"] != collect.request_body(evidence, model, provider, audience)
                    or spec["response_models"] != model["response_models"]
                    or spec["endpoint"] != collect.surplus.ENDPOINTS[model["api"]]):
                raise ValueError("Request disagrees with frozen audience condition or schedule")


def extract_handoff(spec, record):
    response = record.get("response")
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("output_tokens_details")
    details = details if isinstance(details, dict) else {}
    api = spec.get("api", "responses")
    parameters = spec.get("parameters", {k: spec["request"][k]
        for k in ("reasoning", "max_output_tokens") if k in spec["request"]})
    headers = record.get("response_headers")
    headers = headers if isinstance(headers, dict) else {}
    row = {
        "id": spec["id"], "task_id": spec["task_id"], "model_id": spec["model_id"],
        "context_hash": spec["context_hash"], "repo": spec["source"]["repo"],
        "protocol": spec["protocol"], "split": spec["split"], "api": api,
        "audience": spec.get("audience"), "schedule": spec.get("schedule"),
        "parameters": json.dumps(parameters, sort_keys=True),
        "provider": headers.get("x-si-provider-family") if "provider" in spec else "openai",
        "served_model": response.get("model") if isinstance(response.get("model"), str) else None, "collected_at": record.get("started_at"),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cost_usd": None, "status": record.get("state", "not_attempted"),
        "text": None, "prose": None, "bits_per_char": None, "reference_device": None,
        "total_bits": None, "tokens": None, "characters": None,
        "validation_errors": record.get("validation_errors", []),
    }
    if "provider" in spec:
        from scripts import surplus
        cost = surplus.cost_micro(headers)
        row["cost_usd"] = cost / 1e6 if cost is not None else None
    if row["status"] != "finished":
        return row
    if "provider" in spec and (surplus.audit(spec, response, headers) or record.get("validation_errors")):
        row["status"] = "invalid_provenance"
        return row
    if api == "responses" and response.get("status") != "completed":
        row["status"] = "incomplete_response"
        return row
    if api == "messages" and response.get("stop_reason") != "tool_use":
        row["status"] = "refusal" if response.get("stop_reason") == "refusal" else "incomplete_response"
        return row
    if not isinstance(row["served_model"], str) or not row["served_model"]:
        row["status"] = "missing_model_identity"
        return row
    try:
        blocks = response.get("output" if api == "responses" else "content", [])
        if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
            raise ValueError("Invalid content blocks")
        calls = [o for o in blocks if o.get("type") == ("function_call" if api == "responses" else "tool_use")]
        from scripts.collect import AUDIENCE_PROTOCOL
        tool = "send_handoff" if spec["protocol"] == AUDIENCE_PROTOCOL else "spawn_agent"
        if len(calls) != 1 or calls[0].get("name") != tool:
            raise ValueError(f"Expected exactly one {tool} call")
        args = json.loads(calls[0]["arguments"]) if api == "responses" else calls[0]["input"]
        if not isinstance(args, dict) or set(args) != {"message"}:
            raise ValueError("Expected only a message field")
        if not isinstance(args["message"], str) or not args["message"].strip():
            raise ValueError("Empty/non-string handoff")
        row.update(status="ok", text=args["message"])
    except (ValueError, KeyError, TypeError):
        row["status"] = "invalid_handoff"
    return row
