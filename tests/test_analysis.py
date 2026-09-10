from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pytest

from scripts import collect
from test_collection import contexts, models, make_model, fake_client, response, wire_headers

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/analysis.ipynb"


@pytest.fixture
def analysis():
    namespace = {"date": date, "json": json, "Path": Path, "np": np, "pd": pd}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if "functions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, f"{NOTEBOOK}:{cell.id}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def catalog():
    return models() + [{**make_model("second-fixture"), "label": "Second fixture",
                       "release_date": "2026-02-01"}]


def measured():
    rows = []
    for i, value in enumerate([1.0, 3.0, 8.0]):
        for j, model in enumerate(catalog()):
            rows.append({"id": f"{i}-{j}", "task_id": str(i), "model_id": model["id"],
                         "context_hash": str(i), "protocol": "fixture", "split": "pilot",
                         "parameters": "fixed", "served_model": model["id"],
                         "status": "ok", "bits_per_char": value + j * 2})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("change,expected", [
    ("missing", "not_attempted"), ("started", "started"), ("error", "error"),
    ("incomplete", "incomplete_response"), ("missing_model", "invalid_provenance"),
    ("invalid_json", "invalid_handoff"), ("extra_call", "invalid_handoff"),
    ("wrong_tool", "invalid_handoff"), ("blank", "invalid_handoff"),
    ("extra_argument", "invalid_handoff"), ("valid", "ok"),
])
def test_extraction_preserves_failures(analysis, change, expected):
    spec = collect.plan(contexts()[:1], models(), "pilot")[0]
    raw = response(message="Inspectthecache `identifier`").model_dump()
    record = {"state": "finished", "response": raw, "response_headers": wire_headers()}
    if change == "missing":
        record = {}
    elif change in {"started", "error"}:
        record["state"] = change
    elif change == "incomplete":
        raw["status"] = "incomplete"
    elif change == "missing_model":
        raw["model"] = None
    elif change == "invalid_json":
        raw["output"][0]["arguments"] = "NOT JSON"
    elif change == "extra_call":
        raw["output"] *= 2
    elif change == "wrong_tool":
        raw["output"][0]["name"] = "other"
    elif change == "blank":
        raw["output"][0]["arguments"] = '{"message":"  "}'
    elif change == "extra_argument":
        raw["output"][0]["arguments"] = '{"message":"Inspect", "other":true}'
    result = analysis.extract_handoff(spec, record)
    assert result["status"] == expected and result["bits_per_char"] is None
    if change == "valid":
        assert result["text"] == "Inspectthecache `identifier`"


def test_paired_bootstrap_reproducible_and_task_weighted(analysis):
    a = analysis.summarize(measured(), catalog(), resamples=500)
    assert a == analysis.summarize(measured(), catalog(), resamples=500)
    first, second = a["points"]
    assert first["mean_bpc"] == 4
    assert second["mean_bpc"] == 6
    assert second["ci_low"] - first["ci_low"] == pytest.approx(2)
    assert second["ci_high"] - first["ci_high"] == pytest.approx(2)
    assert a["task_ids"] == ["0", "1", "2"]


def test_missing_cells_and_models_do_not_silently_disappear(analysis):
    df = measured()
    df.loc[0, ["status", "bits_per_char"]] = ["incomplete_response", None]
    result = analysis.summarize(df, catalog(), resamples=100)
    assert result["n_matched_tasks"] == 2 and result["n_planned_tasks"] == 3
    assert result["task_ids"] == ["1", "2"]
    df.loc[df["model_id"] == catalog()[0]["id"], ["status", "bits_per_char"]] = ["error", None]
    assert analysis.summarize(df, catalog())["points"] == []
    assert analysis.summarize(df.iloc[:0], catalog())["points"] == []
    one = measured().iloc[:2]
    assert analysis.summarize(one, catalog())["points"][0]["ci_low"] is None


@pytest.mark.parametrize("column,value", [
    ("served_model", "different-snapshot"), ("parameters", "different-settings"),
    ("context_hash", "different-context"), ("protocol", "different-protocol"),
    ("split", "main"), ("bits_per_char", float("nan")),
    ("bits_per_char", float("inf")), ("bits_per_char", -1),
])
def test_mixed_provenance_or_invalid_scores_fail(analysis, column, value):
    df = measured()
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        analysis.summarize(df, catalog())


def test_load_run_validates_plan_and_includes_unattempted(analysis, tmp_path):
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1,
                    client=fake_client(lambda **kw: response()))
    manifest, rows = analysis.load_run(tmp_path)
    assert len(rows) == 3 and [r["status"] for r in rows].count("not_attempted") == 2
    assert analysis.load_run(tmp_path / "missing") == (None, [])
    path = next(tmp_path.rglob("request-*.json"))
    record = json.loads(path.read_text())
    record["request"]["model"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagrees"):
        analysis.load_run(tmp_path)


@pytest.mark.parametrize("panel_size", [0, 2, 10])
def test_notebook_runs_all_without_network_or_paid_calls(tmp_path, panel_size):
    run = tmp_path / "raw"
    if panel_size:
        panel = catalog() if panel_size == 2 else collect.read_json(ROOT / "data/models.json")
        seen = []
        def reply(**body):
            seen.append(body)
            message = "Inspect the cache." + " Include evidence." * (len(seen) % 3 + 1)
            if "messages" in body:
                return {"model": body["model"], "stop_reason": "tool_use", "content": [
                    {"type": "tool_use", "name": "spawn_agent", "input": {"message": message}}]}
            return response(model=body["model"], message=message)
        collect.collect(contexts(), panel, run, execute=True, max_calls=panel_size * 3, client=fake_client(reply))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.id == "setup":
            cell.source = cell.source.replace('RUN_DIR = ROOT / "data/raw/pilot"', f"RUN_DIR = Path({str(run)!r})")
            cell.source = cell.source.replace('OUT_DIR = ROOT / "out/analysis-pilot"', f"OUT_DIR = Path({str(tmp_path / 'analysis')!r})")
            # Test the notebook orchestration without downloading reference weights.
            # The actual scoring math has its own shifted-token/Unicode tests.
            cell.source += '\nsurprisal.bits_per_char = lambda text, **kw: {"bits_per_char": len(text) / 10}\n'
            cell.source += '\nimport socket\ndef deny_network(*args, **kwargs):\n    raise AssertionError("Notebook attempted a network connection")\nsocket.socket.connect = deny_network\n'
    NotebookClient(notebook, timeout=60, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    output = tmp_path / "analysis"
    assert (output / "surprisal.png").stat().st_size > 1000
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_matched_tasks"] == (3 if panel_size else 0)
    assert len(summary["points"]) == panel_size
    if panel_size == 10:
        print(f"Synthetic ten-model plot: {output / 'surprisal.png'}")


def test_messages_extraction_and_truncation(analysis):
    model = make_model("claude-fable-5.1", "messages")
    model["response_models"].append("claude-fable-5-1")
    spec = collect.plan(contexts()[:1], [model], "pilot")[0]
    raw = {"model": "claude-fable-5-1", "stop_reason": "tool_use", "content": [
        {"type": "thinking", "thinking": "Not the measured artifact"},
        {"type": "tool_use", "name": "spawn_agent", "input": {"message": "Inspectthecache."}}],
        "usage": {"input_tokens": 100, "output_tokens": 40,
                  "cache_read_input_tokens": 20, "cache_creation_input_tokens": 10}}
    record = {"state": "finished", "response": raw, "response_headers": wire_headers("anthropic")}
    row = analysis.extract_handoff(spec, record)
    assert row["status"] == "ok" and row["text"] == "Inspectthecache."
    assert row["cost_usd"] == .001 and row["cache_read_input_tokens"] == 20
    assert row["reasoning_tokens"] is None  # Never invent a breakdown.
    raw["stop_reason"] = "max_tokens"
    assert analysis.extract_handoff(spec, record)["status"] == "incomplete_response"
    raw["stop_reason"] = "refusal"
    assert analysis.extract_handoff(spec, record)["status"] == "refusal"
    raw["stop_reason"] = "tool_use"
    record["response_headers"]["x-si-truncated"] = "1"
    assert analysis.extract_handoff(spec, record)["status"] == "invalid_provenance"


def test_load_combines_new_models_and_selects_panel_and_task_prefix(analysis, tmp_path):
    collect.collect(contexts(), models(), tmp_path, tasks=2, execute=True, max_calls=2, client=fake_client())
    before, rows = analysis.load_run(tmp_path)
    assert len(before["models"]) == 1 and sum(r["status"] == "ok" for r in rows) == 2
    newer = make_model("sixth-model")
    collect.collect(contexts(), [newer], tmp_path, tasks=1, execute=True, max_calls=1,
                    client=fake_client(lambda **body: response(model=body["model"])))
    combined, rows = analysis.load_run(tmp_path)
    assert len(combined["models"]) == 2 and len(rows) == 6
    selected, subset = analysis.load_run(tmp_path, model_ids=[models()[0]["id"]], task_limit=2)
    assert selected["models"] == before["models"]
    assert len(subset) == 2 and all(r["status"] == "ok" for r in subset)
    with pytest.raises(ValueError, match="no saved run"):
        analysis.load_run(tmp_path, model_ids=["unknown-model"])
    path = tmp_path / "sixth-model" / "run.json"
    manifest = json.loads(path.read_text())
    manifest["study_hash"] = "wrong-study"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different study"):
        analysis.load_run(tmp_path)


def test_legacy_run_still_readable(analysis, tmp_path):
    spec = collect.plan(contexts()[:1], models(), "pilot")[0]
    for key in ("id", "study_hash", "provider", "api", "parameters", "response_models"):
        spec.pop(key)
    spec["protocol"] = "swe-handoff-v1"
    spec["id"] = collect.digest(spec)
    manifest = {"models": models(), "requests": [spec], "protocol": spec["protocol"], "split": "pilot"}
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / f"request-{spec['id']}.json").write_text(json.dumps({**spec, "state": "finished",
                                                                   "response": response().model_dump()}))
    loaded, rows = analysis.load_run(tmp_path)
    assert loaded["protocol"] == "swe-handoff-v1" and rows[0]["status"] == "ok"


def test_audit_rechecked_not_only_saved_flag(analysis):
    spec = collect.plan(contexts()[:1], models(), "pilot")[0]
    record = {"state": "finished", "response": response().model_dump(),
              "response_headers": {**wire_headers(), "x-si-adapted-params": "reasoning_effort"},
              "validation_errors": []}
    assert analysis.extract_handoff(spec, record)["status"] == "invalid_provenance"


@pytest.mark.parametrize("usage", [[], {"output_tokens_details": []}])
def test_malformed_optional_usage_does_not_lose_handoff(analysis, usage):
    spec = collect.plan(contexts(), models(), "pilot")[0]
    raw = response().model_dump()
    raw["usage"] = usage
    row = analysis.extract_handoff(spec, {"state": "finished", "response": raw,
                                          "response_headers": wire_headers()})
    assert row["status"] == "ok" and row["reasoning_tokens"] is None


def test_mixed_directory_layout_refused(analysis, tmp_path):
    collect.collect(contexts(), models(), tmp_path, execute=True, max_calls=1, client=fake_client())
    (tmp_path / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="separate directories"):
        analysis.load_run(tmp_path)
