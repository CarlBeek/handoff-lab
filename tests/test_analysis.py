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
from test_collection import contexts, models, fake_client, response

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
    return models() + [{**models()[0], "id": "second-fixture", "label": "Second fixture",
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
    ("incomplete", "incomplete_response"), ("missing_model", "missing_model_identity"),
    ("invalid_json", "invalid_handoff"), ("extra_call", "invalid_handoff"),
    ("wrong_tool", "invalid_handoff"), ("blank", "invalid_handoff"),
    ("extra_argument", "invalid_handoff"), ("valid", "ok"),
])
def test_extraction_preserves_failures(analysis, change, expected):
    spec = collect.plan(contexts()[:1], models(), "pilot")[0]
    raw = response(message="Inspectthecache `identifier`").model_dump()
    record = {"state": "finished", "response": raw}
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
    path = next(tmp_path.glob("request-*.json"))
    record = json.loads(path.read_text())
    record["request"]["model"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagrees"):
        analysis.load_run(tmp_path)


@pytest.mark.parametrize("populated", [False, True])
def test_notebook_runs_all_without_network_or_paid_calls(tmp_path, populated):
    run = tmp_path / "raw"
    if populated:
        seen = []
        def reply(**body):
            seen.append(body)
            return response(model=body["model"], message="Inspect the cache." + " Include evidence." * len(seen))
        collect.collect(contexts(), catalog(), run, execute=True, max_calls=6, client=fake_client(reply))
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
    assert summary["n_matched_tasks"] == (3 if populated else 0)
    assert len(summary["points"]) == (2 if populated else 0)
