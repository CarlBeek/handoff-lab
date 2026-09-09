import json

import pytest

from cotlegibility.__main__ import main
from cotlegibility.plot import bootstrap, figure, render, summarize
from cotlegibility.schema import write_jsonl


def catalog():
    return [{"id": model, "label": model, "release_date": date, "release_source": "https://example.org/fixture"}
            for model, date in (("model-a", "2026-01-01"), ("model-b", "2026-02-01"))]


def row(model, context, value, id=None, **kwargs):
    return {"id": id or f"{model}:{context}", "model": model, "source": "controlled", "channel": "subagent_prompt",
            "context_id": context, "session_id": context, "analysis_status": "ok", "bits_per_char": value,
            "text": "private source passage", "meta": {}, **kwargs}


def test_task_weighting_and_paired_intervals():
    rows = [row("model-a", "a", 1), row("model-a", "a", 3, id="replicate"), row("model-a", "b", 4),
            row("model-b", "a", 4), row("model-b", "b", 6)]
    result = summarize(rows, catalog(), resamples=1000)
    a, b = result["points"]
    assert a["mean_bpc"] == 3  # NOT the message-weighted mean 8/3
    assert a["unit_scores"] == {"a": 2, "b": 4}
    assert a["n_units"] == 2 and a["n_messages"] == 3
    assert b["mean_bpc"] == 5
    assert b["ci_low"] - a["ci_low"] == pytest.approx(2)
    assert b["ci_high"] - a["ci_high"] == pytest.approx(2)
    assert summarize(list(reversed(rows)), catalog(), resamples=1000)["points"][0]["ci_low"] == a["ci_low"]


def test_only_complete_contexts_are_compared():
    rows = [row("model-a", "shared", 1), row("model-b", "shared", 2), row("model-a", "extra", 99)]
    points = summarize(rows, catalog())["points"]
    assert [p["mean_bpc"] for p in points] == [1, 2]
    assert all(p["ci_low"] is None and p["ci_high"] is None for p in points)
    assert points[0]["n_available_units"] == 2
    assert summarize(rows[:1], catalog())["points"] == []


def test_session_mode_is_explicitly_observational():
    rows = [row("model-a", "a", 1), row("model-a", "b", 3), row("model-b", "c", 4)]
    result = summarize(rows, catalog(), unit="session", resamples=100)
    assert [p["mean_bpc"] for p in result["points"]] == [2, 4]
    assert any("Observational" in w for w in result["warnings"])
    assert result["points"][1]["ci_low"] is None


def test_filters_missingness_and_duplicate_events():
    a = row("model-a", "a", 1)
    rows = [a, a, row("model-b", "a", 2), row("model-a", "b", 9, analysis_status="error"),
            row("model-a", "c", float("nan")), row("model-a", "d", 8, channel="assistant_message"),
            row("model-a", None, 3), row("unknown", "a", 3)]
    result = summarize(rows, catalog())
    assert result["points"][0]["n_messages"] == 1
    assert result["excluded"] == {"unscored_or_failed": 2, "other_source_or_channel": 1,
                                  "missing_context_id": 1, "missing_release_metadata": 1}
    with pytest.raises(ValueError, match="Conflicting"):
        summarize([a, {**a, "bits_per_char": 2}], catalog())


@pytest.mark.parametrize("field,first,second", [("parameters", {"effort": "low"}, {"effort": "high"}),
                                               ("served_model", "snapshot1", "snapshot2")])
def test_mixed_configurations_are_not_pooled(field, first, second):
    rows = [row("model-a", "a", 1, meta={field: first}), row("model-a", "b", 2, meta={field: second})]
    with pytest.raises(ValueError, match="Mixed settings"):
        summarize(rows, catalog())


def test_context_content_and_release_dates_are_validated():
    rows = [row("model-a", "a", 1, meta={"context_hash": "one"}), row("model-b", "a", 2, meta={"context_hash": "two"})]
    with pytest.raises(ValueError, match="different task contents"):
        summarize(rows, catalog())
    models = catalog()
    models[0]["release_date"] = "2026-02-30"
    with pytest.raises(ValueError):
        summarize([], models)
    models = catalog()
    models[0]["release_source"] = ""
    with pytest.raises(ValueError, match="source URL"):
        summarize([], models)
    models[0]["release_date"] = None
    assert summarize([], models)["points"] == []


def test_bootstrap_reproducible_singleton_and_degenerate():
    assert bootstrap([1]) == (None, None)
    assert bootstrap([2, 2], 100) == (2, 2)
    assert bootstrap([1, 3, 5], 100, 12) == bootstrap([1, 3, 5], 100, 12)
    with pytest.raises(ValueError):
        bootstrap([1, 2], 0)


def test_matplotlib_shows_intervals_but_not_fake_singleton_bars():
    result = summarize([row("model-a", "a", 1), row("model-a", "b", 3), row("model-b", "c", 4)], catalog(), unit="session", resamples=100)
    fig = figure(result)
    ax = fig.axes[0]
    assert len(ax.containers) == 1  # only model-a has enough sessions for an error bar
    interval = ax.containers[0].lines[2][0].get_segments()[0]
    assert list(interval[:, 1]) == [result["points"][0]["ci_low"], result["points"][0]["ci_high"]]
    assert "bits / character" in ax.get_ylabel()
    assert "Model public-release date" == ax.get_xlabel()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_png_export_and_data_sidecar(tmp_path):
    write_jsonl(tmp_path / "manifest.jsonl", [{"reference": {"model": "fixture-reader", "revision": "fixture"}}])
    write_jsonl(tmp_path / "measured_messages.jsonl", [row("model-a", "a", 1), row("model-b", "a", 2)])
    result = render(tmp_path, catalog(), resamples=100)
    assert (tmp_path / "surprisal.png").read_bytes().startswith(b"\x89PNG")
    assert "private source passage" not in (tmp_path / "surprisal.jsonl").read_text()
    assert any("single independent unit" in w for w in result["warnings"])
    assert (tmp_path / "surprisal.jsonl").exists()
    write_jsonl(tmp_path / "manifest.jsonl", [{"reference": None}])
    with pytest.raises(ValueError, match="no reference scores"):
        render(tmp_path, catalog())


def test_empty_chart_and_cli(tmp_path):
    write_jsonl(tmp_path / "manifest.jsonl", [{"reference": {"model": "fixture-reader"}}])
    write_jsonl(tmp_path / "measured_messages.jsonl", [])
    models = tmp_path / "models.json"
    write_jsonl(models, [catalog()])  # a one-line JSON array is also valid JSON
    main(["plot", str(tmp_path), "--models", str(models), "--resamples", "100"])
    assert (tmp_path / "surprisal.png").exists()
    data = json.loads((tmp_path / "surprisal.jsonl").read_text())
    assert data["points"] == []
