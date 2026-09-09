import json
import math
from types import SimpleNamespace

import pytest

from cotlegibility.analyze import measure, prose_segment
from cotlegibility.metrics import judge, perplexity
from cotlegibility.metrics.surface import NONPROSE, passages, prose_only, spacing
from cotlegibility.schema import read_jsonl


def test_spacing_preserves_characters_and_offsets():
    text = "Inspectthecacheupdatelogic. pharmacokinetics FooBar"
    result = spacing(text)
    assert result["restored"] == "Inspect the cache update logic. pharmacokinetics FooBar"
    assert result["restored"].replace(" ", "") == text.replace(" ", "")
    assert result["camel_boundaries"] == 1
    for span in result["spacing_spans"]:
        assert text[span["start"]:span["end"]] == span["original"]
    assert spacing("123 --")["glued_word_share"] is None


def test_code_masks_preserve_offsets_and_prose():
    text = "day/week `Inspectthecache` ./src/mod.py https://example.com/a\n```py\n" + "x " * 100
    masked = prose_only(text)
    assert len(masked) == len(text)
    assert masked.startswith("day/week ")
    assert masked.count("\n") == text.count("\n")
    assert not spacing(masked)["spacing_spans"]
    exclusions = list(NONPROSE.finditer(text))
    assert prose_segment(text, exclusions, 0, len(text)) == "day/week"
    assert prose_segment(text, exclusions, len(text) - 50, len(text)) == ""


@pytest.mark.parametrize("text", ["", "Short.", "word " * 40, "fused" * 80, "a\n" * 60])
def test_passages_are_lossless(text):
    chunks = list(passages(text, 50))
    assert "".join(p[2] for p in chunks) == text
    previous = 0
    for start, end, chunk in chunks:
        assert start == previous and text[start:end] == chunk
        previous = end


@pytest.mark.parametrize("length,stride", [(0, 2), (1, 2), (4, 2), (5, 2), (17, 1), (17, 2), (17, 4)])
def test_reference_scores_each_shifted_token_once(monkeypatch, length, stride):
    torch = pytest.importorskip("torch")
    seen = []

    class Model:
        config = SimpleNamespace(n_positions=5)

        def __call__(self, chunk, labels):
            assert chunk.shape[1] <= 5
            seen.extend(labels[:, 1:][labels[:, 1:] != -100].tolist())
            return SimpleNamespace(loss=torch.tensor(math.log(2), dtype=torch.float64))

    tokenizer = SimpleNamespace(bos_token_id=0, encode=lambda text, **kw: list(range(1, len(text) + 1)))
    monkeypatch.setattr(perplexity, "load", lambda *a: (tokenizer, Model(), "cpu"))
    result = perplexity.bits_per_char("x" * length, stride=stride)
    assert seen == list(range(1, length + 1))
    assert result["tokens"] == length
    assert result["total_bits"] == pytest.approx(length)
    assert result["bits_per_char"] == (1 if length else None)


@pytest.mark.parametrize("repaired_bits", [80, 140])
def test_spacing_difference_has_fixed_denominator(monkeypatch, tmp_path, repaired_bits):
    import cotlegibility.analyze as analysis
    seen = []

    def score(text, **kwargs):
        seen.append(text)
        total = 120 if " " not in text else repaired_bits
        return {"total_bits": total, "tokens": 10, "bits_per_char": total / len(text), "bits_per_token": total / 10}

    monkeypatch.setattr(analysis, "bits_per_char", score)
    text = "Inspectthecacheupdatelogic"
    result = measure(text, True, tmp_path, {"device": "cpu"})
    assert result["spacing_difference"] == (120 - repaired_bits) / len(text)
    measure(text, True, tmp_path, {"device": "cpu"})
    assert len(seen) == 2
    assert len(list((tmp_path / "reference").glob("*.jsonl"))) == 2


def judgment():
    return {**{d: {"grade": "none", "evidence": [], "reason": "Clear."} for d in judge.DIMENSIONS}, "context_missing": False}


def test_judge_checks_evidence_and_shapes():
    result = judgment()
    assert judge.validate(json.dumps(result), "Inspect the cache.") == result
    result["words"].update(grade="substantial", evidence=["fabricated phrase"])
    with pytest.raises(ValueError, match="exact"):
        judge.validate(json.dumps(result), "Inspect the cache.")
    result["words"]["evidence"] = ["cache"]
    assert judge.validate(json.dumps(result), "Inspect the cache.") == result
    for invalid in ("null", "[]", "{}"):
        with pytest.raises(ValueError):
            judge.validate(invalid, "text")


def test_judge_preserves_invalid_response(monkeypatch):
    response = {"text": "not JSON", "stop_reason": "completed", "raw": {"id": "r1"}}
    monkeypatch.setattr(judge, "request", lambda *a, **kw: response)
    result = judge.judge("Inspect the cache.", "reader")
    assert result["status"] == "error" and result["response"] == response


def test_sanity_transformations_are_as_described():
    cases = {}
    for row in read_jsonl("data/sanity.jsonl"):
        cases.setdefault(row["context_id"], {})[row["meta"]["variant"]] = row["text"]
    assert len(cases) == 5
    for variants in cases.values():
        assert variants["fused"] == variants["ordinary"].replace(" ", "")
        assert variants["shuffled"] == " ".join(reversed(variants["ordinary"].split()))
