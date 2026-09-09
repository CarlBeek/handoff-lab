import json
import os

from cotlegibility.metrics.surface import deglue, is_glued, prose_only, surface_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ASTRA = json.load(open(os.path.join(HERE, "..", "data", "exemplars", "astra_lukaspet_2026-09-08.json")))["text"]
PLAIN = ("Implement the display architecture in the UI we own, with a native/original toggle. "
         "The parent will not claim the image until it is real. Do not generate a fake official PNG.")


def test_glued_detection_positive_and_negative():
    assert is_glued("implementdisplayarchitectureinowned")[0]
    assert is_glued("untilrealimageprovided")[0]
    assert is_glued("NOTyetverified")[0]
    for w in ["boustrophedon", "notwithstanding", "abandonment", "MOVEKEY", "weekday", "cryptanalysis"]:
        assert not is_glued(w)[0], w


def test_screenshot_is_far_from_plain_prose():
    a, p = surface_metrics(ASTRA), surface_metrics(PLAIN)
    assert a["glued_word_share"] > 0.5 and p["glued_word_share"] < 0.05
    assert a["function_word_rate"] < 0.15 < p["function_word_rate"]
    assert a["ws_ratio"] < 0.05 < p["ws_ratio"]


def test_gluing_saves_characters_not_tokens():
    a = surface_metrics(ASTRA)
    d = surface_metrics(deglue(ASTRA))
    assert d["chars"] > a["chars"]
    assert 0.95 <= a["token_inflation_vs_deglued"] <= 1.05


def test_prose_only_keeps_slash_pairs_and_drops_paths():
    out = prose_only("day/week and /Users/x/y.md and https://a.b/c and `code` and ```x\ny\n```")
    assert "day/week" in out and "/Users" not in out and "https" not in out and "code" not in out
