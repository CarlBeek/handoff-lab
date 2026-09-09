import math
from types import SimpleNamespace

import pytest
from notebooks import surprisal


def test_prose_policy_preserves_unrepaired_text():
    text = "Inspectthecache day/week `private_id` ./src/file.py https://example.com\n```py\nx=1\n```"
    assert surprisal.prose_only(text) == "Inspectthecache day/week"
    assert surprisal.prose_only(" pharmacokinetics  FooBar ") == "pharmacokinetics  FooBar"
    assert surprisal.prose_only("```py\nunterminated") == ""
    assert surprisal.prose_only("`x`") == ""


@pytest.mark.parametrize("length,stride", [(0, 2), (1, 2), (4, 2), (5, 2), (17, 1), (17, 2), (17, 4)])
def test_every_shifted_token_scored_once(monkeypatch, length, stride):
    import torch
    seen = []
    class Model:
        config = SimpleNamespace(n_positions=5)
        def __call__(self, chunk, labels):
            assert chunk.shape[1] <= 5
            seen.extend(labels[:, 1:][labels[:, 1:] != -100].tolist())
            return SimpleNamespace(loss=torch.tensor(math.log(2), dtype=torch.float64))
    tokenizer = SimpleNamespace(bos_token_id=0, encode=lambda text, **kw: list(range(1, len(text) + 1)))
    monkeypatch.setattr(surprisal, "load", lambda *a: (tokenizer, Model(), "cpu"))
    result = surprisal.bits_per_char("x" * length, stride=stride)
    assert seen == list(range(1, length + 1))
    assert result["tokens"] == length
    assert result["total_bits"] == pytest.approx(length)
    assert result["bits_per_char"] == (1 if length else None)


def test_denominator_is_characters_not_bytes_or_tokens(monkeypatch):
    import torch
    tokenizer = SimpleNamespace(bos_token_id=0, encode=lambda text, **kw: [1, 2, 3])
    class Model:
        config = SimpleNamespace(n_positions=5)
        def __call__(self, chunk, labels):
            return SimpleNamespace(loss=torch.tensor(math.log(2), dtype=torch.float64))
    monkeypatch.setattr(surprisal, "load", lambda *a: (tokenizer, Model(), "cpu"))
    result = surprisal.bits_per_char("é🙂", stride=2)
    assert result["bits_per_char"] == pytest.approx(1.5)
    assert result["characters"] == 2 and result["tokens"] == 3
    for stride in (0, 5):
        with pytest.raises(ValueError, match="stride"):
            surprisal.bits_per_char("x", stride=stride)
