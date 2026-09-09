"""Reference-language-model legibility: bits per character under a fixed, open, pinned model.

This is the middle ground between hand-written surface statistics and LLM judges: model-based,
but deterministic and fully reproducible (pin the model revision). Report bits/char rather than
perplexity so numbers are comparable across texts of different length and across tokenizers.
"""
from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_name: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    return tok, model, device


def bits_per_char(text: str, model_name: str = "gpt2", revision: str | None = None, stride: int = 512) -> dict:
    """Total negative log-likelihood of `text` under the reference LM, normalised per character.

    Uses a sliding window so long texts are scored with full context up to the model's limit.
    Returns bits/char, nats/token, and token count.
    """
    import torch

    tok, model, device = _load(model_name, revision)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    n = ids.size(1)
    nll = 0.0
    scored = 0
    prev_end = 0
    with torch.no_grad():
        for begin in range(0, n, stride):
            end = min(begin + max_len, n)
            target_len = end - prev_end
            chunk = ids[:, begin:end]
            labels = chunk.clone()
            labels[:, :-target_len] = -100
            out = model(chunk, labels=labels)
            # HF averages over the labelled tokens; multiply back to a sum.
            n_labelled = int((labels != -100).sum().item()) - 1  # first labelled token has no prediction
            n_labelled = max(n_labelled, 1)
            nll += out.loss.item() * n_labelled
            scored += n_labelled
            prev_end = end
            if end == n:
                break
    return {
        "ref_model": model_name,
        "ref_tokens": n,
        "ref_nats_per_token": nll / max(1, scored),
        "ref_bits_per_char": nll / math.log(2) / max(1, len(text)),
    }
