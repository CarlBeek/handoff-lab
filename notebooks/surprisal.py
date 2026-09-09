"""The sole metric: fixed GPT-2 surprisal in bits per Unicode character of prose."""
from __future__ import annotations

from functools import lru_cache
import math
import re

MODEL = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
VERSION = "bpc-v3-original-prose"
STRIDE = 512
# A deliberately small, fixed policy; this is not a parser or an English detector.
NONPROSE = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]+`"
    r"|(?:https?://|www\.)\S+"
    r"|(?<!\S)(?:[A-Za-z]:\\|~?/|\./|\.\./)\S+"
    r"|(?<!\S)[\w.@+-]+(?:/[^\s/]+)+\.[A-Za-z0-9]+(?=\s|$|[,;:)])"
)


def prose_only(text):
    """Replace each code/URL/path match with one space; otherwise leave prose intact."""
    return NONPROSE.sub(" ", text).strip()


@lru_cache(maxsize=2)
def load(device="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    return tokenizer, model, device


def bits_per_char(text, *, device="auto", stride=STRIDE):
    """Score every token once, including the first, using BOS and overlapping windows."""
    import torch

    if not text:
        return {"bits_per_char": None, "total_bits": 0.0, "tokens": 0, "characters": 0}
    tokenizer, model, device = load(device)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    limit = getattr(model.config, "n_positions", None) or model.config.max_position_embeddings
    if not 1 <= stride < limit:
        raise ValueError(f"stride must be between 1 and {limit - 1}")
    if tokenizer.bos_token_id is None:
        raise ValueError("The reference tokenizer must define a BOS token")
    ids = torch.tensor([[tokenizer.bos_token_id] + token_ids], device=device)
    total_bits, scored, previous_end = 0.0, 0, 0
    with torch.inference_mode():
        for begin in range(0, len(token_ids) + 1, stride):
            end = min(begin + limit, ids.shape[1])
            chunk = ids[:, begin:end]
            labels = chunk.clone()
            labels[:, :max(1, previous_end - begin)] = -100
            count = int((labels[:, 1:] != -100).sum().item())
            if count:
                loss = model(chunk, labels=labels).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Reference model returned nonfinite loss")
                total_bits += loss * count / math.log(2)
                scored += count
            previous_end = end
            if end == ids.shape[1]:
                break
    if scored != len(token_ids):
        raise AssertionError(f"Scored {scored} of {len(token_ids)} tokens")
    return {"bits_per_char": total_bits / len(text), "total_bits": total_bits,
            "tokens": scored, "characters": len(text), "device": device}
