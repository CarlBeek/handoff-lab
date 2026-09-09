"""Pinned reference-LM surprisal with correct shifted-label overlap accounting."""
from __future__ import annotations

import math
from functools import lru_cache

MODEL = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
VERSION = "surprisal-v2-bos"


@lru_cache(maxsize=2)
def load(model_name=MODEL, revision=REVISION, device="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    return tokenizer, model, device


def bits_per_char(text, model_name=MODEL, revision=REVISION, stride=512, device="auto"):
    import torch

    tokenizer, model, device = load(model_name, revision, device)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    limit = getattr(model.config, "n_positions", None) or model.config.max_position_embeddings
    if not 1 <= stride < limit:
        raise ValueError(f"stride must be between 1 and {limit - 1}")
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("The reference tokenizer must define a BOS token")
    ids = torch.tensor([[bos] + token_ids], device=device)
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
    return {"total_bits": total_bits, "tokens": scored,
            "bits_per_char": total_bits / len(text) if text else None,
            "bits_per_token": total_bits / scored if scored else None,
            "ref_model": model_name, "ref_revision": revision, "device": device,
            "stride": stride, "scorer_version": VERSION}
