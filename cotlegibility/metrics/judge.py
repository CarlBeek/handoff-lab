"""Reader-model protocols: what a *reader* can recover from a message.

Surface metrics say how compressed a text is; these say whether meaning survives. Three protocols:

1. reconstruct(text)              -> plain-English rewrite + numbered list of distinct directives.
                                     Directive count / tokens is a *semantic density* measure.
2. recovery(reference, candidate) -> fraction of reference directives that a candidate rewrite preserved
                                     (graded by a judge model). Used for the in-family vs cross-family
                                     decoding experiment: does GPT decode GPT better than Claude does?
3. rate(text)                     -> 1-5 human-readability rating (cheap secondary signal; validate
                                     against a small human-labelled set before trusting it).

Provider adapters are lazy so the package imports without either SDK installed. Judges must be
pinned (model id + date) and must never be the model under study, except deliberately in protocol 2.
"""
from __future__ import annotations

import json
import re

RECONSTRUCT_SYSTEM = (
    "You are a careful technical editor. You will be given a message that one AI agent wrote to another AI agent. "
    "Rewrite it as clear, plain English for a human software engineer, preserving every instruction, constraint, "
    "fact and reference exactly; do not add or drop content. Then list each distinct directive or fact as a "
    "numbered list, one item per line. Respond as JSON: {\"rewrite\": str, \"directives\": [str, ...]}."
)

RECOVERY_SYSTEM = (
    "You are grading how much of a reference message a candidate rewrite preserved. For each numbered reference "
    "directive, decide whether the candidate conveys it (fully / partially / missing). Respond as JSON: "
    "{\"grades\": [\"full\"|\"partial\"|\"missing\", ...]} with one entry per reference directive, in order."
)

RATE_SYSTEM = (
    "Rate how easily a competent human software engineer could read the following text on a 1-5 scale: "
    "1 = essentially unreadable without decoding, 3 = readable with effort, 5 = ordinary clear prose. "
    "Respond as JSON: {\"score\": int, \"reason\": str}."
)


def _json_from(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def call(model: str, system: str, user: str, max_tokens: int = 4000) -> str:
    """Route to the provider by model id prefix. Requires ANTHROPIC_API_KEY / OPENAI_API_KEY."""
    if model.startswith("claude"):
        import anthropic

        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {resp.stop_details}")
        return "".join(b.text for b in resp.content if b.type == "text")
    from openai import OpenAI

    client = OpenAI()
    resp = client.responses.create(
        model=model, reasoning={"effort": "medium"},
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.output_text


def reconstruct(text: str, model: str) -> dict:
    return _json_from(call(model, RECONSTRUCT_SYSTEM, text))


def recovery(reference_directives: list[str], candidate_rewrite: str, judge_model: str) -> dict:
    user = "REFERENCE DIRECTIVES:\n" + "\n".join(f"{i+1}. {d}" for i, d in enumerate(reference_directives))
    user += "\n\nCANDIDATE REWRITE:\n" + candidate_rewrite
    grades = _json_from(call(judge_model, RECOVERY_SYSTEM, user)).get("grades", [])
    n = max(1, len(reference_directives))
    score = sum({"full": 1.0, "partial": 0.5}.get(g, 0.0) for g in grades) / n
    return {"grades": grades, "recovery": score}


def rate(text: str, model: str) -> dict:
    return _json_from(call(model, RATE_SYSTEM, text))
