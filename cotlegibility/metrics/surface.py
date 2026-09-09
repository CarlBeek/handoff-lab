"""Small, inspectable spacing diagnostics. All offsets refer to the supplied string."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from wordfreq import zipf_frequency

VERSION = "spacing-v2"
ALPHA = re.compile(r"[A-Za-z]+")
CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
SHORT = set("a i an in on to of at is it or as by do if no so up we be my me us".split())
# Mask instead of deleting: passage/source offsets remain meaningful. Slash pairs
# such as day/week survive; relative file paths and code do not enter prose scores.
NONPROSE = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]+`"
    r"|(?:https?://|www\.)\S+"
    r"|(?<!\S)(?:[A-Za-z]:\\|~?/|\./|\.\./)\S+"
    r"|(?<!\S)[\w.@+-]+(?:/[^\s/]+)+\.[A-Za-z0-9]+(?=\s|$|[,;:)])"
)


@lru_cache(maxsize=200_000)
def frequency(word):
    return zipf_frequency(word.lower(), "en")


@lru_cache(maxsize=50_000)
def segment_run(run):
    n = len(run)
    scores, back = [-math.inf] * (n + 1), [0] * (n + 1)
    scores[0] = 0
    for end in range(1, n + 1):
        for start in range(max(0, end - 24), end):
            word = run[start:end].lower()
            if (len(word) == 1 and word not in {"a", "i"}) or frequency(word) < 3:
                continue
            score = scores[start] + frequency(word) - 9
            if score > scores[end]:
                scores[end], back[end] = score, start
    if not math.isfinite(scores[n]):
        return None
    parts = []
    end = n
    while end:
        parts.append(run[back[end]:end])
        end = back[end]
    return tuple(reversed(parts))


def is_glued(run):
    if len(run) < 6 or (run.isupper() and len(run) < 10) or frequency(run) >= 1.5:
        return False, None
    parts = segment_run(run)
    if not parts or len(parts) < 2:
        return False, parts
    short = [p for p in parts if len(p) <= 2]
    good = all(p.lower() in SHORT for p in short) and len(short) <= len(parts) / 3
    return good, parts


def prose_only(text):
    return NONPROSE.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), text)


def spacing(text):
    """Restore dictionary-segmentable runs and retain the exact edited spans.

    CamelCase is counted separately: identifiers and prose cannot be reliably distinguished.
    """
    spans, words, glued_words, camel_boundaries = [], 0, 0, 0
    for match in ALPHA.finditer(text):
        parts = CAMEL.split(match[0])
        camel_boundaries += len(parts) - 1
        start = match.start()
        for part in parts:
            glued, segments = is_glued(part)
            count = len(segments) if glued else 1
            words += count
            if glued:
                glued_words += count
                spans.append({"start": start, "end": start + len(part), "original": part,
                              "restored": " ".join(segments)})
            start += len(part)
    restored, end = [], 0
    for span in spans:
        restored.extend([text[end:span["start"]], span["restored"]])
        end = span["end"]
    restored.append(text[end:])
    return {"restored": "".join(restored), "spacing_spans": spans,
            "glued_word_share": glued_words / words if words else None,
            "camel_boundaries": camel_boundaries, "words": words}


def deglue(text):
    return spacing(text)["restored"]


def passages(text, target=600):
    """Contiguous passages, preferring whitespace; never split a long fused run."""
    if target < 50:
        raise ValueError("Passage size must be at least 50 characters")
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            split = max(text.rfind(" ", start + target // 2, end), text.rfind("\n", start + target // 2, end))
            if split >= 0:
                end = split + 1
            else:
                following = re.search(r"\s", text[end:])
                end = end + following.end() if following else len(text)
        yield start, end, text[start:end]
        start = end
