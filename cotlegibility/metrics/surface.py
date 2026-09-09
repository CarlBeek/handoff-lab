"""Deterministic, tokenizer-aware surface metrics for text legibility.

Everything here is reproducible with pinned package versions: no model calls.

The metrics fall into three groups:
  * orthographic  - whitespace ratio, word lengths, glued-word share (words written without spaces)
  * lexical       - dictionary-word rate, function-word rate, symbol density
  * information   - gzip ratio, character entropy, tokens per character under a chosen BPE tokenizer,
                    and the token "inflation" caused by gluing words together
"""
from __future__ import annotations

import math
import re
import zlib
from functools import lru_cache

from wordfreq import zipf_frequency

# A deliberately small closed-class list. Telegraphic text drops exactly these.
FUNCTION_WORDS = frozenset(
    """the a an of to in for and or is are be with on that this it as by at from not if will can should
    must do does did was were has have had into than then so but no we you i they he she them their our
    your its there here when where which who what how any all each only also just may might would could
    been being about after before over under between through during without within""".split()
)

_ALPHA_RUN = re.compile(r"[A-Za-z]+")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")   # split only lower->Upper boundaries ("ownedUI" -> "owned", "UI")
_SYMBOLS = re.compile(r"[;:/|→←↔=>\\<>\[\]{}()+*&^%$#@~]")
_FENCED = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URLISH = re.compile(r"\S*(?:://|www\.)\S*|(?<!\S)(?:~?/|\./|\.\./)\S+|\S*\\\S*")   # URLs and paths, not "day/week"
SHORT_OK = frozenset("a i an in on to of at is it or as by do if no so up we be my me us".split())
MAX_SEG_WORD = 24
DICT_ZIPF = 3.0        # zipf >= 3 ~ at least 1 per million words; generous for technical vocabulary


@lru_cache(maxsize=500_000)
def zipf(word: str) -> float:
    return zipf_frequency(word, "en")


def is_dictionary_word(word: str, thresh: float = DICT_ZIPF) -> bool:
    w = word.lower()
    if len(w) == 1:
        return w in ("a", "i")
    return zipf(w) >= thresh


@lru_cache(maxsize=200_000)
def segment_run(run: str, thresh: float = DICT_ZIPF) -> tuple[str, ...] | None:
    """Segment an alphabetic run into dictionary words, maximising total log-probability.

    Returns the best segmentation, or None if the run cannot be covered by dictionary words.
    The objective sum(zipf - 9) is the log10 unigram probability, so extra words are penalised
    naturally and 'implement' beats 'im' + 'plement'.
    """
    s = run.lower()
    n = len(s)
    if n == 0:
        return None
    score = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    score[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - MAX_SEG_WORD), i):
            if score[j] == -math.inf:
                continue
            piece = s[j:i]
            if not is_dictionary_word(piece, thresh):
                continue
            sc = score[j] + (zipf(piece) - 9.0)
            if sc > score[i]:
                score[i] = sc
                back[i] = j
    if score[n] == -math.inf:
        return None
    out = []
    i = n
    while i > 0:
        out.append(run[back[i]:i])
        i = back[i]
    return tuple(reversed(out))


def _alpha_runs(text: str) -> list[str]:
    runs = []
    for m in _ALPHA_RUN.finditer(text):
        runs.extend(p for p in _CAMEL_SPLIT.split(m.group(0)) if p)
    return runs


def is_glued(run: str) -> tuple[bool, tuple[str, ...] | None]:
    """A run is 'glued' if it is not itself a dictionary word but segments into >= 2 dictionary words."""
    if len(run) < 6 or (run.isupper() and len(run) < 10) or is_dictionary_word(run) or zipf(run.lower()) >= 1.5:
        # short, a short ALL-CAPS run (acronyms, cipher keys), a common word, or a rare-but-real word -> not glued;
        # long ALL-CAPS runs go through segmentation because shouted space-less text exists (Sonnet 5 card)
        return False, None
    seg = segment_run(run)
    if seg is None or len(seg) < 2:
        return False, seg
    # Guard against chains of 2-letter fragments ("boustrophedon" -> bo+us+tr+op+he+don): short pieces must be
    # common function words and must be a minority of the segmentation.
    short = [q for q in seg if len(q) <= 2]
    if any(q.lower() not in SHORT_OK for q in short) or len(short) > len(seg) / 3:
        return False, seg
    return True, seg


def deglue(text: str) -> str:
    """Insert spaces inside glued runs, leaving everything else (casing, punctuation) unchanged."""
    def fix(m: re.Match) -> str:
        parts = [p for p in _CAMEL_SPLIT.split(m.group(0)) if p]
        fixed = []
        for p in parts:
            glued, seg = is_glued(p)
            fixed.append(" ".join(seg) if glued else p)
        return " ".join(fixed) if len(parts) > 1 and any(is_glued(p)[0] for p in parts) else "".join(fixed)
    return _ALPHA_RUN.sub(fix, text)


@lru_cache(maxsize=8)
def _encoding(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


def _non_latin_share(text: str) -> float:
    """Share of letters outside the Latin script: language mixing / stray foreign-script tokens."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(ch) > 0x024F for ch in letters) / len(letters)


def char_entropy_bits(text: str) -> float:
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values()) if n else 0.0


def prose_only(text: str) -> str:
    """Remove fenced code blocks, inline code spans and path/URL-like tokens: lexical metrics are about prose."""
    text = _FENCED.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URLISH.sub(" ", text)


def surface_metrics(text: str, tokenizer: str = "o200k_base") -> dict[str, float | int]:
    """All surface metrics for one text. `tokenizer` is a tiktoken encoding name.

    o200k_base is the GPT-4o/GPT-5-era tokenizer. Anthropic's tokenizer is not public; when a
    proxy is used the field `tokenizer` records it so the caveat travels with the number.
    Whole-text metrics (chars, tokens, gzip, entropy) use the full text; lexical metrics use prose only.
    """
    text = text or ""
    n = len(text)
    if n == 0:
        return {"chars": 0}
    prose = prose_only(text)
    ws_tokens = prose.split() or text.split()
    runs = _alpha_runs(prose)
    words_total = 0
    glued_words = 0
    glued_runs = 0
    dict_hits = 0
    func_hits = 0
    for r in runs:
        glued, seg = is_glued(r)
        if glued:
            glued_runs += 1
            glued_words += len(seg)
            words_total += len(seg)
            func_hits += sum(p.lower() in FUNCTION_WORDS for p in seg)
            dict_hits += len(seg)
        else:
            words_total += 1
            dict_hits += is_dictionary_word(r)
            func_hits += r.lower() in FUNCTION_WORDS
    enc = _encoding(tokenizer)
    n_tok = len(enc.encode(text, disallowed_special=()))
    deglued = deglue(text)
    n_tok_deglued = len(enc.encode(deglued, disallowed_special=()))
    return {
        "chars": n,
        "ws_tokens": len(ws_tokens),
        "words": words_total,
        "prose_chars": len(prose),
        "ws_ratio": sum(ch.isspace() for ch in prose) / max(1, len(prose)),
        "mean_ws_token_len": sum(map(len, ws_tokens)) / max(1, len(ws_tokens)),
        "frac_ws_tokens_gt15": sum(len(t) > 15 for t in ws_tokens) / max(1, len(ws_tokens)),
        "glued_run_share": glued_runs / max(1, len(runs)),
        "glued_word_share": glued_words / max(1, words_total),
        "dict_word_rate": dict_hits / max(1, words_total),
        "function_word_rate": func_hits / max(1, words_total),
        "symbol_density_per100": 100 * len(_SYMBOLS.findall(text)) / n,
        "non_latin_share": _non_latin_share(prose),
        "gzip_ratio": len(zlib.compress(text.encode("utf-8"))) / len(text.encode("utf-8")),
        "char_entropy_bits": char_entropy_bits(text),
        "tokens": n_tok,
        "tokens_per_100chars": 100 * n_tok / n,
        "chars_per_token": n / max(1, n_tok),
        "token_inflation_vs_deglued": n_tok / max(1, n_tok_deglued),
        "tokenizer": tokenizer,
    }
