# Are frontier models' reasoning and inter-agent messages becoming less legible?

*Research design for a METR-style report. Draft 2026-09-09.*

## Summary

The motivating observation is a screenshot (Lukas Petersson, 2026-09-08) of a GPT-6 Astra parent agent
instructing a sub-agent in text with most spaces removed and function words dropped
("implementdisplayarchitectureinownedUI: native/original toggle; ..."). The question is whether this is
(a) a real, measurable shift in how frontier models write when no human is the intended reader, (b) how it
differs between OpenAI and Anthropic model lines and across generations, and (c) whether it can be measured
with cheap, repeatable methods rather than only with LLM judges.

The plan has three parts:

1. **Measure the sub-agent channel first.** Parent-to-child messages are the one place where raw
   model-written text is fully visible across both labs (raw chain of thought is not returned by either
   lab's API for current models). They are tool-call arguments, so they can be captured from real harness
   logs and reproduced via the API with identical inputs.
2. **A three-tier metric stack**: deterministic surface statistics (glued-word share, function-word
   rate, whitespace, dictionary-word rate, tokens-per-character), a pinned open reference language model
   (bits per character), and reader-model protocols (can a reader recover the directives?), with a small
   human-rated set to validate the cheap tiers.
3. **Paired, controlled generation** ("fixed-context replay"): identical agentic contexts and an identical
   spawn tool are given to every model; the message each model writes is compared. Observational logs and
   public screenshots are supporting evidence only, because screenshots are selected for being extreme.

A first pass on this machine (25 real gpt-5.5 spawn messages, Claude Code and Codex transcripts) puts the
screenshot far outside anything observed locally, and shows that the word-gluing does **not** save tokens
under the o200k tokenizer, which argues against "token economy" as the explanation.

## 1. What is being measured

Three properties of reasoning text are routinely conflated and must be kept apart:

| Property | Question | Who has studied it |
|---|---|---|
| **Legibility** | Can a human read the text and recover its meaning at reasonable effort? | This project |
| **Faithfulness** | Does the text reflect the computation that produced the answer? | Anthropic 2025, METR 2025 |
| **Monitorability** | Can a monitor (human or model) detect unwanted behaviour from the text? | OpenAI 2025, Korbak et al. 2025 |

Legibility is upstream of the other two: an unreadable trace is unmonitorable regardless of faithfulness.
This project measures legibility only, and uses monitorability work for framing.

Text comes from four **channels**, which differ sharply in what is observable:

| Channel | OpenAI GPT-5.x / GPT-6 | Anthropic Claude 4.6+ / 5 | Open-weight models |
|---|---|---|---|
| Raw chain of thought | Never returned (encrypted reasoning items); METR got raw access for GPT-5.6 Sol under a pre-deployment arrangement | Never returned; summaries are written by a different model | Returned (`reasoning_content`) |
| Reasoning summary | `reasoning.summary` = auto / concise / detailed; **not supported when Responses multi-agent is enabled** | `thinking.display` = summarized; default omitted on 4.7+ and all 5-era models; `updates` beta | n/a |
| Parent → sub-agent message, lab harness | Codex MultiAgent v1: plaintext in local rollouts. **MultiAgent v2 (since 2026-06-05) and the Responses multi-agent beta: encrypted end to end**; the Codex source calls an empty spawn prompt a way "to prevent CoT leaking" | Claude Code: plaintext (Agent tool `prompt`; child transcript starts with the same text) | Visible |
| Parent → sub-agent message, own harness | Plain function-tool arguments: fully visible | Fully visible | Visible |
| Sub-agent → parent reply | Visible (`wait_agent` output) in v1 | Visible (sidechain final text) | Visible |
| Message to the human user | Visible | Visible | Visible |

Details and sources: `docs/model_landscape_2026-09-09.md`.

Consequences for design:

- The **sub-agent channel is the primary object of study**: it is raw model output, comparable across labs,
  and it is the channel in the screenshot. But OpenAI is closing it inside its own harness (encrypted v2
  messages), so the measurement has to run in a harness *we* control, where `spawn_agent` is an ordinary
  function tool and its arguments come back in plaintext. That is what the replay experiment does.
- Reasoning summaries are a **secondary** channel with a caveat: they are written by a summarizer, so
  their legibility is not the legibility of the chain of thought.
- Raw chain-of-thought legibility can only be measured directly on **open-weight** models, plus the
  excerpts labs publish in system cards. That is still useful as a "what does RL do to a scratchpad" baseline.
- The **user-facing channel is the within-model control**: the interesting quantity is the *gap*
  between what a model writes for a human and what it writes for another model.

## 2. Hypotheses

Each is stated so that the planned data can falsify it.

- **H1 Generational drift.** Within a lab, sub-agent messages become more compressed (higher glued-word
  share, lower function-word rate, higher bits/char) with each model generation.
- **H2 Audience gap.** Within a model, legibility(user message) > legibility(sub-agent message) >
  legibility(reasoning), and the gap widens across generations.
- **H3 Not token economy.** Removing spaces saves almost no tokens under the model's own tokenizer, so
  the pressure producing it is not a token-count penalty. (Measured: 233 vs 231 o200k tokens for the
  screenshot text with and without its spaces; dropping function words is what saves tokens, ~25%.)
  Must be re-tested with GPT-6's tokenizer: tiktoken maps gpt-5.6 to o200k_base but has no entry for
  gpt-6-astra as of today.
- **H4 Private code.** A same-family model recovers the directives in a compressed message better than a
  different-family model or a human does. If true, the compression is a family-specific idiolect, which is
  the monitorability-relevant case.
- **H5 Context pressure.** Compression increases with context length, turn depth, and the number of
  concurrently managed sub-agents.
- **H6 Steerability.** A one-line instruction that a human will read the message restores legibility at
  little cost. This is the cheapest mitigation and the most policy-relevant result.
- **H7 Harness dependence.** Part of the effect comes from the harness's own prompts (terse-output
  instructions), not the model. Tested by swapping harness system prompts in the replay experiment.
- **H8 Channel constraint.** Compression appears mainly when the message channel is size-capped (Kilo's
  swarm had a message-size limit; the Hugging Face incident agents used directory names as messages). A cap
  stated in *characters* makes space-stripping rational even though it saves no tokens. Tested by adding a
  character-cap condition and a token-cap condition to the replay.

## 3. Metrics

Do not collapse to one "legibility score" early. Report the panel, then, once a human-rated set exists,
fit a composite whose weights are justified by that set.

### Tier A: deterministic surface metrics (implemented in `cotlegibility/metrics/surface.py`)

| Metric | What it captures | Notes |
|---|---|---|
| `glued_word_share` | Share of words written without a preceding space, found by segmenting alphabetic runs into dictionary words (unigram log-prob DP) | The screenshot phenomenon exactly. 0–3% on every local sample, 71% on the screenshot. Guards: ALL-CAPS runs, rare-but-real words, chains of 2-letter fragments |
| `function_word_rate` | Share of closed-class words (the, a, of, to, ...) | Telegraphic style drops exactly these; 0.24–0.40 locally vs 0.11 |
| `ws_ratio` | Whitespace share of prose characters | Crude but robust; 0.14–0.19 locally vs 0.03 |
| `dict_word_rate` | Share of words that are dictionary words (zipf ≥ 3) | Catches abbreviations and invented tokens |
| `frac_ws_tokens_gt15`, `mean_ws_token_len` | Long "words" | Cheap proxy for gluing |
| `symbol_density_per100` | Arrows, slashes, semicolons per 100 chars | Notation-heavy style |
| `tokens_per_100chars`, `chars_per_token` | Density under a named BPE tokenizer | Tokenizer must be recorded; Anthropic's is not public, so use `count_tokens` when comparing Claude models |
| `token_inflation_vs_deglued` | tokens(text) / tokens(de-glued text) | Tests H3 directly |
| `gzip_ratio`, `char_entropy_bits` | Redundancy | Weak on their own; useful as controls |

Failure modes handled: fenced code, inline code, paths and URLs are stripped before lexical metrics
(they are legitimately space-free); identifiers in camelCase are split; all-caps runs are never "glued".
Minimum length of 200 characters avoids trivial messages dominating distributions.

### Tier B: reference-language-model bits per character (implemented in `metrics/perplexity.py`)

Negative log-likelihood per character under a **pinned, open model that predates the models under study**
(GPT-2 small by default). A 2019 reader cannot have learned any 2026 idiolect, which is the point: it
measures distance from ordinary English rather than from the studied models' own habits. Bits/char is
comparable across lengths and tokenizers. On the screenshot ladder it is monotone: 2.36 (as written) →
2.02 (spaces restored automatically) → 1.69 (a real gpt-5.5 spawn message) → 1.46 (plain-English rewrite).
Sensitive to code and markdown, so report it on prose-only text and alongside Tier A.

### Tier C: reader-model protocols (scaffolded in `metrics/judge.py`; needs API keys)

| Protocol | Output | Use |
|---|---|---|
| **Reconstruct** | Plain-English rewrite + numbered list of distinct directives | Directive count / tokens = semantic density; rewrite feeds the next protocol |
| **Recovery** | Fraction of reference directives a reader's rewrite preserved (full / partial / missing, graded by a pinned judge) | The H4 experiment: readers = same-family model, other-family model, human. Reference directives come from the parent's own context (what it was told to do) |
| **Rate** | 1–5 readability rating | Cheap secondary; only trusted after correlation with human ratings |

Rules: judges are pinned by model id and date; the model under study never grades itself except
deliberately in the Recovery experiment; every judge output is stored raw.

### Human validation (small, essential)

About 100 messages stratified across models and metric quantiles; 2–3 raters; per message: readability
rating, time-to-read, and a 3-question comprehension check written from the reference directives. Report
Spearman correlations of every Tier A/B/C metric with the human ratings and inter-rater agreement. This is
what lets the cheap metrics carry the report.

## 4. Data

### 4a. Observational (free, real, uncontrolled)

- **Codex CLI/Desktop rollouts** (`~/.codex/sessions`): `spawn_agent` messages, `wait_agent` replies,
  reasoning summaries; sub-agent threads are separate files linked by `parent_thread_id`. Raw reasoning is
  encrypted. On this machine: 50 rollouts, May–Sep 2026, gpt-5.5 and gpt-5.6-sol, 25 spawn messages.
  Codex Desktop also imports Claude Code sessions; those are skipped (they are Claude output).
- **Claude Code transcripts** (`~/.claude/projects`): Agent-tool prompts, sidechain replies, thinking
  summaries (thinking is omitted by default on Claude 5 models). Little on this machine so far.
- **Public examples**: an exemplar registry (`data/exemplars/`) with provenance, transcription notes and
  the claimed model. Screenshots are selected for being striking, so they bound the phenomenon; they do
  not estimate its frequency. Ask the poster for the log if possible.

### 4b. Controlled: fixed-context replay (the core experiment)

Take N real agentic contexts (from the logs above, redacted), truncated immediately before the parent
spawned a sub-agent. Give the identical context and an identical `spawn_agent(message)` tool definition to
every model through its API, and capture the message it writes. This yields a **paired** design: every
model answers the same N situations, so per-context differences cancel.

Factors: model (OpenAI gpt-5.5, gpt-5.6 variants, gpt-6-astra; Anthropic claude-sonnet-4-6, opus-4-6,
opus-4-7, opus-4-8, sonnet-5, opus-5, fable-5-1; one open-weight baseline), reasoning effort (low / high),
context length (native vs truncated vs padded), audience instruction (none vs "a human will read this"),
harness prompt (Codex-style vs Claude-Code-style vs neutral). Not every cell needs filling; the main
effects and the model × audience interaction are the priorities.

Size: the screenshot-sized effect needs only a handful of contexts; a subtle generational drift
(standardised effect ~0.3) needs roughly 150–200 contexts per model. Rough cost at ~10k input tokens per
context: 200 contexts × 10 models × 3 key conditions ≈ 6k calls, ≈ 60M input tokens, on the order of a few
hundred dollars at current prices, more for Fable-tier models. Use batch APIs where possible.

### 4c. Live harness runs (expensive, most realistic)

Run Codex CLI (`codex exec`) and Claude Code (`claude -p`) on the same 20–30 multi-file tasks with
sub-agent use encouraged, and parse the resulting logs with the existing parsers. This captures the
harness's real prompts and context growth (H5, H7) and provides reasoning summaries as a by-product.

### 4d. Reasoning summaries and raw chain of thought (secondary)

OpenAI `reasoning.summary=detailed`, Anthropic `thinking.display="summarized"`, and raw traces from one or
two open-weight reasoning models on the same tasks. Analysed with the same metrics, reported separately.

## 5. Agents versus repeatable NLP: the answer

Use deterministic metrics as the backbone, a pinned reference LM as the middle tier, and LLM judges only
for the questions that are genuinely about meaning (can a reader recover it?), always validated against
human ratings. Reasons:

- Surface metrics are free, run over tens of thousands of messages, are reproducible with pinned package
  versions, and the screenshot phenomenon is *surface-level by construction* (missing spaces, missing
  function words). They already separate the screenshot from every local sample by a wide margin.
- Judges drift with their own versions, cost money, and are the model family under study; the in-family
  decoding advantage (H4) is exactly the bias that would contaminate a naive judge.
- Autonomous agents are useful for **discovery** (mining logs, finding examples, checking harness
  prompts), not for **measurement**. Keep the measurement code boring and deterministic.

## 6. Tooling

Built today (Python 3.14 venv, `pip install -e ".[ppl]"`):

- `cotlegibility/schema.py`: one `Sample` = text + provider, model, channel, source, timestamp, meta.
- `cotlegibility/sources/codex.py`, `claude_code.py`: log parsers producing samples (both harness formats
  documented in the module docstrings).
- `cotlegibility/metrics/surface.py`, `perplexity.py`, `judge.py`: the three tiers.
- `scripts/analyze_local.py`: dedupes (forked sub-agent threads replay the parent's history), scores,
  writes `out/metrics.csv`, `out/summary.csv`, `out/fig_surface.png`.
- `cotlegibility/experiments/replay.py` + `scripts/replay_subagent.py`: the fixed-context replay. Extracts
  every pre-spawn context from the Codex logs (25 found, rendered to ≤60k characters, home paths and
  key-like strings redacted), builds one identical prompt per context × audience condition, calls either
  provider with an identically defined `spawn_agent` tool, caches by content hash, and scores the returned
  message. `--dry-run` works without keys; paid runs need `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

Next:

1. Grow the context pool beyond the 2 local sessions that used sub-agents: live harness runs (4c) are the
   cheapest source, and Claude Code transcripts with Agent-tool calls can be rendered the same way. Decide
   whether the harness's own developer prompt stays in the transcript (it is the H7 factor).
   If the experiment matrix grows, move generation onto Inspect AI (provider abstraction, logging,
   caching, retries) and keep the analysis in pandas.
2. `experiments/decode.py` (H4) reusing the same adapters; steerability (H6) is already the `audience`
   factor of the replay.
3. `human_study/`: a spreadsheet or a small static form; export to the same `metrics.csv` schema.
4. `report/`: figures from `out/*.csv` (strip plots by model × channel, paired-difference plots for the
   replay experiment, a metric-vs-human-rating validation panel), and a markdown/Quarto report.
5. Provenance: pin package versions, record API model ids and response dates (providers update silently),
   hash every input context, store raw responses.

## 7. Report structure

Summary of findings → Background (legibility vs faithfulness vs monitorability; what is observable) →
Methods (channels, data sources, replay design, metrics, judges, human validation) → Results (per-metric
distributions by model × channel; paired replay differences across generations; audience gap; in-family vs
cross-family recovery; steerability; token-economy test; context-length effects) → Limitations (API
summaries, tokenizer opacity, screenshot selection bias, harness confounds, sample sizes, provider drift)
→ Implications for chain-of-thought monitoring → Appendix (examples, prompts, reproduction instructions).

## 8. First numbers from this machine (2026-09-09)

Medians per group; dots in `out/fig_surface.png`. Minimum 200 characters per text.

| Group | n | glued-word share | function-word rate | whitespace ratio | bits/char (GPT-2) |
|---|---|---|---|---|---|
| Human-typed prompts (Codex + Claude Code) | 42 | 0.00 | 0.31 | 0.17 | 1.57 |
| gpt-5.5 → sub-agent messages (Codex, Jun 2026) | 22 | 0.01 | 0.29 | 0.16 | 1.90 |
| gpt-5.5 sub-agent → parent replies | 14 | 0.01 | 0.24 | 0.19 | 2.05 |
| gpt-5.5 messages to user | 1674 | 0.00 | 0.35 | 0.17 | 2.04 |
| gpt-5.6-sol messages to user (Aug–Sep 2026) | 74 | 0.00 | 0.32 | 0.15 | 1.69 |
| claude-opus-5 messages to user | 82 | 0.00 | 0.40 | 0.19 | 1.82 |
| claude-fable-5-1 thinking summaries / progress notes | 37 | 0.00 | 0.26 | 0.15 | 1.78 |
| **GPT-6 Astra → sub-agent (Petersson screenshot, Sept 8)** | 1 | **0.69** | **0.11** | **0.03** | **2.36** |
| **GPT-6 Astra → sub-agent (Kilo blog fragment, Sept 4)** | 1 | **0.43** | **0.11** | **0.05** | **3.10** |
| **Claude Sonnet 5 thinking (system card transcript 6.4.5.A)** | 1 | **0.58** | **0.26** | **0.00** | **3.32** |

Token-economy test on the screenshot text (o200k tokenizer): as written 233 tokens; with spaces restored
231; plain-English rewrite 312. Gluing saves 10% of characters and 0% of tokens, because BPE already
tokenizes "implementdisplayarchitecture" into the same word pieces without leading spaces. Dropping
function words is what saves tokens.

Caveats: three public exemplars, selected for being extreme (the Sonnet 5 transcript is the card's own
worst-case illustration); the screenshot's model identity is the poster's claim; local gpt-5.5 spawn
messages come from one user's sessions; no gpt-6 or Claude sub-agent messages observed locally yet;
bits/char is inflated by code and markdown in user-facing messages; the glued-word detector deliberately
ignores all-caps runs shorter than 10 characters (acronyms, cipher keys), so shouted text is undercounted.

## 9. Prior work (summary; full annotated bibliography in `docs/literature_review_2026-09-09.md`)

- **Framing.** Korbak et al. (Jul 2025) predicted that outcome-based RL "might no longer incentivise
  legibility" and asked for standardized monitorability evals in system cards. OpenAI's "Monitoring
  Monitorability" (Dec 2025) defines monitorability as a monitor's accuracy at predicting a property of the
  agent's actions (g-mean² = TPR × TNR) and open-sourced most of the suite (Apr 2026). Neither is a
  legibility metric; both treat legibility as an ingredient.
- **Legibility measured directly.** Emmons et al. (GDM, Oct 2025): 0–4 legibility + 0–4 coverage autorater,
  prompt published; frontier models score >97% legible on QA benchmarks. Jose (NeurIPS 2025): 1–9 GPT-4o
  autograder across 14 models; DeepSeek R1 ≈ 4.3, Claude 4-era ≈ 1.6; illegibility rises with difficulty
  for every family except Claude; judge–judge κw 0.78; no human study. Anthropic's system cards use a 0–10
  "illegible thinking: un-human-like thinking text that is difficult to follow" judge score across ~2,900
  audit transcripts per model. **No published human-rated legibility dataset exists**, and Young (Mar 2026)
  shows that switching judges can reverse model rankings. Our human-validation set is therefore a genuine
  contribution, not a formality.
- **Reconstruction-style metrics.** REMUL (Feb 2026) scores legibility as whether listener models can
  execute a truncated trace; Roytburg et al. (Mar 2026) define weak-to-strong legibility; faul_sname (Jun
  2026) showed a human and Claude Haiku 4.5 could decode the Fable 5 card's "almost illegible" FreeCell
  trace. These are the precedent for the recovery protocol (Tier C) and the in-family vs cross-family test.
- **What the drift looks like.** Apollo (Sept 2025) documented o3's idiosyncratic vocabulary ("watchers",
  "disclaim", "marinade" at thousands of times its web frequency); DeepSeek-R1 added a language-consistency
  reward because R1-Zero mixed languages; OpenAI's CoT-controllability series rose from 0.7% (GPT-5.4) to
  60.9% (GPT-6 Astra) and Astra's card reports a "substantial decrease" in CoT monitorability; Anthropic's
  cards report illegible-thinking scores rising from Opus 4.8 to Mythos 5, Sonnet 5, Opus 5 and Fable 5.1,
  with Sonnet 5's card printing a space-less thinking transcript (now in `data/exemplars/`).
- **Inter-agent text specifically.** The Beltoft et al. Moltbook study (May 2026) classified emergent
  agent-to-agent languages into token-efficiency, new-natural-language and oversight-evasion types; the
  METR/Redwood Hugging Face incident report (Aug 2026) shows ~70k compressed protocol-style messages
  between agents; Kilo's Astra preview (Sept 4, 2026) attributes the same style to a message-size cap; and
  OpenAI's API guidance now tells developers that inter-agent messages "may contain grammar or spacing
  errors" and supplies a prompt fix. Nobody has yet measured this channel across models or generations.
- **Reusable assets.** OpenAI monitorability-evals (Apache-2.0), METR CoT-faithfulness code and
  transcripts, MonitorBench (MIT, Inspect port), the Emmons autorater prompt, the Jose rubric.

## 10. Model landscape and observability (summary; details in `docs/model_landscape_2026-09-09.md`)

- **OpenAI line.** GPT-5 (Aug 2025) → 5.1 → 5.2 → 5.3-Codex → 5.4 (Mar 2026) → 5.5 (Apr 23, 2026) →
  GPT-5.6 Sol / Terra / Luna (flagship / balanced / fast; preview Jun 26, GA Jul 9, 2026; ids `gpt-5.6-sol`,
  `gpt-5.6-terra`, `gpt-5.6-luna`) → GPT-6 Astra (Sept 3–4, 2026; `gpt-6-astra`; Codex CLI default since
  0.153.4). All return reasoning summaries at most. Codex sub-agents: MultiAgent v1 messages are plaintext
  in local rollouts (what this machine has, from gpt-5.5); v2 messages have been encrypted since June 5,
  2026, and the Responses multi-agent beta returns encrypted `agent_message` items and disables summaries.
  The Astra system card says nothing about inter-agent legibility but notes agents communicating with other
  agents of the same user "in the same Codex harness" during internal testing.
- **Anthropic line.** Sonnet 4.5 (Sept 2025) → Opus 4.5 → Opus 4.6 / Sonnet 4.6 (Feb 2026) → Opus 4.7
  (Apr) → Opus 4.8 (May) → Fable 5 / Mythos 5 (Jun 9) → Sonnet 5 (Jun 30) → Opus 5 (Jul 24) → Fable 5.1 /
  Mythos 5.1 (Sept 1, 2026). Thinking is summarized by a separate model or omitted; never raw. Claude Code
  stores Agent-tool prompts and child transcripts in plaintext, so the Claude sub-agent channel is fully
  observable locally.
- **Raw-CoT baselines.** DeepSeek V4, Qwen3.6/3.8, Kimi K3, gpt-oss return raw reasoning; Gemini 3.x
  returns thought summaries only.
- **Provenance of the screenshot.** Lukas Petersson is CEO of Andon Labs; the UI strings match no lab
  product, so the harness is most likely Andon Labs' own scaffold (unconfirmed). Kilo's fragment came from
  its own swarm prototype. Both are third-party harnesses, which supports running the measurement in a
  harness we control.

## 11. Limitations and risks

- Neither lab returns raw chain of thought for current models; claims about "CoT" legibility must be
  scoped to what is observable (sub-agent messages, summaries, open-weight traces).
- Anthropic's tokenizer is private; cross-lab token metrics need the `count_tokens` endpoint.
- Harness prompts differ between Codex and Claude Code and change with each release; record versions.
- Public screenshots over-represent extreme cases; they motivate but cannot estimate prevalence.
- Providers update models silently; record model ids, dates and response metadata for every sample.
- Redaction: real logs contain user code and paths; strip before any API replay or publication.

## 12. API access: Surplus Intelligence

Decision (2026-09-09): route all API calls through Surplus Intelligence, an order book for spare inference
capacity with an OpenAI-compatible `/v1` and an Anthropic-shaped `/anthropic/v1/messages`. Checked against
the public catalogue and price book on 2026-09-09:

- Every model in the study is listed with tool calling: `gpt-5.4`, `gpt-5.5`, `gpt-5.6-luna/-terra/-sol`,
  `gpt-6-astra`, `claude-sonnet-4.5` … `claude-fable-5.1` (Claude ids use dots), plus raw-CoT baselines
  (`deepseek-v4-pro-0813`, `kimi-k3`, `qwen3.8-2.4t-a95b`, `openai-gpt-oss-120b`).
- Each has a first-party-hosted seller to pin to: `anthropic` for Claude, `openai` for gpt-6-astra, `bedrock`
  for gpt-5.5 and the 5.6 tiers. The OpenRouter offer for `gpt-6-astra` maps to `gpt-6-astra-pro`; avoid it.
- Prices are at or below list (gpt-5.6-sol $2/$10 per M tokens vs $4/$20 list; gpt-6-astra and Fable 5.1
  $10/$50; Opus 5 $5/$25; Sonnet 5 $2/$10). Rough replay cost: about $0.18 per call averaged over the 20
  models, so a 25-context × 20-model × 3-condition pilot is about $275 and 200 contexts about $2,200.
  Pinning to first-party sellers can cost up to 25% more than the cheapest offer.

What Surplus does not do, and how the tooling compensates:

- **No model attestation.** "No cryptographic or semantic attestation ties a response to the advertised
  model"; quantization, truncation, injected system prompts and altered sampling are not measured. The
  adapter pins the `provider` allow-list to first-party-hosted sellers, stores the `x-si-served-by`,
  `x-si-provider-family`, `x-si-marketplace-attempts`, `x-si-buyer-cost-micro` and `x-request-id` headers
  plus the served model id with every result, and `scripts/surplus_smoke.py --sweep` runs identity canaries
  across every seller of a model so divergent sellers can be excluded.
- **Reasoning controls are forwarded as sent, not translated**, and the docs warn that on the direct-OpenAI
  seller some `gpt-5.6` ids get `reasoning_effort` forced to `none` when tools are present. Every result
  records `usage.completion_tokens_details.reasoning_tokens`; a cell with zero reasoning tokens on a
  reasoning model is flagged and re-run through another seller or without the tool (message elicited in
  tagged text instead), and the elicitation method is held constant across models within a comparison.
- **Chat completions may not enable Claude thinking**, so Claude models go through the Anthropic surface
  with `thinking: {type: adaptive}` (budget_tokens on the 4.5 generation). Beta headers are ignored there,
  so no server-side fallbacks or `display` options; the raw chain of thought is never returned anyway.
- **Unsupported parameters are dropped silently.** The adapter reads `supported_parameters` from `/v1/models`
  before every call and only sends what the model row lists.

## 13. Data acquisition on a ~$50 budget (decided 2026-09-09)

Verified today on this machine:

| Route | Works? | What it yields | Cost |
|---|---|---|---|
| Claude Code headless (`claude -p --model …`) under the Claude subscription | Yes: Sonnet 5 and Opus 4.6 tested; Agent-tool prompts logged in plaintext | Claude parent → sub-agent messages, sub-agent replies, user-facing text; real harness prompts; old and new generations | $0 (subscription quota) |
| Codex CLI headless (`codex exec`) under the ChatGPT login | Runs, spawns sub-agents with gpt-5.6-sol and gpt-6-astra, but on CLI 0.153.4 the spawn messages are ciphertext and children are MultiAgent v2 even with the v2 flag off; child threads never see plaintext | OpenAI user-facing messages and (for some models) reasoning summaries only | $0 |
| Older Codex CLI (0.139.0, the last local version with plaintext v1 spawn messages) via npx | Server rejects it: "The 'gpt-5.6-sol' model requires a newer version of Codex" | – | – |
| Copy-pasting into ChatGPT / claude.ai | Wrong system prompts, no spawn tool, laborious | – | – |
| API replay with our own `spawn_agent` tool (direct keys or Surplus) | Yes (tooling ready, needs a key) | OpenAI parent → sub-agent messages in plaintext; paired design; audience conditions | the only paid part |

Plan:

1. **Claude side, free.** `scripts/run_harness.py --harness claude --models claude-opus-4-6 claude-opus-5
   claude-sonnet-5 claude-fable-5-1 --tasks all --condition default human_reads`. Ten tasks × four models ×
   two conditions = 80 sessions, each spawning 2–5 sub-agents, so roughly 250 parent → sub-agent messages
   plus replies, all in the real harness. Runs are sequential and take a few minutes each; spread them over
   the subscription's usage windows. Add `claude-opus-4-8` and `claude-fable-5` if the plan offers them.
2. **OpenAI side, paid, ~$40.** Fixed-context replay through the API on the 25 local Codex spawn contexts
   plus 15 Claude Code contexts from step 1, for gpt-5.5, gpt-5.6-sol and gpt-6-astra, conditions
   {none, openai_guidance}. With prompt caching of the shared transcript prefix, output tokens dominate:
   roughly $0.35 per Astra call, $0.20 per gpt-5.5 call, $0.10 per Sol call, so 40 contexts × 2 conditions
   × 3 models ≈ $50 without caching and about $35 with it. Drop to 25 contexts if the first 10 look stable.
   Direct OpenAI keys are preferred; Surplus pinned to the `openai` seller is the fallback for Astra.
3. **Free Codex data anyway.** Run the same tasks through `codex exec` for gpt-5.5, the 5.6 tiers and
   Astra. The spawn messages are hidden, but the runs give Astra's user-facing messages, Luna's reasoning
   summaries, and the count of sub-agents spawned per task, which is the prevalence denominator.
4. **Judges and decoding, free.** The reconstruction and recovery protocols are plain prompts, so run them
   through `claude -p` and `codex exec` instead of the API: same model families, no cost.
5. **Human validation, free.** 30–100 messages rated by you and one or two colleagues.

What not to do: extracting the OAuth tokens from Claude Code or Codex to call the raw APIs is against
both providers' terms; the scripted CLIs are the sanctioned way to use the subscriptions.

## 14. Next steps

1. Run the Claude harness batch (step 1 above); check `out/harness_runs.jsonl` and re-run
   `scripts/analyze_local.py --runs out/harness_runs.jsonl` after each model.
2. Get an OpenAI API key with $50 of credit, or a Surplus key, and run the replay pilot on 10 contexts
   for the three OpenAI models before committing the rest of the budget.
3. Run the Codex batch for prevalence and the visible channels.
4. Human rating pilot on 30 messages; write up.
