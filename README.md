# Handoff Lab

Measuring agent-handoff surprisal across model generations.

A small, controlled study of whether newer models write less predictable
agent-to-agent messages. **One metric: pinned GPT-2 bits per character of prose.**
One Matplotlib graph: model-family release date versus mean surprisal, with 95%
task-bootstrap intervals. Higher is not automatically less intelligible.

## The project

- `scripts/prepare_tasks.py` freezes five pilot and 50 disjoint main-study tasks.
- `scripts/collect.py` collects one outgoing handoff per task/model, without running a child.
- `scripts/surplus.py` handles public quotes, pinned routing, and response checks.
- `notebooks/analysis.ipynb` reads raw responses, scores prose locally, and draws the graph.
- `notebooks/surprisal.py` is the small, tested reference scorer.
- `data/models.json` holds the model panel, native generation settings, approved routes, and sourced release dates.
- `data/task_manifest.json` records the pinned dataset, sampling seed, task IDs, and context hashes.
- `tests/` covers preparation, safe collection, scoring, bootstrap, and notebook execution.

No agent framework, judge, spacing repair, multi-metric report, or benchmark
execution harness. Collection and analysis are independent.

## Install

From the repository root (Python 3.11+; collector supports macOS/Linux):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 1. Prepare tasks — no paid API calls

```sh
python scripts/prepare_tasks.py
```

Downloads only the pinned test Parquet from
[SWE-bench Lite BM25 13K](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_bm25_13K).
It uses issue descriptions and already-retrieved code; no repository cloning or Docker.

Tasks are sampled without replacement in shuffled repository rounds. Small
repositories can exhaust, so balance is approximate. This is a deliberate
cross-repository sample, not prevalence-weighted software usage. The script:

- reads only task identity, source revision, issue, and formatted context columns;
- extracts the code section, removing patch-writing instructions/example patches;
- never includes solution/test patches, grading labels, or discussion hints;
- preserves the issue and code without an LLM rewrite;
- saves exact packets to `data/contexts.jsonl` and the selection to `data/task_manifest.json`.

Identical reruns are safe. Different selections require another `--out` directory;
existing frozen data is not overwritten. Generated context packets are ignored by Git
and can be reproduced from the pinned source; the small selection manifest is tracked.
The upstream retrieval budget is 13,000 reference-tokenizer **code** tokens;
issue text adds more. Do not budget these as 2,000-token prompts.

## 2. Preview, then explicitly collect

```sh
python scripts/collect.py
```

Default: **offline** preview of five pilot tasks × eight models = 40 requests.
It needs no API key and writes nothing. To check availability and rough prices,
read the public Surplus order books (still no paid requests):

```sh
python scripts/collect.py --quote
python scripts/collect.py --model claude-fable-5.1 --quote
```

Collection requires setting `SURPLUS_API_KEY` in your environment. Start with one
request per model, inspect the saved responses, then finish the five-task pilot:

```sh
python scripts/collect.py --execute --max-calls 8
python scripts/collect.py --execute --max-calls 32
```

All paid requests go to **Surplus**, using its Responses endpoint for OpenAI and
Messages endpoint for Anthropic. There are no direct-provider credentials or calls.
The collector first prefers healthy, trusted offers whose host is the model
developer's API (`api.openai.com` or `api.anthropic.com`), then `openrouter.ai`.
These are explicit local approvals in `scripts/surplus.py`, not arbitrary offers
bearing a marketplace “trusted” label.

One provider is selected and pinned per model run. Resuming keeps that provider
even if a cheaper/preferred one appears. If it becomes unavailable, collection
stops; there is no runtime provider/model fallback or automatic retry. This pins
the provider family, **not an individual seller, price, or immutable model**.
See [Surplus routing controls](https://www.surplusintelligence.ai/docs/marketplace/routing-controls).

The panel is GPT-5.5, GPT-5.6 Sol, GPT-6 Astra; Claude Opus 4.6, 4.8, and 5;
Claude Fable 5 and 5.1. No Sonnet or 4.5-class models. Each requests
medium effort and an 8,192-token output cap. OpenAI uses `reasoning.effort`;
Anthropic uses adaptive thinking and `output_config.effort`. Equal effort labels
do not imply equal compute. Native API shapes follow
[OpenAI's model guide](https://developers.openai.com/api/docs/guides/latest-model) and
[Anthropic's effort guide](https://platform.claude.com/docs/en/build-with-claude/effort).

Both APIs receive the same instruction, evidence, and strict `spawn_agent(message)`
schema. Tool choice is **automatic**, with an instruction to delegate exactly once:
[Fable 5.1 does not support forced tool choice](https://platform.claude.com/docs/en/models/fable-5-1/overview).
Missing/multiple/malformed calls are failures, not retried. The worker is described
as receiving the code excerpts and handoff, but not the issue or parent conversation.
The child is never executed. There are no brevity, shorthand, or readability instructions.

`--max-calls` bounds **new calls in this invocation**; per-request output caps
bound output tokens. These are **not a dollar ceiling**: input tokens also cost
money, and hidden reasoning can consume the output budget. Quotes estimate input
tokens as characters/4 and assume the full output cap; tokenization, cache, fees,
and changing prices can change the bill. The quoted total covers all pending
target tasks, not just the next `--max-calls` batch.

Optional `--stop-after-usd 2` stops once **known new spend in this invocation**
reaches $2. It checks after each response and can overshoot by one request; it is
not a hard budget. Missing cost metadata stops collection. Inspect actual pilot
usage and failures before scaling up.

The shared study freezes tasks and protocol; each model freezes its own configuration,
route, and complete request plan. Adding a model does not change existing plans:

```text
data/raw/main/
  study.json
  gpt-6-astra/
    run.json
    request-<hash>.json
  claude-fable-5.1/
    run.json
    request-<hash>.json
```

Each attempt is written before sending and finalized afterward, preserving the
exact request, full response, timestamps, request IDs, usage, returned model ID,
Surplus routing/adaptation/cost headers, and failures. Completed records are not
overwritten. An OS lock prevents concurrent collectors in one study directory.

Reruns skip **every** existing attempt, including errors and interrupted/uncertain
calls. An API error, unexpected provider/model, snapshot change, gateway truncation,
or reported generation-parameter adaptation stops the run. Inspect it before
resuming; uncertain calls may have been billed. Do not delete records to force retries.

**Important limit:** these checks verify exposed metadata, not the actual model
weights or full outbound request. Surplus documents that native-looking requests
can be rebuilt and unsupported fields silently dropped; adaptation headers are
not exhaustive. In particular, do not assume Anthropic's requested effort survived
just because it appears in our saved input. OpenRouter is another intermediary.
A pilot is needed to assess live compatibility, and even a successful pilot does
not prove effective settings. See
[parameter compatibility](https://www.surplusintelligence.ai/docs/reference/parameter-compatibility).

## 3. Main study, more tasks, or another model

The pilot checks the pipeline and handoff quality; its five tasks are disjoint
from the main study. After inspecting it:

```sh
python scripts/collect.py --split main --quote
python scripts/collect.py --split main --execute --max-calls 200
```

The default main target is **25 tasks per model**, drawn from a frozen ordered
set of 50. Later, extend to 50 using the **same directory**:

```sh
python scripts/collect.py --split main --tasks 50 --quote
python scripts/collect.py --split main --tasks 50 --execute --max-calls 200
```

`--tasks` is a cumulative target, not an additional request count. Existing 25-task
attempts are skipped; only the next 25 are added. This adds different tasks, not
repeated generations of the old tasks. Failed attempts still count as attempted.

To add a model, copy the closest entry in `data/models.json` and update its local
`id`, Surplus market `model`, `label`, `vendor`, `release_date`, `release_source`,
and explicitly accepted `response_models`. Keep the appropriate native `api`, `parameters`, and
ordered `providers`. No Python change is needed for compatible Responses or
adaptive-thinking Messages models. A different API needs a small adapter; a new
trusted provider needs explicit approval in `PROVIDER_HOSTS`.

Preview and pilot only that new catalog ID, then collect its matched main tasks:

```sh
python scripts/collect.py --model NEW-ID --quote
python scripts/collect.py --model NEW-ID --execute --max-calls 5
python scripts/collect.py --model NEW-ID --split main --tasks 25 --execute --max-calls 25
```

Repeat `--model` for a subset; omit it for the whole catalog. Previous results are
untouched. Use `--tasks 50` for a new model if the existing panel already has 50.
Changing a saved model's configuration requires a new local `id` or `--out`
directory; changing tasks/protocol requires a new study directory. Returned model
aliases and dated snapshots are recorded; a changed reported snapshot is not pooled.

## 4. Analyze locally

```sh
jupyter lab notebooks/analysis.ipynb
```

Choose the `.venv` Python kernel. The notebook defaults to `data/raw/pilot`.
**Run All never calls a paid API.** GPT-2 weights/tokenizer download once from
Hugging Face; scoring then runs locally on Apple GPU, CUDA, or CPU.

Set `RUN_DIR = ROOT / "data/raw/main"` and `OUT_DIR = ROOT / "out/analysis-main"`
for main results. Pilot and main are never pooled. `MODEL_IDS = None` includes all
**saved** model runs; set it to a list of IDs to hold the analysis panel fixed while
collecting another model. `TASK_LIMIT = None` uses all frozen tasks; set it to 25
to retain the initial prefix when extending collection.

The notebook displays coverage, token usage, known costs and unknown-cost counts,
scores original prose, computes the paired task bootstrap, plots the timeline,
and shows low/middle/high-scoring
examples. Code/URLs/common paths are excluded by one visible regex; spacing is not
repaired. Characters include retained spaces/punctuation and are not UTF-8 bytes.

Only tasks scored for every selected model enter the graph. Adding a partially
collected model can shrink this shared set; the notebook reports the matched count
and task IDs. Failed, malformed, truncated, and all-code responses have no score.
One task has no CI; fewer than
ten is flagged as a pilot. The notebook refuses mixed settings or served snapshots.
Source IDs and one task/model observation—not tokens or repeated snippets—are the units.

Exports under `out/analysis-pilot/`:

- `surprisal.png` — the single graph;
- `samples.jsonl` — exact handoffs, scored prose, statuses, usage, and BPC;
- `summary.json` — plotted values, matched task IDs/scores, coverage, reference/software versions, and run hash.

## Interpretation and preservation

This measures **initial-investigation handoff writing under one fixed protocol**,
not hidden CoT, long-running autonomous behavior, task success, human comprehension,
or loss-of-control risk. Public benchmark tasks may have appeared in training.
Jargon, language, formatting, and incomplete retrieval can affect surprisal.
Inspect representative examples; do not tune the procedure to obtain a preferred ordering.

The 95% intervals describe task-sampling variation, not certainty that this proxy
measures intelligibility. Related tasks can remain dependent; missing-task exclusion
can bias results. Family release dates do not reconstruct historical alias behavior.
Full definitions and processing steps are in the notebook.

This is protocol `swe-handoff-v2` (automatic tool choice). Old `swe-handoff-v1`
direct-provider runs remain readable in their own directories but cannot be
extended or mixed with this protocol. The pre-Surplus checkpoint is `3984358`.

The previous implementation is recoverable at commit `5c92e6b` (the earlier
checkpoint at `research-v1-checkpoint` also remains). Old handmade contexts are
preserved in `data/legacy_contexts.jsonl`; existing `data/exemplars/`,
`data/sanity.jsonl`, and ignored `out/` data are unchanged and never mixed into
the new study. Raw requests and notebook exports are Git-ignored; review them
before sharing.

The public repository includes code, the model catalog, the frozen task-selection
manifest, synthetic controls, and attributed public excerpts. Generated task
packets, API responses, local session transcripts, and analysis exports stay local
by default. Keep credentials in environment variables; local `.env` files are
ignored and are not loaded automatically by the collector.

```sh
python -m pytest -q
```

Tests use fixtures/mocked HTTP responses and never make paid API requests.

Project code is available under the [MIT License](LICENSE). Third-party datasets
and quoted material retain their original terms.
