# Agent-handoff surprisal

A small, controlled study of whether newer models write less predictable
agent-to-agent messages. **One metric: pinned GPT-2 bits per character of prose.**
One Matplotlib graph: model-family release date versus mean surprisal, with 95%
task-bootstrap intervals. Higher is not automatically less intelligible.

## The project

- `scripts/prepare_tasks.py` freezes five pilot and 50 disjoint main-study tasks.
- `scripts/collect.py` collects one outgoing handoff per task/model, without running a child.
- `notebooks/analysis.ipynb` reads raw responses, scores prose locally, and draws the graph.
- `notebooks/surprisal.py` is the small, tested reference scorer.
- `data/models.json` holds the three requested models, generation settings, and sourced release dates.
- `data/task_manifest.json` records the pinned dataset, sampling seed, task IDs, and context hashes.
- `tests/` covers preparation, safe collection, scoring, bootstrap, and notebook execution.

No agent framework, judge, spacing repair, multi-metric report, local-log importer,
provider router, or benchmark execution harness. Collection and analysis are independent.

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

Default: read-only preview of five pilot tasks × three models = 15 requests.
It needs no API key and creates no raw run.

Only the following command spends money, after you set `OPENAI_API_KEY`:

```sh
python scripts/collect.py --execute --max-calls 15
```

Uses the [official Responses API](https://developers.openai.com/api/docs/guides/function-calling)
with a strict `spawn_agent(message)` function. The worker is described as receiving
the code excerpts and handoff, but not the issue or parent conversation. The child
is never executed. There are no brevity, shorthand, or readability instructions.

Requests go directly to OpenAI, with no retries or provider/model fallback.
The catalog starts with GPT-5.5, GPT-5.6 Sol, and GPT-6 Astra, medium reasoning,
and an 8,192-token output cap. Verify model access for your account.
[Hidden reasoning counts toward output usage and the cap](https://developers.openai.com/api/docs/guides/reasoning).
Equal effort labels do not imply equal compute.

`--max-calls` bounds **new calls in this invocation**; per-request output caps
bound output tokens. These are **not a dollar ceiling**: input tokens also cost
money. Inspect actual pilot usage before scaling up. There is no Batch adapter in
version one; [direct Batch](https://developers.openai.com/api/docs/guides/batch)
is a possible later cost optimization.

Each run stores `run.json` plus one `request-<hash>.json` per attempt under
`data/raw/pilot/`. The record is written before the call and finalized afterward,
preserving exact requests, full responses, timestamps, provider request IDs,
usage, returned model IDs, and failures. Completed records are not overwritten.
An OS lock prevents concurrent collectors in one directory.

Reruns skip **every** existing attempt, including errors and interrupted/uncertain
calls. An API error stops the run. Inspect it before resuming; uncertain calls may
have been billed. Do not delete records to force retries. A changed configuration
requires a new output directory and constitutes a separate run.

## 3. Analyze locally

```sh
jupyter lab notebooks/analysis.ipynb
```

Choose the `.venv` Python kernel. The notebook defaults to `data/raw/pilot`.
**Run All never calls a paid API.** GPT-2 weights/tokenizer download once from
Hugging Face; scoring then runs locally on Apple GPU, CUDA, or CPU.

The notebook displays coverage and token usage, scores original prose, computes
the paired task bootstrap, plots the timeline, and shows low/middle/high-scoring
examples. Code/URLs/common paths are excluded by one visible regex; spacing is not
repaired. Characters include retained spaces/punctuation and are not UTF-8 bytes.

Only tasks scored for every requested model enter the graph. Failed, malformed,
truncated, and all-code responses have no score. One task has no CI; fewer than
ten is flagged as a pilot. The notebook refuses mixed settings or served snapshots.
Source IDs and one task/model observation—not tokens or repeated snippets—are the units.

Exports under `out/analysis-pilot/`:

- `surprisal.png` — the single graph;
- `samples.jsonl` — exact handoffs, scored prose, statuses, usage, and BPC;
- `summary.json` — plotted values, matched task IDs/scores, coverage, reference/software versions, and run hash.

After inspecting the pilot for realism, preview and explicitly collect the main set:

```sh
python scripts/collect.py --split main
python scripts/collect.py --split main --execute --max-calls 150
```

Change the notebook's `RUN_DIR` to `ROOT / "data/raw/main"` and `OUT_DIR` to
`ROOT / "out/analysis-main"`. Pilot and main are never pooled.

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

The previous implementation is recoverable at commit `5c92e6b` (the earlier
checkpoint at `research-v1-checkpoint` also remains). Old handmade contexts are
preserved in `data/legacy_contexts.jsonl`; existing `data/exemplars/`,
`data/sanity.jsonl`, and ignored `out/` data are unchanged and never mixed into
the new study. Raw requests and notebook exports are Git-ignored; review them
before sharing.

```sh
python -m pytest -q
```

Tests use fixtures/mocked HTTP responses and never make paid API requests.
