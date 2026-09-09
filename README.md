# Agent-text intelligibility

A small, descriptive study of whether frontier-model reasoning summaries and
agent-to-agent messages are getting harder to read. The aim is to find and inspect
changes in the text—not to estimate the probability of losing control of a model.

The headline result is **one Matplotlib graph of mean reference surprisal over model
release dates, with 95% task-bootstrap confidence intervals**. Higher means more
unexpected writing to a fixed reference model, not a calibrated loss of intelligibility.

The project has one message format and five commands. The main metric and optional diagnostics are:

- **Reference surprisal:** how unexpected the prose is to a fixed GPT-2, in bits per character.
- **Spacing difference:** how much total surprisal falls after inserting likely missing spaces, divided by the original character count.
- **Optional passage judge:** how much effort is needed to decode words, reconstruct syntax, and recover intended meaning, with exact supporting quotations.

Glued-word share is retained as a diagnostic. Character-frequency entropy, gzip,
readability formulas, composite scores, agent execution, experimental-factor
matrices, and routing services are no longer part of the workflow.

## Start locally—no API calls

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[reference,api,dev]'
python -m cotlegibility import --exemplars data/exemplars --out out/demo/messages.jsonl
python -m cotlegibility analyze out/demo/messages.jsonl --out out/demo
open out/demo/report.html
```

The first reference-scoring run downloads the pinned GPT-2 revision. Subsequent
runs use the local model and measurement cache. `--surface-only` skips GPT-2;
it is useful for checking imports and reports, but is not the full measurement.
Use `--device cpu` for consistent device selection, or leave automatic selection on.

The existing three public fragments are **selected illustrations, not a sample
from which to infer model-wide prevalence**. Their original text and source notes
are unchanged. Try the separate, explicitly synthetic sanity set:

```sh
python -m cotlegibility analyze data/sanity.jsonl --out out/sanity
```

Its 20 cases span ordinary prose, removed spaces, shuffled words, and underspecified
meaning. They are informal checks, not a validated intelligibility benchmark.

## Use existing messages

```sh
python -m cotlegibility import --codex ~/.codex/sessions --out out/local/messages.jsonl
python -m cotlegibility import --claude ~/.claude/projects --out out/claude/messages.jsonl
python -m cotlegibility analyze out/local/messages.jsonl --out out/local
```

An import requires explicit input paths. It does not contact a model. Multiple
sources can be combined using repeated `--jsonl FILE`, alongside the other flags.
Sources and channels remain separate in the report. Importing a narrower directory
or preparing a small Sample JSONL is the simplest way to limit a personal study.

Every message requires `id`, `provider`, `model`, `channel`, `source`, and exact
`text`. Optional fields are `timestamp`, `context_id`, `session_id`, `status`, and
`meta`. Put task context for the judge in `meta.reader_context`. Status is one of
`ok`, `missing`, `error`, or `unobservable`. Event IDs remove copied events, never
identical text from distinct runs. Conflicting records with the same ID fail loudly.

Local transcripts can contain private code and conversations. Outputs and raw data
are git-ignored; the HTML also contains original text. **Do not publish it without
review.** Reference scoring stays local. Opting into a judge sends source passages,
nearby text, and any reader context to that provider.

## Collect a small matched comparison

`data/contexts.jsonl` contains six frozen delegation tasks. `models.json` is an
editable, three-model starting matrix, not a claim about API availability or model
release order. Check that the exact model IDs are available to your account. Keep
settings fixed within each comparison; equal effort labels do not imply equal compute.

```sh
python -m cotlegibility collect --dry-run
# Set OPENAI_API_KEY, and ANTHROPIC_API_KEY only if adding Claude models.
python -m cotlegibility collect --replicates 1 --max-calls 18 --out out/collection
python -m cotlegibility analyze out/collection/messages.jsonl --out out/collection
```

Each request elicits one `spawn_agent` handoff; the child is never run. Prompts and
tool schemas are fixed across models. This compares **visible handoff writing in
this protocol**, not hidden CoT or native coding-agent behavior. Six tasks × three
models × one replicate is 18 requests. Start there before increasing replicates.

Collection uses direct official endpoints, with no silent retries or fallback
models. Exact requests, full responses, usage, requested/served model, and settings
are saved in `raw/`. Request hashes make reruns resumable. A per-run call limit is
not a dollar limit; output limits/settings are in `models.json`. Failed attempts are
cached too. To retry failures, use a new output directory (or deliberately move the
specific failed cache file aside). Unattempted cells remain visible as missing.
Changing the model/context matrix rebuilds `messages.jsonl` for that matrix while
leaving earlier raw records intact.

## Plot the single-metric timeline

After collecting and analyzing matched messages:

```sh
python -m cotlegibility plot out/collection
open out/collection/surprisal.png
```

The chart has one point per model: mean original-prose GPT-2 BPC, averaging
messages/repetitions within each task first, then tasks equally. Error bars are
95% percentile bootstrap intervals from 10,000 resamples of whole tasks, with
the same resampled task IDs across models. It uses only tasks scored for **every
dated model** in the release catalog; missing cells are reported. One task yields
a point without an interval. Fewer than 10 tasks triggers a pilot-sample warning.
The six starter tasks are a pilot, not sufficient evidence for a broad historical claim.

`data/model_releases.json` supplies exact model IDs, display labels, public-release
dates, and source URLs. Edit it to match the models in your study; it is separate
from API generation settings. Unknown/undated models are reported as excluded,
never placed at their message timestamps. Dates currently index family launches,
not necessarily the served snapshots of later moving aliases. The bootstrap does
not measure uncertainty about the validity of BPC as an intelligibility proxy.

The output is a single PNG, with no browser or extra plotting dependency.
`surprisal.jsonl` records exact plotted values, intervals,
unit-level scores, selected event IDs, coverage, release sources, scoring provenance,
and bootstrap settings. It does not copy message text into the chart. Use
`--out PATH.png`, `--resamples N`, and `--seed N` to customize exports.

For an explicitly **observational** plot of existing logs, use session clusters:

```sh
python -m cotlegibility plot out/local --source codex_rollout --unit session
```

Session means receive equal weight; each model's sessions are bootstrapped
independently. Workloads are not matched and the chart says so. `--channel` chooses
one message type (default `subagent_prompt`). Sources/channels are never pooled;
mixed generation configurations or served snapshots within a model fail loudly.
The selected public fragments and synthetic controls cannot populate the default
matched-model chart. An empty chart states the missing-data limitation explicitly.

## Add one fixed reader, optionally

```sh
python -m cotlegibility analyze out/collection/messages.jsonl --out out/collection \
  --judge-model gpt-5.6-sol --judge-parameters '{"reasoning":{"effort":"medium"},"max_output_tokens":4096}' \
  --judge-limit 30
```

The judge uses anchored `none / minor / substantial / unresolved` categories for
words, syntax, and meaning. Non-`none` ratings must quote exact source text; malformed
answers remain errors, not low-difficulty scores. Missing context is flagged separately.
Producer identity is not supplied as metadata, but may be inferable from the text.

The request budget is shared round-robin across source/model/channel/settings
groups, with stable hash ordering of passages within each. It is a balanced
convenience sample. Report denominators explicitly exclude unjudged passages.
Keep the same judge/settings across producer models; a changed served-model ID is
saved per passage. `--judge-limit 0` reuses cached judgments without new API requests.
As with collection, failed judgments are cached. No judge is run unless requested.

## Read the results

Each analysis directory contains:

| File | Purpose |
| --- | --- |
| `report.html` | Self-contained local report: distributions, coverage, exact passages, restored text, evidence |
| `summary.csv` | Group counts, median/P90 message BPC, spacing diagnostics, judged-passage difficulty fractions |
| `distributions.png` | Dots for individual messages and bars for medians; no implied chronological order |
| `measured_messages.jsonl`, `passages.jsonl` | Exact originals, provenance, source offsets, and individual measurements |
| `manifest.jsonl`, `cache/` | Versions, settings, dataset hash, and reusable reference/judge results |

`python -m cotlegibility report out/collection` rebuilds presentation without model
calls. Start with model distributions, then inspect high-surprisal passages and a
few ordinary ones. Do not combine selected screenshots, synthetic controls, and
collected/logged messages into one prevalence number. For a historical trend, use
verified snapshot/release metadata, matched tasks/settings/channels, and more than
one session or task per model. Collection timestamp alone is not model chronology.

See [METHODS.md](docs/METHODS.md) for definitions, preprocessing, and limitations.
Run tests with `.venv/bin/python -m pytest`; tests never make live API requests.

## Checkpoint and scope

The previous implementation is preserved at tag `research-v1-checkpoint` (commit
`62d7e2a`). Its experiment runners and conclusion-oriented research documents were
removed from the active tree; existing exemplars and ignored generated data were
preserved. For example, `git show research-v1-checkpoint:docs/RESEARCH_PLAN.md`
reads the old plan without restoring the old machinery.
