# What is measured

The unit is a visible message with a producing model, source, channel, context or
session identifier when available, and exact original text. Descriptive comparisons
stay within comparable sources/channels/settings. Native logs describe the observed
workload; the small fixed-context collector describes a particular elicitation
protocol. Neither is automatically representative of all use of a model.

## Reference surprisal, not empirical character entropy

For prose string x, tokenized by one fixed reference model:

    B(x) = -sum_i log2 p_ref(token_i | preceding reference tokens)
    BPC(x) = B(x) / number_of_Unicode_characters(x)

The implementation pins GPT-2 and its tokenizer to revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. It prepends the tokenizer's BOS token,
scores every text token once, and uses overlapping 1,024-token windows with stride
512. Shifted labels are counted explicitly. Bits/token uses the same fixed
reference tokenizer, never the producer's tokenizer. Character counts include
spaces and punctuation in the scored prose, not UTF-8 byte counts.

This is model-based cross-entropy/surprisal. Unigram character entropy discards
ordering and therefore cannot distinguish a sentence from its shuffled version.
An unfamiliar domain, names, spelling, syntax, and language can all affect BPC.
It is an anomaly detector for English-like prose—not a direct comprehension score.
Small reference models can be useful reading-difficulty baselines, but that does
not validate this application. See [Oh and Schuler (2023)](https://aclanthology.org/2023.tacl-1.20/)
and the [Transformers perplexity guide](https://huggingface.co/docs/transformers/en/perplexity).

## Spacing comparison

A deterministic, dictionary-frequency-based segmenter proposes missing spaces in
uncommon alphabetic runs. It preserves all existing characters, case, and punctuation;
it does not paraphrase. Known rare words and small fragments have conservative guards.
CamelCase boundaries are counted separately; they are not automatically called
compression. Every proposed edit is inspectable.

For spacing-restored string r(x), report both BPC(r(x)) and:

    spacing_difference = [B(x) - B(r(x))] / len(x)

The shared denominator prevents newly inserted spaces alone from producing an
apparent improvement. A positive value indicates reduced *total* reference surprise
after this specific repair; a negative value is retained. It is not a percentage
of lost intelligibility explained. Failed or misleading segmentation, ambiguous
abbreviations, and non-English text are important limitations.

Glued-word share is the fraction of inferred word pieces belonging to detected
fused runs. It is a diagnostic with a heuristic denominator, not measured word error.

## Processing and observation limits

- Exact originals are immutable input data. A separate view excludes fenced/inline
  code, URLs, and common file-path forms. Matching is heuristic; bare identifiers,
  unfenced code, and some paths can remain. Full-document exclusions also apply to
  passages that cross a code fence. Exclusions become single spaces in scored prose;
  leading/trailing whitespace is trimmed. The removed fraction is reported.
- Contiguous source passages target 600 characters, split at whitespace without
  discarding short messages or breaking long fused runs. Passage scores restart
  reference context; they are not additive parts of the whole-message score.
- Empty/all-code/failed/unobservable messages have no reference score, not zero.
  No minimum word-count filter selectively removes compressed messages.
- Codex summaries, Claude visible thinking blocks, user-facing assistant text,
  child-agent messages/replies, and delegation prompts remain distinct channels.
  Visible reasoning may be a provider-produced summary, not raw internal reasoning.
  Encrypted or unavailable reasoning cannot establish either readability or its loss.
- Codex ancestor events copied into child rollouts are excluded by event identity;
  an unavailable child's completion is labeled with an unknown producing model,
  never the parent's. Claude streaming snapshots reuse the final message/block
  snapshot. Parsers depend on log formats and are covered by small fixture tests,
  not a guarantee about every historical client version.

## Single-metric Matplotlib timeline

The headline metric is original-prose reference BPC, with no spacing correction,
judge weighting, or composite transformation. For model m and task t, average the
message-level BPC values of its repetitions. The plotted model point is the
unweighted mean of those task means. Long messages and tasks with more repetitions
therefore do not automatically dominate the estimate.

Use tasks with valid scores for every dated model in the supplied release catalog.
Resample those task IDs with replacement 10,000 times, preserving model pairing,
and recompute each model mean. The 2.5th and 97.5th percentiles form a nominal 95%
interval. Never resample tokens or passages as independent observations. Record
the seed, resample count, complete-task set, and missing-data coverage. Removing
incomplete tasks can still create selection bias, which the interval does not fix.
For two models' difference, use the paired resampled differences; overlap of their
individual intervals is not a significance test.

For local logs, the explicit session mode averages messages per session and then
sessions per model, with independent session-level resampling. This is observational:
workload, settings, and harness changes can explain apparent model differences.
Both interval procedures assume sampled units are informative about a relevant
task/session population. Curated tiny sets do not confer model-wide 95% coverage.
One independent unit has no interval; fewer than 10 receives a pilot warning.

Release dates are supplied with source URLs, never inferred from observation dates.
The starter catalog uses public family-launch dates. Later alias responses are not
guaranteed historical snapshots, so this is a release-indexed comparison of the
observed outputs, not a reconstruction of behavior at launch. Sources, channels,
and protocols are not silently mixed. Returned snapshots and generation settings
must be consistent within each plotted model. All chart data and settings are
exported alongside the Matplotlib PNG.

## Optional judge and passage inspection

The judge rates word decoding, syntax reconstruction, and intended meaning separately
with four anchored grades. Exact quotations support difficult ratings. It also flags
missing context, is told not to equate technical difficulty with unclear writing,
and receives a small surrounding-text window plus any supplied task context.
This measures a fixed model's assessment, not whether a human actually understood
the text. Confident invented interpretations remain possible despite quote validation.

Reports show individual-message distributions, medians and P90, coverage counts,
and original passages next to repairs and judge evidence. Judge fractions use
successfully judged passages as their denominator; long messages can contribute
more passages. Those fractions are not message-level prevalence. Sessions/tasks are
the independent units to consider in later inferential work, not every passage.
The report deliberately has no overall intelligibility score or fitted historical
trend. With a tiny personal corpus, inspect direction, spread, and counterexamples.

The most useful small validation step is to skim a few high, middle, and low-scoring
passages. Fluent but underspecified controls can score low on reference surprise;
clearly written specialist prose can score high. If a proxy does not match the
phenomenon you see, report the mismatch instead of retuning it until the model
ordering agrees with an expectation.

The collection adapter uses official [Responses function calling](https://developers.openai.com/api/docs/guides/function-calling)
and the OpenAI judge uses [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Claude responses use the same local validation contract. Raw responses and served
model IDs are retained; request limits and cached failures prevent silent rerun costs.
