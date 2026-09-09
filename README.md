# CoT legibility analysis

Tooling for a METR-style report on how legible frontier models' chain-of-thought and, especially,
their messages to sub-agents are, and how that changes across model generations (OpenAI GPT-5.x → GPT-6,
Anthropic Claude 4.x → Claude 5 / Fable).

The research design lives in `docs/RESEARCH_PLAN.md`. The code is a small Python package:

```
cotlegibility/
  schema.py              Sample: one text with provider / model / channel / source metadata
  sources/codex.py       ~/.codex/sessions rollouts  -> spawn_agent messages, wait_agent replies, summaries
  sources/claude_code.py ~/.claude/projects jsonl    -> Agent-tool prompts, sidechain replies, thinking summaries
  metrics/surface.py     deterministic metrics: glued-word share, function-word rate, whitespace, tokens, gzip ...
  metrics/perplexity.py  bits/char under a pinned open reference LM (gpt2 by default)
  metrics/judge.py       reader-model protocols: reconstruction, directive recovery, readability rating
scripts/analyze_local.py  run everything observable on this machine, write out/metrics.csv + out/fig_surface.png
data/exemplars/           public examples with provenance: the GPT-6 Astra screenshot (Petersson), the Kilo Astra fragment, the Sonnet 5 system-card thinking transcript
```

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -e ".[ppl]"     # add [api] for the judge / replay experiments
.venv/bin/python scripts/analyze_local.py --ppl                 # ~2 min with the GPT-2 reference model
```

Reference-LM scoring runs on Apple MPS when available. API-based protocols need `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` (or `ant auth login`) and are not run by the local script.

## Harness runs (free under the Claude and ChatGPT subscriptions)

```
.venv/bin/python scripts/run_harness.py --dry-run
.venv/bin/python scripts/run_harness.py --harness claude --models claude-sonnet-5 --tasks probe
.venv/bin/python scripts/run_harness.py --harness claude --models claude-opus-4-6 claude-opus-5 claude-sonnet-5 claude-fable-5-1 --tasks all --condition default human_reads
.venv/bin/python scripts/analyze_local.py --runs out/harness_runs.jsonl --ppl
```

Each run gets a fresh synthetic repo (`scripts/make_target_repo.py`), tasks come from `tasks/tasks.json`, and the
manifest `out/harness_runs.jsonl` lets the analysis join transcripts to model / task / condition by `cwd`.
Claude Code logs parent → sub-agent prompts in plaintext. Codex ≥ 0.153 encrypts them (MultiAgent v2), so
Codex runs only yield user-facing text, reasoning summaries and spawn counts; OpenAI-side sub-agent messages
come from the API replay below.

## Experiments

```
.venv/bin/python scripts/replay_subagent.py --dry-run          # extract real pre-spawn contexts, no API calls
.venv/bin/python scripts/replay_subagent.py --models gpt-5.5 claude-opus-5 --audience none human_reads --limit 5
```

## Running through Surplus Intelligence

All models in the study are available through one router, https://www.surplusintelligence.ai (OpenAI-compatible
`/v1`, plus an Anthropic-shaped `/anthropic/v1/messages`). Setup:

1. Sign in at surplusintelligence.ai (Twitter, Discord, or wallet connect; a wallet is created for you).
2. Fund the wallet with USDC on Base (Coinbase Onramp or MoonPay in-app, or bridge) and approve USDC for
   SettlementV2 on the Buy page. Funds stay in your wallet.
3. Copy the API key from the Buy page (prefix `inf_`) and `export SURPLUS_API_KEY=inf_...`.
4. Pre-flight: `.venv/bin/python scripts/surplus_smoke.py --plan` (no key needed) then
   `.venv/bin/python scripts/surplus_smoke.py --live --models gpt-5.6-luna claude-sonnet-5` (a few cents).
5. Pilot: `.venv/bin/python scripts/replay_subagent.py --models gpt-5.6-sol gpt-6-astra claude-opus-5 claude-fable-5-1 --audience none openai_guidance --limit 5`.

Provenance rules baked into `cotlegibility/providers/surplus.py`: requests are pinned to first-party-hosted
sellers (`anthropic`, `openai`, `bedrock`) via the `provider` allow-list; the `x-si-*` route headers, the
served model id, usage and reasoning-token counts are stored with every result; Claude models go through the
Anthropic surface so extended thinking is real. Surplus does not attest model identity or quality, so run
`scripts/surplus_smoke.py --sweep <model>` to compare sellers before trusting a cheaper route.
