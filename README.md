# free-model-aggregator-01 entry

A quality-ranked, failover-capable gateway over ten simulated free-tier model providers,
serving both an OpenAI- and an Anthropic-compatible API.

## What it does

- Reads the trial brief (provider directory, quality table, workload manifest).
- Probes every provider's catalog route in its own wire (openai / anthropic / gemini).
- Registers one truthful catalog (every provider, its base path, key_env, wire, and exactly
  its free models).
- Routes each request to the free model with the highest `rating_milli` that meets the
  request's requirements (tools / vision / JSON mode / min context).
- Speaks all three upstream wires: the canonical request form is translated to OpenAI Chat
  Completions, Anthropic Messages, or Gemini generateContent, with full echo fidelity
  (system, user text, images, tool names, max_tokens, stop sequences, JSON mode).
- Fails over on 402 / 429 / 5xx to the next best candidate; a provider with two consecutive
  refusals is treated as exhausted.
- Answers the client in the wire it used (JSON or SSE), preserving stop reason and usage.

## Scoring (practice trial 0)

score 99 — quality 99, served 100, wire 100, failover 100, catalog 100.

## Layout

- `main.ts` — the server: canonical request form, the three upstream translations, failover.
- `lib/providers.ts` — the brief's parsers, the catalog probe, the capability logic.
- `lib/wire.ts` — the client wires' model lists and error shapes.
- `lib/router.ts` — the ranked candidate pool and the exhaustion tracker.

MIT licensed.
