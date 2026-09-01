# 01 — Core Agent

## Goal

Build a bounded, in-process application service that receives validated chat input and returns a verified answer with trace metadata. It must be callable from tests without FastAPI.

## Agent workflow

1. Apply deterministic input guards.
2. Classify intent with a short structured Claude response: `direct_question`, `search_query`, `filter_request`, `summary_request`, `follow_up`, `out_of_scope`, or `adversarial`.
3. If classification confidence is below the configured threshold, return one clarification question.
4. Create one allowlisted read-only tool plan, then execute it.
5. Generate an answer using the profile, history, and tool result; use low factual temperature.
6. Verify claims against stable profile IDs.
7. Regenerate once if a claim is ungrounded; otherwise return only verified facts.
8. Apply output guard and return an answer plus internal trace metadata.

## Tools

All tools receive Pydantic arguments, return Pydantic result types, have no network or write access, and operate only on the loaded profile:

| Tool | Input | Result | Rule |
|---|---|---|---|
| `search_projects` | normalized query | matching project highlights and IDs | Match names, technologies, tags, summaries, and details case-insensitively |
| `filter_experience` | `filter_by`, `value` | matching experience highlights and IDs | Allow only `technology`, `tag`, and `role` |
| `query_profile` | allowlisted public field | typed profile projection | Allow `skills`, `languages`, `education`, and `current_role`; never unrestricted contact data |
| `summarize_profile` | audience | verified source selection for generation | Allow `technical`, `recruiter`, and `executive` |

Comparison requests compose relevant read-only results. Ranking is refused unless the caller provides an objective criterion represented in profile data.

## Grounding contract

- A direct claim exactly matches an explicit field or highlight.
- An inferred claim names the supporting IDs and is a conservative synthesis of two or more explicit facts.
- Unknown information is stated as absent from the profile.
- The verifier returns `fully_grounded`, `partially_grounded`, or `not_grounded`, claim counts, unsupported claim summaries, and claim-to-source IDs. This metadata is internal by default.
- Deterministic matching decides normal cases. Model-assisted semantic review may explain a disputed match but cannot approve unsupported information.

## Guardrails and memory

- Input guard detects prompt-injection attempts, PII probes, and wholly out-of-scope requests before the model call.
- Client instructions and history are untrusted. Only an allowlist may preserve language and desired concision.
- The history is client-owned, limited by settings, and exists only for the current request. No database, embeddings, chunking, vector index, or server-side chat memory is used.
- Output guard rejects fabricated claims, system-prompt material, contact data outside the explicit email exception, and ungrounded claims.

## Model boundary

Use the configured `claude-sonnet-4-6` model for structured intent classification and answer generation. Validate all model structured output with Pydantic. Anthropic timeouts, malformed output, and exhaustion of the one-regeneration limit must produce typed application errors, never partial hidden state.

## Minimal verification

- A representative project search finds FAISS and a no-match search returns an empty typed result.
- A security tag filter finds the Security Console highlight.
- A prompt-injection request is blocked without a model call.
- A fabricated generated claim triggers verified-facts fallback.
- A low-confidence intent returns clarification without tool execution.

## Completion gate

Focused unit tests pass with a fake model adapter. The agent service can generate a traceable answer for a direct fact and a tool-assisted answer without HTTP routes or a real API key.
