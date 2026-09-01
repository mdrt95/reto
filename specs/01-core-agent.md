# 01 — Core Agent

## Goal

Build a bounded, in-process application service that receives validated chat input and returns a verified answer with trace metadata. It must be callable from tests without FastAPI.

## Agent workflow

1. Apply deterministic input guards.
2. Reject questions naming entities absent from the profile with a not-found answer; refuse subjective ranking with a clarification.
3. Attempt structured intent classification as an optimization: `direct_question`, `search_query`, `filter_request`, `summary_request`, `follow_up`, `out_of_scope`, or `adversarial`.
4. If classification is unavailable, incomplete, or low-confidence, route unmistakable resume requests through deterministic universal search; otherwise clarify or fail closed.
5. Create one allowlisted read-only tool plan, then execute it.
6. Generate an answer using the profile, history, and tool result; use low factual temperature.
7. Verify claims against stable profile IDs.
8. Regenerate once if a claim is ungrounded; otherwise return only verified facts.
9. Apply output guard and return an answer plus internal trace metadata.

## Tools

All tools receive Pydantic arguments, return Pydantic result types, have no network or write access, and operate only on the loaded profile:

| Tool | Input | Result | Rule |
|---|---|---|---|
| `search_projects` | normalized query | matching project highlights and IDs | Match names, technologies, tags, summaries, and details case-insensitively |
| `filter_experience` | `filter_by`, `value` | matching experience highlights and IDs | Allow only `technology`, `tag`, and `role` |
| `query_profile` | allowlisted public field | typed profile projection | Allow `skills`, `languages`, `education`, and `current_role`; never unrestricted contact data |
| `summarize_profile` | audience | verified source selection for generation | Allow `technical`, `recruiter`, and `executive` |
| `search_resume` | normalized query, optional topic/source scope | selected derived facts and IDs | Cover experience, projects, skills, education, languages, summary, and optional career preferences |

Comparison requests compose relevant read-only results. Ranking is refused unless the caller provides an objective criterion represented in profile data.

## Grounding contract

- Fact IDs and matching canonical source IDs authorize fact selection and ordering only; they never ground provider prose.
- Fact-ID responses are reconstructed from canonical `ResumeFact` values with bounded English/Spanish templates. Fuzzy similarity, semantic entailment, and a second model judge are forbidden approval mechanisms.
- Provider prose is deliverable only for direct claims through the exact existing-English evidence compatibility path.
- Syntheses select and order multiple canonical fact IDs for deterministic rendering; generated inferred prose is not treated as grounded.
- Unknown information is stated as absent from the profile.
- The verifier returns `fully_grounded`, `partially_grounded`, or `not_grounded`, claim counts, unsupported claim summaries, and claim-to-source IDs. This metadata is internal by default.
- Deterministic matching decides normal cases. Model-assisted semantic review may explain a disputed match but cannot approve unsupported information.
- When no tool selected facts, no fact ID is authorized; the model cannot self-select facts.

## Guardrails and memory

- Input guard detects prompt-injection attempts, PII probes, contact-request probes (EN/ES), and wholly out-of-scope requests before the model call.
- Client instructions and history are untrusted. Only an allowlist may preserve language and desired concision.
- History and optional verified follow-up state are client-owned, limited by settings, and exist only for the current request. No database, embeddings, chunking, vector index, or server-side chat memory is used.
- Output guard rejects fabricated claims, system-prompt material, any contact data (phone or email), and ungrounded claims.

## Model boundary

Use the configured `claude-sonnet-4-6` model for structured intent classification and answer generation when available. Validate all model structured output with Pydantic. Provider failure uses source-backed deterministic rendering for clear resume requests and produces a typed application error only when the service cannot safely answer.

## Minimal verification

- A representative project search finds FAISS and a no-match search returns an empty typed result.
- A security tag filter finds the Security Console highlight.
- A prompt-injection request is blocked without a model call.
- A fabricated generated claim triggers verified-facts fallback.
- A low-confidence intent returns clarification without tool execution.
- A question about a nonexistent employer returns a not-found answer with no model call.
- A ranking request returns a clarification.

## Completion gate

Focused unit tests pass with a fake model adapter. The agent service can generate a traceable answer for a direct fact and a tool-assisted answer without HTTP routes or a real API key.
