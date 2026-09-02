# Decisions — Banorte CV Agent

## D-001: Deliver the public chat application before external-agent interoperability

**Status:** Accepted

The application will first deliver a stable FastAPI-backed public chat experience. A protocol adapter is a separate deliverable, not an assumption baked into the core API.

**Why:** OpenAI-style Responses payloads, A2A, and MCP solve different integration problems. Combining them in one unvalidated endpoint would create false compatibility claims and make testing ambiguous.

**Consequence:** The external integration choice remains pending. Do not advertise Claude.ai, A2A, MCP, or Open Responses compatibility until the selected adapter has an integration test against its target client.

## D-002: Establish one runtime source of truth for professional claims

**Status:** Accepted

`MDRT Resume.json` is the reviewed authoring input. `data/profile.json` is the sole runtime source of truth. The profile JSON included in `PLAN.md` is illustrative only.

**Why:** The two existing documents differ in wording, tags, and optional fields. Two live sources would make grounding and tool results inconsistent.

**Consequence:** Changes to the runtime profile require reconciliation with the authoring input and a factual review. Tool-facing list fields use explicit empty-list defaults.

## D-003: Ground every answer with stable source references and safe fallback

**Status:** Accepted

Each generated claim may retain selected fact IDs and profile source IDs. Those IDs authorize only canonical fact selection and ordering; they NEVER prove that provider prose semantically matches a fact. The service reconstructs fact-ID responses from canonical `ResumeFact` values. Legacy provider prose is deliverable only when a direct claim and its evidence both exactly occur in canonical source text. Generated inferred prose and uncited prose are never delivered.

**Why:** Natural-language similarity alone cannot prove a professional claim is supported.

**Consequence:** Evaluation scenarios must include expected source IDs; observability records claim-to-source mappings without logging conversation text.

## D-004: Treat all client instructions as untrusted

**Status:** Accepted

Client instructions, histories, and tool-like text cannot modify system behavior. Only an explicit allowlist may carry harmless presentation preferences.

**Why:** Elevating remote input to system context creates a prompt-injection path.

**Consequence:** The input guard runs before intent classification and the system prompt remains server-owned.

## D-005: Never disclose contact data; block contact requests at the input guard

**Status:** Amended 2026-09-01

The agent never discloses Marco's phone number or email address, under any framing. An explicit or implicit request for contact details is blocked by the input guard as a PII probe, before classification or any model call, and receives a bilingual (EN/ES) safe reply.

**Why:** A public chatbot should not make automated contact-data harvesting effortless, and a narrow "explicit request" exception proved impossible to distinguish reliably from a probe phrased as a legitimate ask.

**Consequence:** `query_profile` must not return unrestricted contact data, logs must exclude contact information, and the earlier `contact_requested` code path is removed. There is no runtime condition under which the agent emits an email address or phone number.

## D-006: Define measurable release gates

**Status:** Accepted

The fixed evaluation suite has deterministic accuracy, safety, tool-use, latency, and cost gates defined in `PLAN.md`.

**Why:** Metrics without thresholds cannot establish whether the demo is actually ready.

**Consequence:** A model-graded evaluation may inform quality analysis but cannot replace the deterministic release checks.

## D-007: Keep conversations ephemeral and apply operational limits

**Status:** Accepted

The frontend supplies its transcript for each turn; the backend retains only correlation metadata and applies request-size and per-IP rate limits.

**Why:** This is sufficient for the portfolio-agent use case while reducing privacy, cost, and abuse risk.

**Consequence:** Document transcript caps, sanitized failure responses, and provider `PORT` binding before deployment.

## D-008: Use a modular-monolith, ports-and-adapters architecture

**Status:** Accepted

The FastAPI application owns HTTP delivery only. Application services orchestrate use cases; domain models and pure tools operate on validated profile data; infrastructure adapters own Anthropic, logging, configuration, and any future protocol integration.

**Why:** This keeps the CV domain testable without HTTP or model calls and lets a future A2A, MCP, or Responses-style adapter reuse the same agent service.

**Consequence:** Dependencies point inward. Route handlers, SDK clients, and provider payloads must not leak into domain models or tool signatures.

## D-009: Use Python 3.12, FastAPI, Pydantic v2, a static frontend, and Render first

**Status:** Accepted

The platform is a Python 3.12 FastAPI service with Pydantic v2 validation, a no-build static HTML/JavaScript frontend, and a Render Web Service deployment target. The service binds to `PORT`, falling back to `8000` locally.

**Why:** The project needs a compact, type-safe API and a deployable demo rather than frontend framework complexity.

**Consequence:** The runtime is stateless, configuration is environment-based, logs are JSON to stdout, and Railway remains a documented fallback rather than a second initial deployment target.

## D-010: Keep the v1 agentic workflow bounded and observable

**Status:** Accepted

Each turn follows: request validation → input guard → structured intent classification when available → deterministic tool plan → tool execution → answer generation when available → grounding verification → output guard → response/logging. A low-confidence or unavailable classifier may be bypassed only when deterministic routing proves the request is resume-related; otherwise the service clarifies or fails closed.

**Why:** The agent must demonstrate intentional tool use without giving an LLM unchecked control over data or execution.

**Consequence:** Tools are allowlisted, read-only, typed, and bounded to the validated profile. The orchestrator permits at most one initial tool plan and one constrained regeneration after a grounding failure.

## D-011: Use structured-profile queries, not embeddings or chunking, in v1

**Status:** Accepted

The CV is a small, typed JSON document; v1 searches normalized fields, tags, technologies, and stable IDs directly in memory. It does not build embeddings, vector indexes, or chunks.

**Why:** Retrieval infrastructure would add latency, cost, non-determinism, and an extra grounding surface without improving recall for this dataset.

**Consequence:** If the scope later expands to long-form documents, add a separate ingestion specification defining document segmentation, metadata, embeddings, hybrid retrieval, and citation validation. Do not infer that strategy from the unrelated Sybil project description.

## D-012: Use one configured Anthropic generation model and deterministic verification first

**Status:** Accepted

`claude-sonnet-4-6` is the configured model for factual answer generation and structured intent classification. Grounding, profile lookups, validation, and guardrails are deterministic wherever possible; a model-assisted semantic check is a narrow fallback only.

**Why:** Multiple model tiers are unnecessary for a small CV corpus and complicate cost attribution and evaluation.

**Consequence:** Model name, timeout, temperature, and token limits are configuration values. Any future model substitution requires rerunning the fixed evaluation suite and recording the change.

## D-013: Treat application memory as client-owned bounded transcript state

**Status:** Accepted

The application does not maintain a conversational database in v1. The frontend sends a bounded recent transcript and may send compact verified state (`last_topic`, `last_source_ids`, `last_entities`, `last_tool`, and `response_language`) on every turn; the backend uses both for that request only and retains correlation metadata, not conversation contents.

**Why:** The CV agent needs short follow-ups, not durable personalization. This eliminates a datastore, retention risk, and session-consistency problem.

**Consequence:** The API specifies history caps and transcript validation. A future persistent-memory feature requires explicit consent, retention, deletion, and access-control specifications.

## D-014: Enforce a public API contract at `/api/chat`; add adapters later

**Status:** Accepted

The core service exposes a project-owned `POST /api/chat` contract. Any OpenAI-style Responses, A2A, or MCP integration translates to and from the same application service in a separate adapter.

**Why:** A stable first-party contract is required for the frontend and avoids falsely presenting a provider-shaped payload as a standards implementation.

**Consequence:** External adapters cannot bypass input validation, guardrails, grounding, or output validation. Each has its own contract-test suite when selected.

## D-015: Validate every boundary and return sanitized problem responses

**Status:** Accepted

Pydantic validates configuration, profile data, HTTP requests, tool arguments/results, model structured outputs, and observability events. Errors are mapped to stable public codes with request IDs; provider internals stay in logs only.

**Why:** This prevents malformed data from becoming prompts, tool input, or client-visible stack traces.

**Consequence:** Validation failures are client errors; guardrail rejections are safe successful replies. Upstream model failures return deterministic verified answers when selected profile facts exist, and return retry-safe server errors only when no safe answer can be produced. Unexpected failures remain sanitized.

## D-016: Make observability privacy-preserving and decision-oriented

**Status:** Accepted

Emit one structured log event per completed turn with request/correlation IDs, stage latency, model use, intent, tool plan/result count, grounding summary, guardrail outcome, cost estimate, and sanitized error code.

**Why:** The demo must prove behavior and diagnose regressions without retaining private prompts or contact data.

**Consequence:** Never log transcript text, prompts, raw model output, raw provider payloads, API keys, email addresses, or phone numbers. Trace data is retained according to the deployment provider's log policy only.

## D-017: Use a minimal, risk-based test and evaluation set

**Status:** Accepted

Tests cover the profile loader, each typed tool's representative match/no-match behavior, input guard rejection, grounding fallback, and API success/validation/error paths. The evaluation set covers one or two representative scenarios per risk category rather than exhaustive permutations.

**Why:** Small focused tests protect the highest-risk contracts while keeping the project proportionate to a demo challenge.

**Consequence:** New behavior needs a test only when it creates a new boundary, failure mode, or regression risk. Do not grow a giant test suite merely to increase test count.

## D-018: Enforce the focused offline checks through GitHub Actions

**Status:** Accepted

Run the existing pytest suite, source compilation, and fixed evaluation-scenario validation on every push and pull request using Python 3.12. The workflow has read-only repository permissions and never receives an Anthropic API key.

**Why:** The project needs an independent, repeatable control that prevents basic regressions from reaching the shared repository, without paying for model calls or expanding the test surface.

**Consequence:** Live model evaluation remains an explicit, credentialed release step. CI failure blocks confidence in a change; a green CI run does not substitute for the production smoke test or live evaluation gate.

## D-019: Derive one universal fact catalog and index from the canonical profile

**Status:** Accepted

`search_resume` derives typed facts at runtime from the validated `Profile`: professional summary fields, experience, projects, skills, education, languages, and optional career preferences. Each fact has a deterministic `fact_id`, existing profile `source_id`, topic, display text, entity, and searchable keywords. Normalization removes accents and punctuation, case-folds text, and maps a bounded English/Spanish synonym set. Specialized tools remain preferred when they return a valid source-backed plan; universal search is the fallback.

**Why:** Hand-maintaining a second search corpus would create competing biographical truth. Narrow phrase branches cannot provide complete multilingual recall.

**Consequence:** `data/profile.json` remains the ONLY biographical truth source. Index entries and rendered factual values must be derived from model data. The index is in-memory and deterministic; it uses no embeddings, provider calls, or separately authored facts.

## D-020: Make classification and generation optional optimizations

**Status:** Accepted

Valid specialized classifier plans win. Incomplete, low-confidence, locally invalid, authentication, transport, or provider-failure outcomes may fall back to deterministic routing only for clearly resume-related input. Out-of-scope and adversarial decisions never become profile answers. When generation is unavailable or fails grounding, typed selected facts are rendered deterministically; a 503 is reserved for turns where no safe deterministic answer or clarification exists.

**Why:** Provider availability must not determine whether already-loaded verified profile facts can be answered.

**Consequence:** Missing profile data receives an explicit language-aware “not specified in the profile” response. Deterministic routing is a bounded domain recognizer, not a general intent guesser.

## D-021: Ground multilingual claims through selected fact/source identity

**Status:** Amended 2026-09-01

Fact IDs and matching source IDs authorize selection and order only. They do not validate, entail, or authorize arbitrary generated English or Spanish claim text. For fact-ID responses, the delivery boundary ignores provider prose and reconstructs public text from canonical selected `ResumeFact` values using bounded English/Spanish templates. The verifier does not use fuzzy similarity, semantic entailment, or a second model judge. Provider prose remains available only for direct claims through the exact existing-English evidence compatibility path; inferred provider prose is rejected because exact excerpts cannot prove a synthesis. Model prose is deliverable for a fact-ID response only through the deterministic containment gate defined in D-029; every other path still renders canonical fact text.

**Why:** English substring matching rejects faithful Spanish, but citation identity and string similarity also cannot prove factual entailment. Stable typed identity safely authorizes which canonical facts may be rendered—not what a provider may say about them.

**Consequence:** Provider prompts receive only the turn's allowed fact IDs plus valid source IDs as selection hints. Public wording comes from canonical facts and deterministic templates. The output guard still runs after rendering and blocks any contact data (phone or email) regardless of source.

## D-022: Carry compact verified follow-up state in the public contract

**Status:** Accepted

`POST /api/chat` accepts and returns an optional state object containing `last_topic`, `last_source_ids`, `last_entities`, `last_tool`, and `response_language`. The frontend carries it with the bounded transcript. Follow-ups resolve only through those verified references; missing state or multiple plausible entities produces clarification.

**Why:** Transcript prose cannot safely reconstruct source identity, and server-side memory would add retention and consistency risk.

**Consequence:** The original request and response fields remain compatible. State is client-owned, bounded, validated, ephemeral, and contains no contact data or speculative facts.

## D-023: Model career preferences as optional canonical data

**Status:** Accepted

The profile schema supports optional `career_preferences`, but Marco's current `data/profile.json` leaves it absent because desired roles, seniority, locations, and work arrangements have not been explicitly stated. Opportunity data, if introduced later, must use a separate `opportunity:*` namespace.

**Why:** Recruiting preferences are biographical claims and cannot be inferred from skills, current employment, or project history.

**Consequence:** The assistant says the preference is not specified until the canonical profile is factually updated. It does not generate speculative opportunity facts.

## D-024: Reject unknown named entities before classification

**Status:** Accepted

Before intent classification or any model call, a deterministic check scans the user message for capitalized named entities (excluding sentence-initial words and Marco's own name) and compares them against the derived fact catalog. If any named entity appears nowhere in the catalog, the agent immediately answers "I couldn't find anything about X in Marco's profile" (bilingual EN/ES) with grounding status `profile_missing`, and skips classification, tool execution, and generation entirely. Universal search additionally treats prepositions and auxiliary verbs (EN/ES) as stop words so they can never score as keyword matches.

**Why:** A wrong question — one naming a company, person, or product absent from the profile — must not receive a right-sounding answer stitched together from unrelated facts that happen to share a keyword. Fabrication risk from a probe like "Tell me about Marco's experience at Google" comes from the model bridging an unknown entity to loosely related profile facts, not from the model inventing text out of nothing.

**Consequence:** A new deterministic pre-check runs ahead of the D-010 workflow. It has its own false-positive risk (a legitimate entity absent from the catalog wording) that must be covered by test cases; the catalog must stay derived from `data/profile.json` so the check does not drift from the truth source.

## D-025: Clarify subjective ranking deterministically and localize all boundary replies

**Status:** Accepted

A request for a subjective ranking ("best", "most important", "top", "favorite" and Spanish equivalents) without an objective, profile-representable criterion receives a deterministic bilingual clarification asking for one (a technology, tag, or role), rather than an invented ranking. Its grounding status is `clarification`. All boundary messages — out-of-scope redirect, clarification, verified-facts-only fallback, output-guard block, and input-guard replies — are rendered bilingually (EN/ES) through the existing deterministic language detector, not only the happy-path answers.

**Why:** "Best project" or "most important skill" has no objective answer in a factual CV corpus; answering it invents a ranking criterion the profile never asserted. Boundary messages are exactly the turns most likely to reach a Spanish-speaking user probing the agent's limits, so they cannot be English-only without breaking the multilingual guarantee everywhere it matters most.

**Consequence:** `filter_experience` and `search_resume` ranking-shaped queries route to clarification instead of a best-effort tool call. Every guardrail and fallback reply needs an EN/ES pair, checked by the language detector already used for successful answers.

## D-026: Trust one platform proxy for client addressing

**Status:** Accepted

Deployment sits behind exactly one trusted platform proxy (Render). Uvicorn runs with `--proxy-headers` and `FORWARDED_ALLOW_IPS` (default `*` in the `Dockerfile`, overridable via environment) so per-IP rate limiting reads the real client address from the forwarded headers instead of Render's internal edge IP.

**Why:** Without `--proxy-headers`, every request appears to originate from Render's proxy, collapsing the per-IP rate limit (D-007) into one shared global budget and defeating its purpose. `FORWARDED_ALLOW_IPS=*` is only safe because Render's Docker web services sit behind exactly one internal proxy hop that the application cannot bypass.

**Consequence:** Rate-limit tests must exercise the forwarded-header path, not just a raw `TestClient` connection. If the deployment target ever changes to a multi-hop or untrusted-proxy topology, `FORWARDED_ALLOW_IPS` must be narrowed to the specific trusted hop instead of `*`.

## D-027: Cache the turn-independent generation prefix at the provider

**Status:** Accepted

Answer generation sends its stable content — the behavioral instruction, the contact-stripped profile payload, the output schema, and the grounding rules — as `system` text blocks, with one `cache_control: {"type": "ephemeral"}` breakpoint on the last of them. Only per-turn content (`user_message`, `history`, `tool_result`, `allowed_source_ids`, `allowed_fact_ids`) stays in the user message. The prefix is serialized with `sort_keys=True` so its bytes never depend on dictionary ordering. Intent classification stays uncached.

**Why:** That prefix was previously embedded in the same JSON blob as the user message, so it was re-billed at full input price on every call despite being byte-identical across every user and every turn. Cache reads cost roughly a tenth of base input price, and the prefix measures about 1,736 tokens — above the 1,024-token minimum cacheable prefix for `claude-sonnet-4-6`, so entries are actually created. The classifier's stable prefix is a few hundred tokens, below that minimum, where a breakpoint would create no entry and read nothing back. Moving the profile into `system` also strengthens D-004: trusted server data and operator instructions now sit in the operator channel, ahead of the untrusted user turn.

**Consequence:** The cached prefix must stay free of per-turn bytes; a single request-scoped value placed before the breakpoint would silently drop the hit rate to zero with no error, so a test asserts the prefix is identical across turns. Any future edit to the profile payload, output schema, or rules invalidates every entry once and re-warms on the next request, which is acceptable because those change only on deploy. The default five-minute TTL is correct while requests keep arriving; each read refreshes the entry at no extra cost. Confirming the cache actually fires in production requires reading `response.usage.cache_read_input_tokens`, which the adapter does not yet capture.

## D-028: Send selected facts, not the profile, to the generator; mark fallbacks explicitly

**Status:** Accepted

Live server logs showed two "summarize experience" turns taking 33.1 s and 30.8 s, both ending in `grounding_status=tool_fallback`: the generator's cached system prefix shipped the entire profile as JSON (about 1,736 tokens) plus history plus the tool result, with `max_tokens=900` and one retry on invalid JSON, so two slow attempts exceeded `MODEL_TIMEOUT_SECONDS=30`. The fallback then rendered canned role/company/team_context text that the user mistook for a real summary. Separately, a Spanish question "platícame sobre sus habilidades" (skills) was classified `summary_request` at 0.90 confidence because the transcript started with "Summarize Marco's experience" — the classifier's full history biased it toward the earlier intent. The summary tool also logged `tool_result_count=0` because `ProfileSummaryPlan` carries `source_ids`, not `matches`, which the count logic didn't check.

Four changes:
1. `ResponseGenerator.generate` (the `orchestrator.py` Protocol and `ClaudeResponseGenerator` implementation) now receives `allowed_facts: list[ResumeFact]` — the canonical facts whose IDs the orchestrator already selected for the turn — instead of the full `Profile`. `allowed_fact_ids` is removed; it is derived from `allowed_facts`. `generation_system_blocks()` no longer takes a profile or embeds one at all; its cached prefix is just the instruction, output schema, and rules. The per-turn payload sends `facts: [{fact_id, source_id, text}]` and `history` truncated to the last 4 items. `max_tokens` drops from 900 to 600, and a rule caps selection at 8 fact IDs under 20 words each. `profile_prompt_payload` is deleted (facts are catalog-derived and never contain phone/email, so the guarantee it provided is now structural). The retry loop skips its second attempt when the first attempt's wall time (via `time.monotonic()`) exceeds half of `Settings.model_timeout_seconds`, since a second attempt then cannot finish before the client-side timeout fires anyway.
2. `ClaudeIntentClassifier.classify` sends only the last 2 history items as `recent_history` (renamed from `history`), with `message` placed before it in the payload, and an explicit rule that a new topic in `message` overrides any earlier intent. A `hints.profile_field_synonyms` block maps EN/ES vocabulary (skills/habilidades/tecnologías/stack, languages/idiomas, education/estudios, companies/empresas, current role/puesto actual) to the typed `profile_field` values.
3. As a deterministic backstop, `AgentService._execute_tool` reroutes a `SUMMARY_REQUEST` to `query_profile` with the matching field whenever the message contains an unmistakable skills/languages/education marker ("skill", "habilidad", "tecnolog", "stack", "idioma", "language", "educa", "estudios", "degree"), regardless of what the classifier returned.
4. Both fallback-to-verified-facts paths — `_tool_fallback_response` (generation unavailable) and the not-fully-grounded branch of `respond` when it falls through to `_verified_tool_facts` — now prefix the answer with a bilingual notice on its own line ("I couldn't compose a written answer right now, so here are the verified profile facts instead:" / the Spanish equivalent) so a fallback is never mistaken for a generated answer. `_fact_selection_response` (the normal fact-ID rendering path) is untouched. `ProfileSummaryPlan` fallbacks render each fact as a `- ` bullet line, and `tool_result_count` for `ProfileSummaryPlan` is now `len(tool_result.source_ids)` instead of the previously always-zero `matches` lookup.

**Why:** The generator cannot time out on content it never receives, and it cannot cite a fact outside the turn's authorized selection if that selection is the only data it was given — this is a stronger boundary than the prior "profile is present but `allowed_fact_ids` restricts citation" contract. Trimming classifier history stops old intents from leaking into new topics. The visible notice turns a previously silent, deceptive fallback into an honest one.

**Consequence:** The generator can now only ever cite facts the orchestrator selected for that turn; any fact_id, source_id pairing check in `grounding.py` remains unchanged and still authoritative. Every `ResponseGenerator` implementation (including test doubles) must accept `allowed_facts` instead of `profile`/`allowed_fact_ids`. Existing tests asserting exact fallback-answer text had to be updated to expect the notice prefix. `generation_system_blocks()` no longer needs — and must never regain — a profile parameter; if per-turn fact content were ever moved into the cached prefix, the cache would silently stop matching across turns.

**Update:** A live call against `claude-sonnet-4-6` with 40 facts showed the generator still failing every summary/skills turn: `stop_reason="max_tokens"` on both attempts at `max_tokens=600`, and the raw output mirrored the prompt's `output_schema` key back as a wrapper object (`{"output_schema": {"text": ..., "claims": [...]}}`) around long prose and verbose claims, so the JSON was unterminated and every attempt raised `InvalidStructuredOutputError` with no record of why. Three fixes: the prompt's schema key is renamed to `response_format` with an explicit "return one top-level object, do not wrap it" rule (also applied to the classifier and rephraser prompts for consistency), plus hard brevity caps (`text` ≤ 40 words, ≤ 6 claims, each claim `text` ≤ 12 words, `evidence` omitted) and `max_tokens` restored to 900; a defensive unwrap in the shared structured-output parser accepts a single-key wrapper object transparently; and a `stop_reason == "max_tokens"` response is refused immediately as `InvalidStructuredOutputError("... truncated")` without spending a second attempt, since a same-prompt retry would very likely truncate again. Separately, every fallback path now records a `fallback_reason` code (`classifier_unavailable`, `classifier_invalid_output`, `generator_unavailable`, `generator_invalid_output`, `generator_truncated`) on `AgentTrace`/`TurnLogEvent` and emits a content-free `generation_fallback` warning log, so a fallback answer is diagnosable instead of silent.

## D-029: Deliver rephrased answers only through a deterministic containment gate; canonical bilingual narratives are the floor

**Status:** Accepted

Every experience highlight, project highlight, experience record, project record, and education record now carries an optional human-reviewed `narrative` (`en`/`es`) restating only its existing `summary`/`detail` content. (A) These canonical bilingual narratives are the floor: `_render_resume_result` and `fact_display_text` render them whenever present, falling back to raw fact text otherwise — this is unconditional and needs no model call. (B) A model rephrase of the turn's already-selected facts (`ClaudeRephraser`, via the new `Rephraser` port) is delivered instead of (A) only when it passes `verify_rephrase` (`src/agent/rephrase.py`), a deterministic gate run in the `fact_rendered` path of `AgentService._fact_selection_response`, after generation and before the output guard.

The gate runs six checks in order, first failure wins: (1) empty/whitespace text; (2) an escalation-vocabulary token (led, managed, senior, director, lideró, gestionó, etc., EN/ES) appearing in the rephrase but absent from the selected facts' own vocabulary; (3) an entity-like token (capitalized mid-sentence, containing a digit, or containing `.`/`#`/`+`) absent from the selected facts' vocabulary and absent from every other catalog fact too (`foreign_vocabulary`); (4) the same but present in some other, unselected catalog fact (`leaked_fact` — distinguishes "not in the profile" from "true but not authorized for this turn"); (5) a selected fact whose canonical text opens with a mapped verb (built, implemented, assisted, collaborated, integrated) must have that verb's meaning preserved in the rephrase, checked against a bounded EN/ES synonym set, never an upgraded one; (6) sentence and word budgets scaled to the number of selected facts. Any failure keeps the (A) rendering and records `rephrase_outcome = "rejected:<code>"`; a provider outage during rephrase records `"unavailable"`; success records `"accepted"` and sets `grounding_status = "rephrased"` instead of `"fact_rendered"`. `AgentTrace.rephrase_outcome` and `TurnLogEvent.rephrase_outcome` carry this without ever logging the rejected text itself. `Settings.rephrase_enabled` (default `True`) lets the rephraser be wired off entirely.

**Why:** Selected-fact citation (D-021) already stops the model from inventing new information, but it does not stop paraphrase from *inflating* real information — "assisted senior engineers" becoming "led the team" cites the same fact ID while changing what actually happened. A single generation call cannot self-certify this; a second, independent, deterministic pass over vocabulary and verb meaning is required, and it must run on the model's actual output text, not on its claimed sources.

**Consequence:** The gate cannot detect nuance shifts that stay inside the selected vocabulary and verb frames (e.g., "helped coordinate" → "coordinated"); the fixed paraphrase evaluation scenarios and a human read of sampled live paraphrases before release cover that residual.

## D-030: Render lists deterministically and skip fact selection when a tool already selected few facts

**Status:** Accepted

Live server logs showed "Está bien, platícame sobre sus habilidades" (skills) routed to `query_profile`, then hit `generator_truncated` at `max_tokens=900` after 14.3 s, and fell back to a raw 34-line skills list anyway — the generation step was never able to add anything to a list the tool had already fully assembled. Separately, ordinary turns were taking 10–20 s because every one made three sequential model calls (classify → generate fact selection → rephrase), even when the tool's own result already narrowed the answer to a handful of facts the generator could only reorder or subset.

Two changes to `AgentService.respond` in `src/agent/orchestrator.py`:
1. When `tool_result` is a `ProfileQueryResult` for `skills`, `languages`, `education`, `current_role`, or `companies`, the answer is rendered deterministically by `_list_rendered_response` — no generator or rephraser call. Skills are grouped by `Skills` model category with bilingual labels ("Programming languages"/"Lenguajes de programación", etc.), one line per category. Education and current-role lines use each record's existing bilingual `narrative` when present, else the plain value string; languages and companies use the plain values. `grounding_status` is `list_rendered` and `claim_source_ids` come directly from the tool result's `source_ids`.
2. For every other tool result except `ProfileSummaryPlan` (which still needs the generator to pick ~6 of ~40 candidate facts), `_tool_ordered_fact_ids` computes the tool's own fact selection in its original order. When a `Rephraser` is configured and that selection has 1–8 facts, the generator call is skipped entirely and `_fact_selection_response` (refactored to take `ordered_fact_ids: list[str]` directly, shared with the existing post-generation path) renders those facts, still gated by `verify_rephrase` (D-029) before any rephrase is delivered. `AgentTrace.generator_skipped` (also added to `TurnLogEvent`, wired through `src/api/chat.py`) records whether this path fired.

A third, independent fix: the rephraser's `GenerationUnavailableError` handling in `_fact_selection_response` now derives a specific `rephraser_unavailable` / `rephraser_invalid_output` / `rephraser_truncated` reason via `_fallback_reason_for` (extended to accept a `"rephraser"` stage) instead of the generic `"unavailable"` string, and emits the same content-free `generation_fallback` warning the classifier/generator stages already emit. `ClaudeRephraser.rephrase` now also calls `_raise_if_truncated`, so a `max_tokens` stop is refused immediately instead of failing structured-output validation with no diagnosable reason.

**Why:** A tool-selected list or a filter result of a handful of facts is already the answer; sending it to the generator only risks truncation, adds 5–10 s of latency, and (per D-028's contract) can only reorder or subset what the tool chose. Skipping it caps every non-summary turn at two model calls (classify + rephrase) instead of three. The rephrase-failure fix closes the one remaining silent fallback: a Spanish summary turn had previously logged `rephrase_outcome="unavailable"` with no way to tell why.

**Consequence:** `_selected_fact_ids`/`_allowed_facts` (used to bound the generator when it does run) are unaffected; `_tool_ordered_fact_ids` is a separate, order-preserving view used only for the skip decision and its rendering. Existing tests that combined a small tool result (≤ 8 facts) with a configured rephraser now exercise the skip path instead of their double's `generate()` — `test_faithful_rephrase_is_accepted_and_delivered`, `test_escalating_rephrase_is_rejected_and_falls_back_to_canonical_rendering`, and `test_rephraser_outage_falls_back_to_canonical_rendering` were updated accordingly, as were the two employer-history tests that previously asserted a generator-fallback answer for what is now a `list_rendered` field. `ProfileSummaryPlan` is deliberately excluded from the skip so summaries keep using the generator's fact-selection step.

## D-031: Select exactly matched facts, prefer project search for project questions, and choose summary facts deterministically

**Status:** Accepted

Live evidence surfaced three residual defects on top of D-029/D-030: (1) "What security-related work has Marco done?" matched two `filter_experience` highlights, but `_selected_fact_ids`/`_tool_ordered_fact_ids` expanded each matched highlight source_id to its parent experience fact too, via a bidirectional prefix check (`source_id.startswith(f"{fact.source_id}.")` matches a highlight source against its parent's exact source_id). The rephrase and canonical-rendering paths then opened with the job-description sentence (`role at company. team_context`) before the actual security facts. (2) "Which projects used AI?" was classified `filter_request(technology=AI)`, which matched an unrelated experience highlight (`hl-isv-module`, technologies `"AI agents"`) before `_execute_tool` ever considered project search — an explicit project question answered with an employment fact instead of the Sybil project. `_project_fallback_terms` also silently dropped its own `"ai"` token check for "AI?" (trailing punctuation defeated a plain `str.split()`). (3) Summary turns cost three sequential model calls (classify ~3 s, generator fact-selection ~8 s, rephrase ~5 s) because `ProfileSummaryPlan` carried only `source_ids`, forcing every summary through the generator even though its fact selection is knowable ahead of time; the provider-outage fallback then rendered an ad-hoc `role at company` / `team_context` pair instead of the profile's own reviewed narrative.

Three changes, all in `src/tools/profile_tools.py` and `src/agent/orchestrator.py`:
1. `_selected_fact_ids` and `_tool_ordered_fact_ids` now include a catalog fact only when its `source_id` **exactly equals** a matched `source_id` — no more `startswith` prefix checks in either direction. A highlight match authorizes only that highlight's fact; a base project/experience match already carries its own exact source_id, so it is unaffected.
2. `_execute_tool` checks `_is_explicit_project_question(message)` **before** building the `filter_experience` plan whenever the intent is `FILTER_REQUEST` (or `SEARCH_QUERY` with a filter plan): it tries `search_projects` first via `_search_projects_with_fallback`, and only falls through to the existing filter/fallback behavior when that search finds nothing. `_project_fallback_terms` now tokenizes with `re.findall(r"[a-z0-9]+", ...)` instead of `str.split()`, so `"AI?"` still yields `"AI"`, and it recognizes `"rag"`, `"llm"`, `"retrieval"`, and `"data platform"`/`"data platforms"` as additional bounded terms.
3. `ProfileSummaryPlan` gained `fact_ids: list[str]` (default empty, ordered, ≤ 8), filled deterministically by `summarize_profile`: every audience starts with each experience's base fact; `recruiter` adds each experience's first 3 highlights then every education base fact; `technical` adds each experience's first 2 highlights then every project's base fact plus its first 2 highlights; `executive` adds the current experience's base fact and first 2 highlights, then education. `AgentService` no longer excludes `ProfileSummaryPlan` from the ≤8-facts/rephraser generator-skip path (`_selected_fact_ids`/`_tool_ordered_fact_ids` now read `plan.fact_ids` directly for it), and `_verified_tool_facts`'s `ProfileSummaryPlan` branch renders each selected fact via `fact_display_text` (bilingual narrative when present) instead of the old `role at company` / `team_context` pair.

**Why:** A highlight match is a specific, narrow claim; presenting its parent job description first weakens the answer's precision and slows the rephrase gate. An explicit project question must never be satisfiable by an unrelated employment fact just because a technology token happens to co-occur in a highlight's tag list. Summaries are a small, fixed selection once source order is fixed, so making that selection deterministic removes one full model round-trip and gives every fallback path an already-reviewed narrative to fall back on.

**Consequence:** Summaries now complete in two model calls (classify + rephrase) instead of three, matching the D-030 pattern already used for filter/search results — a summary turn is no longer 40–60% slower than a filter turn as observed in the live evidence. `ProfileSummaryPlan.fact_ids` is a new required contract for any caller building the plan directly (tests updated to use `summarize_profile`'s own output rather than hand-writing `source_ids`). A project question with genuine ambiguity between an employment fact and a project (both plausibly true) always resolves to the project, which is correct for this profile (one experience, one project) but should be revisited if a future profile makes both readings equally likely.

## D-032: Route specific-technology, history follow-up, and tagged questions deterministically

**Status:** Accepted

Live evaluation surfaced three further defects: (1) "Has Marco worked with FAISS?" classified `direct_question(profile_field=skills)` and rendered the whole skills list instead of the one project highlight (`sybil-hl-hybrid`) that actually names FAISS — naming one specific technology deserves the facts that mention it, not a category dump. (2) "What technologies did you use for that?" with prior turn "Tell me about Sybil." (no client-carried `ConversationState`) also fell through to the skills list, because the existing follow-up resolver only reads verified server-side state and had no way to resolve "that" from the raw chat history the client actually sent. (3) "Tell me about Marco's security work in your own words." classified `summary_request` and rendered the full experience narrative — including the unrelated `team_context` sentence ("assisted senior engineers") — even though the message names the `security` tag explicitly, which `filter_experience` already knows how to answer precisely.

Three additions to `src/agent/orchestrator.py` (`AgentService`), all built from `build_resume_fact_catalog`/`highlight.tags` — no hardcoded technology or tag values (D-020):
1. `_mentions_project_or_experience_technology` + `_technology_search_result`: before the `query_profile` dispatch, a `DIRECT_QUESTION`/`FOLLOW_UP` with `profile_field` in `{None, "skills"}` is checked against every keyword cataloged under a `"projects"`/`"experience"` fact; on a hit, `search_resume` is tried with `topic=None` then explicitly `"projects"` then `"experience"` (the generic topic detector can pick an unrelated topic from a verb like "worked" before ever looking at the technology token), and the first non-empty result wins. No match at any topic falls through to the prior behavior unchanged.
2. `_history_entity_plan`, called from `_follow_up_plan` only when `state` is `None` or has no single verified entity: the follow-up phrase list gained "for that"/"on that"/"that one"/"in that"/"con eso"/"en ese"/"para eso"/"de eso", and when one of these fires, the last 4 history turns (user or assistant) are scanned via whole-word match for a catalog entity name (excluding Marco's own name). Exactly one distinct entity found builds `SearchResumeArguments(query=f"{entity} {message}", topic=None)`, or — if the message also names a technology/stack marker — `query=entity, topic=<entity's own topic>`; zero or multiple candidates still fall back to "clarify".
3. `_profile_tag_match`: reads `experience[].highlights[].tags` directly (not the mixed technology/tag keyword bag, to avoid a technology word masquerading as a tag) and, when the message names one of those tags verbatim, reroutes a `SUMMARY_REQUEST` — or a `FILTER_REQUEST` that lacks its own filter plan — to `filter_experience(filter_by="tag", value=<tag>)` ahead of the existing `_summary_field_override`/default-"profile" paths.

Separately, `eval/run_eval.py`'s required/forbidden token check changed from a plain case-insensitive substring test to a whole-word/phrase regex (`token_present`, `\b`-bounded) — the substring form could pass or fail a token based on its being embedded in an unrelated longer word.

**Why:** The classifier's structured output is advisory, not authoritative — deterministic, profile-derived signals in the message itself (a named technology, a recently mentioned entity, an existing tag) are strictly more reliable than a category-level intent label, and D-020 already establishes that pattern for other misrouted intents.

**Consequence:** A question naming one profile technology, tag, or a single recently mentioned entity now bypasses the classifier's coarse field/intent guess entirely; ambiguous or multi-entity history still fails closed to "clarify" rather than guessing. `eval/scenarios.json`'s `faiss`/`follow-up`/`paraphrase-security` cases and the new `tests/test_eval_runner.py` token-matcher tests cover the regression.

## D-033: Plan precise direct answers before model-controlled projection

**Status:** Accepted

Every classified in-scope turn now has an internal typed `AnswerPlan`: exactly one `direct` or `synthesis` mode, plus topic, scope, requested field, response language, and the selected canonical fact/source IDs. Explicit bounded questions in the current message are planned before classifier-controlled tool projection. The deterministic planner covers partial employment dates and current status, named technologies, profile-defined tags, project records, experience records, and named project entities. Explicit list fields remain deterministic `direct` plans; open transformation requests and other tool-selected flows remain `synthesis` plans.

Employment `start_date`, `end_date`, and `current` are separate field facts in the resume catalog. Their renderer parses only `YYYY` and `YYYY-MM`, uses local English/Spanish month tables, and never invents a day. Once a direct plan selects facts, `DirectAnswerRenderer` renders their reviewed canonical narratives (or the employment-field sentence) without calling either generator or rephraser. A classifier outage can therefore still return HTTP 200 when those facts suffice. The internal trace records plan and rendering modes plus the selected IDs; operational logs record the modes and selection counts without exposing answer content.

**Why:** A classifier's `profile_field=companies` is a coarse suggestion, not authorization to replace explicit evidence such as “FAISS,” “security-related,” or “experiencia.” Likewise, asking for a known date or record does not need probabilistic prose. Planning first makes the smallest sufficient fact set an architectural boundary instead of a prompt preference.

**Consequence:** Employer projection is used only for explicit employer/company/history wording. Verified history can resolve an otherwise ambiguous pronoun, but an explicit current topic bypasses history-derived routing. Direct answers no longer exercise rephrase behavior; tests for rephrase and grounding use explicit synthesis wording so the two modes remain independently covered. No public request/response schema changed.

**Verification update:** Direct-topic recognition operates only on canonical normalized tokens (`projects`/`proyectos` → `project`, `seguridad` → `security`) so equivalent EN/ES questions select the same facts. Explicit synthesis markers are checked before every direct branch, including temporal and named-entity routing. Deterministic clarification and not-found responses now carry an empty typed direct plan rather than leaving `answer_mode` unset, while out-of-scope/guardrail blocks remain outside the answer-mode contract. Rephraser provider/structured-output failures are recorded in both `rephrase_outcome` and `fallback_reason` because canonical rendering was used as the fallback.

**Architecture update:** `src/agent/answer_planning.py` owns two profile-scoped, provider-free collaborators. `AnswerPlanner` resolves current-message precedence, direct versus synthesis mode, topic/scope/requested field, boundary plans, and canonical fact/source selection. `DirectAnswerRenderer` converts an already-authorized direct plan into deterministic English/Spanish text, including partial employment dates and current/missing status. `AgentService` retains tool execution, provider coordination, history, grounding, guardrails, trace/DTO assembly, and conversation-state updates. The planner derives ordered fact IDs only from typed tool results and the canonical profile catalog instead of depending on `AgentService`; neither extracted collaborator can call a provider or import the orchestrator. This keeps `AnswerPlan` in the contract layer and prevents a circular dependency.

## D-034: Bound synthesis by dimension, selected evidence, and a fixed delivery budget

**Status:** Accepted

Explicit summary, impact, significance, comparison, explanation, and conclusion/own-words requests now carry a `synthesis_dimension` on `AnswerPlan`. `AnswerPlanner` resolves the dimension and response topic from normalized current-message evidence, then ranks at most four canonical facts for that dimension. Broad experience and project summaries use representative record/highlight facts; impact plans retain only facts whose canonical text states an explicit outcome; narrower filter/search results remain the boundary for significance, comparison, explanation, and conclusion transformations. Fact IDs are deduplicated and semantically overlapping candidates are collapsed before any provider sees them. Equivalent English/Spanish summary and impact requests therefore select the same canonical IDs.

`AgentService` sends only the plan's selected facts to the configured transformation provider; its tool payload is projected to the same selection while preserving the original typed tool-result contract. `ClaudeRephraser` returns a typed `SynthesisTransformation`: every proposition carries one or more selected `fact_ids`, unclaimed top-level prose is rejected, and each proposition is checked against only the facts it cites before the whole answer is checked against the plan. The same per-claim citation boundary applies when synthesis uses the general generator. `verify_synthesis_text` preserves the existing entity-leakage, responsibility-inflation, and verb-drift checks while adding lexical closure over cited facts, a non-scaling maximum of three sentences and 75 words, request-language enforcement, and rejection of unsupported business-outcome concepts. A structural gate rejects one-sentence-per-fact dumps and, unless detail was explicitly requested, conclusions with more than one supporting example. The provider prompt now requires aggregation/compression instead of one sentence per fact. A provider outage, invalid structured output, missing fact-ID proposition mapping, unclaimed prose, or deterministic verification rejection returns HTTP-success-compatible canonical content through `SynthesisFallbackRenderer`; that renderer chooses one representative reviewed fact rather than dumping every selected narrative. The public API schema is unchanged.

**Why:** The prior fact-scaled rephrase budget and one-sentence-per-fact prompt made a five-fact selection become a five-paragraph answer. Selection grounding prevented unrelated IDs but did not ensure the prose answered the requested dimension, and the fallback repeated all selected narratives. Planning the transformation boundary before provider access makes concision and evidence scope enforceable properties rather than prompt preferences.

**Consequence:** Synthesis traces and content-free turn logs record the dimension, `transformed` versus `canonical_fallback`, transformation outcome/reason, selected IDs/counts, and final word/sentence counts. Legacy generator responses that cite only source IDs are no longer accepted as synthesis: every factual proposition must carry selected fact IDs or the service uses the canonical fallback. The focused contract covers five required EN/ES examples and one shared outage/rejection boundary; Issue #3 retains cross-scenario evaluation and the real UI release gate.

## D-035: Announce a fallback only when it renders a list of facts

**Status:** Accepted (amends D-024)

`_FALLBACK_NOTICE` is no longer prefixed to the synthesis canonical fallback
(`AgentService._synthesis_response`). It remains on the two multi-fact fallback paths:
`_fact_selection_response`'s ungrounded branch and `_tool_fallback_response`.

The synthesis fallback renders exactly one human-reviewed narrative through
`SynthesisFallbackRenderer`, bounded to three sentences and 75 words. Removing the
notice also returns those words to the body, which the notice's own length had been
subtracted from. The other two paths join several facts — a `- ` bullet list for
`ProfileSummaryPlan`, blank-line-separated records otherwise — and keep the notice.
`rendering_mode` still records `canonical_fallback`, and `transformation_outcome`
still records why, so no diagnostic information is lost.

**Why:** D-024 added the notice after a live turn in which canned
`role at company` / `team_context` text was mistaken for a real summary. D-029 then
made human-reviewed bilingual narratives the rendering floor, so the synthesis
fallback no longer emits the kind of output that motivated the warning: it emits
prose. Prefixing an apology to a correct, idiomatic answer misreports it as a
failure, and did so on every Spanish summary and impact question, where synthesis
falls back more often than in English. A list of facts still reads as raw material
rather than an answer, so D-024's reasoning continues to hold there.

**Consequence:** In practice the notice now appears rarely, because the synthesis
planner intercepts most requests that previously reached the multi-fact paths. That
is intended: the notice marks a specific output shape, not a fallback in general.
Bilingual boundary coverage is unaffected — the out-of-scope redirect, clarification,
not-found, input-guard, and output-guard replies remain bilingual and tested.

## D-036: Buy evidence breadth in words, not facts

**Status:** Accepted (amends D-034)

`MAX_SYNTHESIS_FACTS_BY_LANGUAGE` sets synthesis evidence breadth per response
language: three facts in English, two in Spanish. Both languages rank the same
candidate facts by the same dimension-relevance score; Spanish takes a shorter prefix
of that identical ranking. It never selects a fact English would not have selected,
never reorders, and never changes topic, scope, or requested field.

**Why:** Spanish states the same content in materially more words than English, so a
single shared 75-word budget buys fewer facts. Holding the fact count equal across
languages did not produce equal answers — it produced a Spanish answer that was
rejected as `too_long` and fell back to one canonical narrative, which is *less*
evidence than the two facts it can now actually deliver. Selecting two facts and
delivering them beats selecting three and delivering one.

**Consequence:** This knowingly departs from Issue #2's "equivalent English and
Spanish synthesis requests resolve to equivalent fact scopes" read as identical fact
sets. The weaker guarantee that replaces it is prefix equivalence: same dimension,
topic, scope, requested field, and ranking, with Spanish truncated one earlier. Two
tests were rewritten to assert that contract rather than set equality. Issue #3's
parity gate must be written against prefix equivalence, or this decision revisited.
`MAX_SYNTHESIS_PROPOSITIONS` (2) no longer sits strictly below the Spanish fact limit,
so a Spanish answer may map one proposition per fact; the `fact_dump` gate still
requires three such propositions and remains unreachable at either limit.

Live provider gate after this change: all five required Issue #1 examples render
deterministically, and all five required Issue #2 examples deliver transformed
synthesis within the 3-sentence/75-word budget in both languages.

## D-037: Make the scenario matrix the enforceable answer contract

**Status:** Accepted (supersedes the evaluation behavior of D-029 and D-030)

`eval/scenarios.json` becomes the single source of truth for automated evaluation and
the manual UI checklist, and `eval/run_eval.py` enforces every field it declares.

Previously the runner treated `expected_source_ids` as a subset, so an answer that
selected the right facts plus unrelated extras passed; `tool_required` proved only that
some tool ran, not the right one on the right topic; and `inference_permitted` was
parsed and never checked. Scope is now exact in both directions, tool checks assert the
named tool, and inference policy is enforced through `rendering_mode`.

Enforcing inference through the rendering mode rather than a claim-kind count reflects
what the answer plan actually made true. After D-033 and D-034 every delivered answer is
either reconstructed from canonical facts (`canonical`, `canonical_fallback`,
`canonical_not_found`, `clarification`) or a contained transformation of a selected fact
set (`transformed`). The legacy free-generation path that D-029 and D-030 were written
around is no longer reachable through `respond`: `"generated"` appears at one production
site and in no test, and no route produces it. Only `transformed` carries wording the
provider chose, so it is the only mode an inference can reach. A guardrail block or
out-of-scope redirect never reaches the answer plan and therefore has no rendering mode.

Provider failure is reproduced by substituting a failing stage rather than waiting for a
real outage, and provider calls are counted through wrapped stages, so "a direct answer
makes zero generation calls" is verified rather than assumed. Both are deterministic and
run without network access.

**Why:** Automated evaluation passed while the live UI produced broader answers, verbose
concatenations, and provider-related failures. A subset check cannot catch an extra fact,
and a boolean that is read but never compared cannot catch anything. The matrix is only
evidence if the checker can fail, so `tests/test_eval_contract.py` feeds it responses
that violate one expectation each and asserts the violation is reported.

**Consequence:** The matrix fixes thirty scenarios: ten paired English/Spanish
behaviors, five incompatible classifier doubles, and the five retained guardrail cases.
Parity is asserted as prefix equivalence per D-036, not set equality. Building the matrix
surfaced one live defect: Spanish summary recognition keyed on the fixed phrase
`"resume la"`, so "Resume los proyectos de Marco" was not routed to synthesis at all and
selected five unrelated experience and education facts. Spanish now matches the bare
imperative verb, gated on detected language so the English noun "resume" never triggers
it. The full matrix passes against the configured provider.

## D-038: Answer every turn at HTTP 200, and name entities from one detector only

**Status:** Accepted (closes the last classifier-failure raise left by D-020)

Three rules, all reached through the same observed turn — a user wrote *"The part where
it says he worked at google"* and received no assistant response at all.

**A turn never ends in a raise.** `AgentService._respond` re-raised
`GenerationUnavailableError` when the classifier failed twice and no local recovery
matched, which `POST /api/chat` renders as HTTP 503 and the frontend as nothing. The
classifier only ever chose a tool; it never chose facts. So an anchor evidenced in the
message (`detect_resume_topic`) now drives `search_resume` directly, and a message with
no evidenced anchor gets the same deterministic clarification any unresolvable turn gets.
Failing closed means selecting nothing, not returning nothing.

An unanchored question still clarifies rather than widening to `summary` facts, which is
the substitution `detect_resume_topic` exists to prevent. `"What results has Marco
produced?"` is the one release-adjacent phrasing in that position: it is answered
normally, and clarifies only while the classifier is down.

**A falsified antecedent is answered, not guessed at.** A phrase that asserts what was
already said and carries its antecedent inline (`the part where it says …`, `cuando
dijiste …`) is checked against the profile before anything else runs. Every answer this
agent produces is assembled from profile facts, so an antecedent matching no fact was
never delivered, and the honest answer says so. A bare referent (`that bit`) names
nothing checkable and stays on the clarification path. Accumulated `delivered_fact_ids`
(#12) will let this distinguish *never said* from *not in the profile* per conversation;
until then the profile-wide check is the stronger claim available.

**Query terms are not entities.** `ResumeSearchResult.unmatched_terms` holds raw query
terms that matched no fact, and rendering it produced *"I couldn't find anything about
**his** in Marco's profile"* and *"No encontré nada sobre **resume, marco.**"*. It is now
a retrieval diagnostic only; `find_unknown_entities` is the sole source of a named entity
in any answer. Pronouns, discourse verbs, and summarize verbs joined `_STOP_WORDS`, and
`normalize_resume_text` trims edge dots so `marco.` and `marco` are one token — which
also fixes the retrieval failure underneath the wording, since a stocked topic no longer
looks empty because the question contained *his*.

**Why:** every other boundary in this system degrades to a deterministic answer at
HTTP 200; one path did not, and it was reachable from ordinary phrasing.

**Consequence:** `_is_title_case` now measures capitalization after the opening word,
which is capitalized by orthography in every sentence and made short questions look
title-cased purely for being short. `Did Marco work at Google?` reached the 0.6 ratio and
skipped entity detection entirely, while `Tell me about Marco's experience at Google.`
did not — the same entity, two verdicts. Both now return the same not-found naming
Google, and genuine title-case prose is still skipped.
