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

## D-005: Minimize public contact-data disclosure

**Status:** Accepted

The agent never discloses Marco's phone number. It discloses the professional email only after an explicit request for contact details.

**Why:** A public chatbot should not make automated contact-data harvesting effortless.

**Consequence:** `query_profile` must not return unrestricted contact data, and logs must exclude contact information.

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

**Status:** Accepted

Fact IDs and matching source IDs authorize selection and order only. They do not validate, entail, or authorize arbitrary generated English or Spanish claim text. For fact-ID responses, the delivery boundary ignores provider prose and reconstructs public text from canonical selected `ResumeFact` values using bounded English/Spanish templates. The verifier does not use fuzzy similarity, semantic entailment, or a second model judge. Provider prose remains available only for direct claims through the exact existing-English evidence compatibility path; inferred provider prose is rejected because exact excerpts cannot prove a synthesis.

**Why:** English substring matching rejects faithful Spanish, but citation identity and string similarity also cannot prove factual entailment. Stable typed identity safely authorizes which canonical facts may be rendered—not what a provider may say about them.

**Consequence:** Provider prompts receive only the turn's allowed fact IDs plus valid source IDs as selection hints. Public wording comes from canonical facts and deterministic templates. Privacy and contact-output guards still run after rendering.

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
