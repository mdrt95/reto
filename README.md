# Banorte CV Agent

A grounded, public-facing assistant for Marco Reyes' professional profile, built for the Reto IA Banorte challenge.

The service answers from one validated local profile, searches a deterministic in-memory fact catalog, and uses Claude only inside bounded classification, structured-generation, and rephrasing steps. Selected fact/source IDs authorize canonical facts; deterministic grounding, containment, privacy, and output checks decide what can be returned.

## Current capabilities

- Answers English and Spanish questions about experience, projects, skills, education, languages, professional summary, and explicitly recorded career preferences.
- Supports direct questions, normalized search/filter requests, cross-topic synthesis, and progressive follow-ups.
- Returns a deterministic not-found response for named entities absent from the profile instead of substituting unrelated facts.
- Blocks known prompt-injection, out-of-scope, and contact-data requests before any model call.
- Never exposes the phone number or email stored in the private source profile.
- Falls back to canonical, verified wording when classification, rephrasing, or containment checks fail and a deterministic answer can still be proven.
- Provides a built-in static chat UI, a strict first-party REST API, and an authenticated OpenAI Responses-compatible adapter with JSON and SSE responses.

## Architecture

The application is a modular monolith with ports-and-adapters boundaries:

```text
browser ───────────────→ POST /api/chat ───────┐
OpenAI-style client ──→ POST /v1/responses ───┤
                                                ↓
request validation → input/privacy guard → unknown-entity check
→ deterministic answer plan and/or bounded Claude classification
→ one typed, read-only profile tool → canonical fact selection
→ optional bounded Claude rephrase → deterministic containment/grounding
→ output/privacy guard → protocol-specific response
```

`data/profile.json` is the sole runtime source of biographical claims. Pydantic validates it at startup, then the application derives its fact catalog and normalized search index in memory. Tools are typed, allowlisted, read-only, and bounded; neither the model nor the HTTP client chooses arbitrary code or data access.

Conversation data is client-owned and ephemeral. The service stores no transcript database. `/api/chat` accepts a bounded transcript plus compact verified state; `/v1/responses` takes history from message items resent in the current request and, for `previous_response_id`, resolves compact verified state from a bounded, TTL-limited, process-local snapshot that holds catalog IDs and enum values only — no message or answer text. Logs contain correlation and decision metadata, not prompts, answers, provider payloads, API keys, email addresses, or phone numbers.

There is intentionally no vector database, embedding pipeline, document chunking, durable memory, write-capable tool, A2A endpoint, MCP server, or Chat Completions endpoint.

## Run locally

Requirements: Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --requirement requirements.txt
```

Create a local `.env` file yourself (the repository intentionally does not include one):

```dotenv
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_COMPAT_TOKEN=your_separate_client_token
ENVIRONMENT=development
```

`ANTHROPIC_API_KEY` enables Claude-backed classification and rephrasing. The root UI and `/health` work without it, and some recognized questions still have deterministic answers, but provider-dependent requests can return `503`. Production startup rejects a missing Anthropic key.

`OPENAI_COMPAT_TOKEN` is a separate shared secret invented by the operator. It enables and protects `/v1/responses` and `/v1/models`; it is never the Anthropic key. Omit it if the compatibility adapter should remain disabled.

Start the service:

```bash
uvicorn src.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Never commit `.env` or either secret.

### Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `ANTHROPIC_API_KEY` | unset | Anthropic provider credential; required when `ENVIRONMENT=production` |
| `OPENAI_COMPAT_TOKEN` | unset | Bearer token for both `/v1/responses` and `/v1/models`; unset disables them |
| `MODEL_NAME` | `claude-sonnet-4-6` | Anthropic model used by the core agent |
| `MODEL_TIMEOUT_SECONDS` | `30` | Provider timeout and retry budget input |
| `MAX_INPUT_CHARS` | `12000` | Maximum current message and maximum aggregate history characters |
| `MAX_HISTORY_MESSAGES` | `12` | Maximum prior user/assistant message items |
| `RATE_LIMIT_PER_MINUTE` | `30` | Rolling in-memory limit per client IP, shared by both chat routes |
| `RESPONSES_STATE_TTL_SECONDS` | `1800` | Lifetime of a `/v1/responses` `previous_response_id` state snapshot |
| `RESPONSES_STATE_MAX_ENTRIES` | `1000` | Cap on live snapshots; overflow evicts the oldest |
| `REPHRASE_ENABLED` | `true` | Allows verified model rephrasing; `false` forces canonical rendering |
| `ENVIRONMENT` | `development` | Set to `production` for deployed configuration validation |
| `LOG_LEVEL` | `INFO` | Application log level |
| `PROFILE_PATH` | `data/profile.json` | Validated runtime profile path |

## HTTP interfaces

| Method and path | Authentication | Purpose |
|---|---|---|
| `GET /` | none | Built-in static chat UI |
| `GET /health` | none | Readiness and application version |
| `POST /api/chat` | none | Strict first-party chat contract |
| `POST /v1/responses` | bearer token | OpenAI Responses-compatible adapter |
| `GET /v1/models` | bearer token | Lists the logical model `banorte-cv-agent` |

### First-party API: `POST /api/chat`

Only `message` is required. Unknown fields, invalid roles, blank text, and invalid state are rejected.

```bash
curl --request POST http://127.0.0.1:8000/api/chat \
  --header 'Content-Type: application/json' \
  --data '{
    "message": "Tell me about Sybil.",
    "history": [],
    "preferences": {"language": "en", "verbosity": "concise"}
  }'
```

Successful response:

```json
{
  "id": "req_<uuid>",
  "answer": "...",
  "conversation_id": "conv_<uuid>",
  "status": "completed",
  "state": {
    "last_topic": "projects",
    "last_source_ids": ["project:proj-sybil"],
    "last_entities": ["Sybil"],
    "last_tool": "search_projects",
    "response_language": "en",
    "focus_source_id": "project:proj-sybil",
    "delivered_fact_ids": ["fact:project:proj-sybil"],
    "discussed_topics": ["projects"],
    "discussed_source_ids": ["project:proj-sybil"]
  }
}
```

The exact state values depend on the selected evidence. For a follow-up, resend the returned `state` unchanged and include any desired bounded `history`. State contains verified identifiers and referents, including the accumulated record of delivered facts; it does not contain a server-side session handle or transcript.

Expected first-party errors use sanitized `application/problem+json` bodies with `code` values `invalid_request` (`422`), `rate_limited` (`429`), `generation_unavailable` (`503`), or `internal_error` (`500`). Guardrail blocks and profile not-found outcomes are safe `200` answers, not transport errors.

### OpenAI Responses adapter: `POST /v1/responses`

Set `OPENAI_COMPAT_TOKEN`, then send it as a bearer token. `input` may be a string:

```bash
curl --request POST http://127.0.0.1:8000/v1/responses \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your_separate_client_token' \
  --data '{
    "model": "banorte-cv-agent",
    "input": "What has Marco built?"
  }'
```

Or it may be an array whose final usable message is a `user` turn:

```json
{
  "input": [
    {"role": "user", "content": "Tell me about Sybil."},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": [{"type": "input_text", "text": "Tell me more."}]}
  ]
}
```

Earlier `user` and `assistant` messages become bounded history. Text content parts are joined; non-text parts, non-message items, and `system`/`developer` messages are ignored. The adapter also accepts but ignores `instructions`, sampling parameters, client tool declarations, and other unknown fields. These fields cannot modify server-owned behavior.

#### Continuing a conversation with `previous_response_id`

Every response carries an `id` of the form `resp_` followed by 32 lowercase hex
characters (message items use `msg_` on the same shape). Send that `id` from a
prior turn as `previous_response_id` to continue the conversation without
resending history:

```json
{
  "input": "And his role there?",
  "previous_response_id": "resp_0f1e2d3c4b5a69788796a5b4c3d2e1f0"
}
```

The adapter resolves the ID against a bounded, process-local store that maps it to the compact verified `ConversationState` the core agent produced on that turn. That snapshot holds only verified referents — topic, source and entity IDs, focus, delivered-fact IDs, response language — matching the `state` object `/api/chat` returns. It never holds message or answer text and is not a transcript database.

Continuation contract:

- **Supported:** `previous_response_id` is honored via server-side resolution of verified state. It is not required — resending `user`/`assistant` items in `input` remains a valid way to continue.
- **ID format:** `resp_` + 32 lowercase hex. Every turn mints a new ID; the one you send back is the previous turn's, not a stable session handle.
- **Size limits:** a continuation is bounded exactly like any turn — `MAX_INPUT_CHARS` for the message and aggregate history, `MAX_HISTORY_MESSAGES` for resent items. The resolved snapshot does not count against these; it is already bounded by `ConversationState`'s own field limits.
- **Expiration:** snapshots expire `RESPONSES_STATE_TTL_SECONDS` after they are written (default 1800), are **not** refreshed on read, and the store is capped at `RESPONSES_STATE_MAX_ENTRIES` with oldest-first eviction. All snapshots are lost on process restart and are not shared across instances.
- **Error semantics:** a `previous_response_id` that is unknown, expired, malformed, or minted under a different bearer token fails closed with HTTP `404`, code `previous_response_not_found`, and no provider call. The client-supplied ID is never logged or echoed.
- **Credential binding:** snapshot store keys are namespaced by a non-reversible tag of the presenting bearer credential, so a `resp_*` ID never resolves under a different `OPENAI_COMPAT_TOKEN` (e.g. after a token rotation, or if per-client tokens are added later).
- **Security invariants:** the resolved snapshot is untrusted input like everything else — guardrails, grounding, privacy checks, size limits, and the rate limiter all still run. No client-supplied conversation state is accepted; only an ID the server itself minted resolves, and only to state the server itself verified.

You may resend `user`/`assistant` items in `input` alongside `previous_response_id`; the resent items form the turn's bounded history and the snapshot supplies the verified referents.

A client that cannot retain the `resp_*` ID, or that targets a different instance or a restarted process, should fall back to resending prior `user`/`assistant` message items in `input`.

Worked two-turn exchange:

```jsonc
// turn 1 request
{ "model": "banorte-cv-agent", "input": "Where did Marco work?" }

// turn 1 response (abridged)
{ "id": "resp_0f1e2d3c4b5a69788796a5b4c3d2e1f0", "object": "response",
  "status": "completed", "output_text": "Marco worked at Google.", "error": null }

// turn 2 request — previous_response_id only, no resent history
{ "input": "And his role there?",
  "previous_response_id": "resp_0f1e2d3c4b5a69788796a5b4c3d2e1f0" }

// turn 2 response (abridged) — a new id, grounded using the resolved referents
{ "id": "resp_9a8b7c6d5e4f302118273645540f1e2d", "object": "response",
  "status": "completed", "output_text": "He was a senior engineer there.", "error": null }
```

An unresolvable `previous_response_id` returns, with no provider call:

```json
{ "error": { "message": "Previous response not found or expired. Resend prior turns in 'input'.",
             "type": "invalid_request_error", "param": null, "code": "previous_response_not_found" } }
```

`tests/test_responses_continuity_contract.py` pins this exchange as an executable, provider-free contract.

A non-streaming success returns a Responses-shaped object with one assistant `output_text` message and a duplicate top-level `output_text`. The echoed `model` is the client-supplied string, or `banorte-cv-agent` when omitted. `usage` currently reports zeros because provider token accounting is not exposed through this adapter.

```json
{
  "id": "resp_<uuid>",
  "object": "response",
  "created_at": 1234567890,
  "status": "completed",
  "model": "banorte-cv-agent",
  "output": [{
    "type": "message",
    "id": "msg_<uuid>",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "...", "annotations": []}]
  }],
  "output_text": "...",
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
  "error": null,
  "metadata": {}
}
```

Semantic input errors return `400`, an unresolvable `previous_response_id` returns `404` with code `previous_response_not_found`, invalid/missing bearer tokens return `401`, the disabled adapter or unavailable answer generation returns `503`, and rate limiting returns `429`. Adapter-handled failures use the OpenAI-style `{"error": {...}}` JSON envelope. FastAPI/Pydantic parsing failures still use the application's sanitized `422 application/problem+json` boundary.

#### Streaming

Set `"stream": true` to receive `text/event-stream`:

```bash
curl --no-buffer --request POST http://127.0.0.1:8000/v1/responses \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your_separate_client_token' \
  --data '{"input":"What has Marco built?","stream":true}'
```

The stream emits ordered Responses events from `response.created` through `response.output_text.delta` and `response.completed`, each with an incrementing `sequence_number`. It has no `[DONE]` sentinel; the connection ends after `response.completed`.

This is framed streaming, not provider token streaming. The core agent finishes and verifies the entire answer before the first SSE byte is emitted, then slices the approved answer into deltas. It therefore preserves grounding and returns pre-stream failures as ordinary JSON, but provides no time-to-first-token benefit.

### Model discovery: `GET /v1/models`

```bash
curl http://127.0.0.1:8000/v1/models \
  --header 'Authorization: Bearer your_separate_client_token'
```

The endpoint returns a one-item OpenAI-style model list containing `banorte-cv-agent`. It uses the same adapter token and disabled-by-default behavior.

## Guardrails and grounding

- Client messages, histories, state, OpenAI `instructions`, and tool declarations are untrusted data; server instructions remain authoritative.
- Known prompt-injection phrases, contact requests, and narrow out-of-scope probes are blocked deterministically before model access.
- Unknown named entities fail closed to a profile-not-found answer.
- The model receives only the facts selected for the turn. Structured model output is schema-validated, bounded, mapped back to selected IDs, and checked for unsupported wording and seniority/responsibility inflation.
- A failed model rephrase does not become public text; the agent uses human-reviewed canonical English/Spanish narratives when possible.
- The final output guard independently blocks stored email, phone, and internal-instruction leakage.

These are deliberately bounded controls, not a claim of general prompt-injection detection or universal semantic verification.

## Verify

Run the same offline checks used by GitHub Actions:

```bash
python -m pytest -q
python -m compileall -q src tests eval
python -m eval.run_eval
```

Each fixed evaluation scenario declares an expected outcome (`answer`, `blocked`, `not_found`, or `clarify`). To execute the scenarios against the configured Anthropic model:

```bash
python -m eval.run_eval --execute
```

The credentialed evaluation can incur provider usage. GitHub Actions intentionally runs only offline checks and receives no provider key.

## Deployment

The included `Dockerfile` installs the pinned requirements and starts Uvicorn on `0.0.0.0:${PORT:-8000}` with proxy headers enabled. Render is the documented initial host.

For production configure at least:

```text
ANTHROPIC_API_KEY=<secret>
ENVIRONMENT=production
LOG_LEVEL=INFO
```

Add `OPENAI_COMPAT_TOKEN=<different-secret>` only when an external client needs the Responses adapter. Configure `/health` as the HTTPS readiness check.

The image defaults `FORWARDED_ALLOW_IPS=*` because the documented Render topology places exactly one trusted platform proxy in front of the container. Reassess that setting before deploying behind a different proxy topology; otherwise forwarded client addresses may be spoofable. Rate limiting is process-local and shared across `/api/chat` and `/v1/responses` per client IP. Counters reset on restart and are not coordinated across multiple instances. The `/v1/responses` `previous_response_id` snapshot store is process-local on the same terms: a snapshot does not resolve after a restart or on a different instance, and the client falls back to resending history.

## Honest limitations

- OpenAI compatibility is intentionally limited to `POST /v1/responses`, its documented JSON/SSE subset, and `GET /v1/models`; it is not a complete OpenAI API implementation.
- Client instructions, sampling options, and client-declared tools are ignored. `previous_response_id` is honored but resolves only against a bounded, process-local, TTL-limited snapshot of verified state — never a stored transcript — and never across a restart or a second instance.
- SSE deltas are produced only after full answer verification.
- `/api/chat` has no authentication; deployment protection depends on narrow functionality, privacy controls, input limits, and the in-memory rate limiter.
- No CORS middleware is configured. The built-in same-origin UI works, but a browser client on another origin requires an explicitly allowlisted CORS change.
- No persistent session, distributed rate limiter, cross-instance response store, or multi-instance coordination exists; the `previous_response_id` snapshot store is single-process only.
- Deterministic fallback coverage is intentionally bounded; some provider outages return a sanitized `503`.
- Profile changes require a factual review and validation because `data/profile.json` is production truth.

## Project structure

```text
src/agent/          orchestration, answer planning, provider ports, grounding
src/api/            first-party chat and readiness routes
src/protocol/       external protocol adapters
src/guardrails/     deterministic input and output privacy checks
src/tools/          typed profile queries and in-memory fact search
src/observability/  privacy-preserving structured logging
data/               validated runtime professional profile
frontend/           dependency-free accessible chat interface
tests/              focused unit and contract tests
eval/               fixed evaluation scenarios and runner
specs/              sequential implementation specifications
```
