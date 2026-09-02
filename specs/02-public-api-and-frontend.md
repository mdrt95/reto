# 02 — Public API and Frontend

## Goal

Expose the core agent through a stable first-party HTTP contract and a minimal static chat interface. This phase does not implement A2A, MCP, or a provider-specific Responses adapter.

## `POST /api/chat` contract

### Request

```json
{
  "message": "What security-related work has Marco done?",
  "history": [
    { "role": "user", "content": "Tell me about Marco's work." },
    { "role": "assistant", "content": "..." }
  ],
  "preferences": { "language": "en", "verbosity": "concise" },
  "state": null
}
```

`message` is required, non-blank, and limited by `MAX_INPUT_CHARS`. `history` is optional, oldest-first, has only `user` or `assistant` roles, and is capped by `MAX_HISTORY_MESSAGES` and total character limit. `preferences` is optional and allowlisted to `language` and `verbosity`. `state` is an optional client-carried object with verified topic/source/entity/tool/language fields; all other fields are rejected.

### Success response

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
    "last_tool": "search_resume",
    "response_language": "en"
  }
}
```

The response never includes hidden prompts, raw tool results, raw provider output, phone number, or grounding internals. A later authenticated diagnostics endpoint may expose sanitized metadata; it is not part of v1.

### Error contract

All errors use `application/problem+json`:

```json
{
  "type": "https://example.invalid/problems/invalid-request",
  "title": "Invalid request",
  "status": 422,
  "code": "invalid_request",
  "request_id": "req_<uuid>"
}
```

| Condition | HTTP | Public code |
|---|---:|---|
| Request/history/preference/state validation failed | 422 | `invalid_request` |
| Request exceeds rate limit | 429 | `rate_limited` |
| Model provider is unavailable and no safe deterministic answer exists | 503 | `generation_unavailable` |
| Unexpected failure | 500 | `internal_error` |

Guardrail and out-of-scope outcomes are `200` completed answers with a safe redirect; they are not HTTP errors.

## Frontend

- Serve one responsive `frontend/index.html` from the same origin as the API.
- Include chat bubbles, send button, loading state, suggested prompts, accessible labels, and escaped/allowlisted Markdown rendering.
- Maintain only the bounded transcript and optional verified state needed to make the next request. Do not place API keys or system instructions in browser code.
- Disable sending while a request is in flight; show a friendly retry action for sanitized 429/503/500 responses.

## Security and delivery

- Enforce request-size limits and per-IP rate limits before model execution.
- Use same-origin API calls. If a separate frontend origin is later required, explicitly allow only that origin with CORS.
- Generate request IDs at the HTTP boundary and propagate them through logs and responses.

## Minimal verification

- Valid request returns a completed response from a fake core-agent service.
- Empty message and invalid history return the documented 422 problem response.
- A guardrail block returns a safe 200 answer.
- The frontend can send one message, render a response, and display a recoverable error state.

## Completion gate

The local browser flow works against the first-party contract, and focused API tests verify the success, validation, and sanitized-error contracts. Stop safely here before deployment.
