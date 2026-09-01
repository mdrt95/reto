# 03 — Quality, Operations, and Deployment

## Goal

Prove the implemented behavior, make it observable without retaining content, and deploy a reproducible public demo.

## Focused evaluation set

Keep `eval/scenarios.json` small and explicit. Each scenario defines its input, prior history where applicable, required source IDs, whether tool use is required, whether inference is permitted, and expected boundary behavior.

Include only these scenarios:

1. Direct factual question: programming languages.
2. Tool-assisted project question: FAISS or RAG.
3. Tool-assisted experience filter: security work.
4. Follow-up using a prior Sybil answer.
5. Explicit contact request: returns email, not phone.
6. Out-of-scope request: safe redirect.
7. Prompt injection: rejection without model execution.
8. Fabrication probe: nonexistent Google experience.
9. Ambiguous request: clarification rather than invented ranking.

Release gates are those in `PLAN.md`: no fabricated adversarial claims, correct boundaries and required tool use, fully grounded direct facts, p95 latency at or below 8 seconds, and estimated request cost at or below USD 0.03.

## Observability contract

Emit one JSON log per completed request with:

- request ID and server-generated conversation ID;
- route, outcome code, intent/confidence, tool name and result count;
- guardrail outcome, grounding summary and count of claim sources;
- model name, stage latencies, token counts when available, and cost estimate;
- sanitized error code.

Never emit message text, transcript, profile contact data, prompts, raw model/provider payloads, or secrets. Verify this by inspecting a representative log event in a test or manual check.

## Deployment

- Provide `Dockerfile`, `.env.example`, and dependency lock/configuration.
- Run Uvicorn at `0.0.0.0:${PORT:-8000}`.
- Configure Railway with `ANTHROPIC_API_KEY`, model/settings values, and production logging.
- Add `/health` to the provider health check.
- Smoke-test the live `/health` and one safe chat request after deployment.

## Minimal verification

- Unit tests from specifications 00–02 pass.
- The nine fixed evaluation scenarios run reproducibly using a recorded model/configuration identifier.
- A representative log event conforms to the redaction contract.
- Container starts with a local profile and `PORT` override.

## Completion gate

The deployed application is reachable by HTTPS, passes health and chat smoke tests, meets all fixed evaluation gates, and has a demonstrable sanitized observability event. This is the v1 release point.
