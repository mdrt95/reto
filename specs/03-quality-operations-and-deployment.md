# 03 — Quality, Operations, and Deployment

## Goal

Prove the implemented behavior, make it observable without retaining content, and deploy a reproducible public demo.

## Answer-contract scenario matrix

`eval/scenarios.json` is the single source of truth for both automated evaluation and
the manual UI checklist. Each scenario declares its input and optional history, any
deliberate double, and the complete contract its answer must satisfy: expected outcome,
HTTP status, answer mode, rendering mode, topic, tool, response language, exact required
and forbidden fact and source IDs, required and forbidden tokens, sentence and word
budgets, permitted generation calls, and whether inference is permitted.

Fact and source sets are **exact**. A missing expectation and an unexpected extra are
both failures; a superset is no longer tolerated.

The matrix fixes paired English/Spanish scenarios for: exact employment start date,
broad experience, broad projects, specific security work, named technology, experience
summary, project summary, impact, provider failure during direct retrieval, and provider
failure during synthesis. It also fixes deliberately incompatible classifier doubles —
`companies` against project, experience, and start-date questions; `summary_request`
against an exact security question; `skills` against a named-technology question — each
of which must still produce the expected mode, topic, fact scope, rendering mode, and
status from the current message alone. The guardrail scenarios (contact probe,
out-of-scope, injection, fabrication probe, ambiguous ranking) are retained unchanged.

Provider failure is reproduced deterministically by substituting a failing stage, not by
waiting for a real outage, so these scenarios are reproducible offline and in CI.

`inference_permitted` is enforced, not merely parsed. A scenario that forbids inference
must resolve to a deterministic rendering mode — `canonical`, `canonical_fallback`,
`canonical_not_found`, or `clarification` — or to a guardrail boundary that never reaches
the answer plan at all. `transformed` is the only mode whose wording the provider chose.

## Hard release gates

- Every direct scenario selects only its required facts, with no parent, sibling, or unrelated extras.
- Every direct scenario makes zero generation or rephrase calls after deterministic selection.
- Every date question returns the canonical date without invented day precision.
- Experience and project questions return the correct topic and never an employer-only projection unless employers were explicitly requested.
- Every synthesis output stays within the 3-sentence and 75-word default, and none concatenates all selected canonical narratives.
- No output introduces an unsupported entity, technology, number, responsibility, impact, or outcome.
- No clear résumé question with sufficient canonical facts returns HTTP 503.
- Equivalent English and Spanish scenarios resolve to the same dimension, topic, scope, and ranking, with Spanish a prefix of the English selection (D-036).
- Provider-outage scenarios return the expected direct or concise canonical fallback at HTTP 200.
- Tool-required scenarios verify the correct tool and topic scope, not merely that some tool ran.
- Internal traces carry the expected answer mode, rendering mode, fact and source selection, fallback reason, and final size counts.

Operational gates from `PLAN.md` continue to apply: p95 latency at or below 8 seconds and
estimated request cost at or below USD 0.03.

## Two verification layers

The scenario matrix is a regression suite, and passing it proves the implementation
handles those phrasings. It cannot prove the hard gate's promise that reasonable résumé
questions are robustly handled, because a fixed matrix never varies the dimension real
users vary constantly: surface form. Verification is therefore split by cost.

**Offline layer.** Paraphrase families, accent and punctuation perturbation, referent
chains, and system invariants run against `AnswerPlanner`, `detect_response_language`,
and the deterministic profile tools, which are provider-free by construction (D-033),
with the classifier stubbed. Hundreds of variants in milliseconds, on every push.

**Live layer.** The fixed scenario matrix, kept small, covering what genuinely needs a
provider: prose, containment, and transformation.

Every paraphrase family declares both the variants expected to resolve and the variants
deliberately out of scope, each with a stated reason. A variant count is uninterpretable
without its denominator, and an out-of-scope phrasing may be unresolved but never unsafe.

## Lean verification policy

Maintain exactly these groups, and no more:

1. the table-driven direct-answer contract group (`tests/test_direct_answer_plans.py`);
2. the table-driven synthesis contract group (`tests/test_synthesis_contract.py`);
3. the shared provider-failure and fallback group, carried within the two groups above;
4. the offline routing-invariant group (`tests/test_routing_invariants.py`), which
   varies surface form across declared paraphrase families and asserts the system
   invariants below, with negative controls proving its own detectors can fire;
5. the evaluator-contract group (`tests/test_eval_contract.py`), which proves the checker
   can fail — a matrix that only ever passes is not evidence;
6. the fixed bilingual UI release script below.

### System invariants

Asserted across every phrasing the offline layer knows, not per scenario:

- the generator is never invoked with an empty allowed-fact set;
- no question with sufficient canonical facts returns HTTP 503 or a clarification;
- removing accents changes neither the detected language nor the resolved routing;
- equivalent English and Spanish families resolve to the same dimension and topic.

Do not add tests for exact generated prose, wording variants, spelling variants, or
implementation details. Add a regression only for a new boundary, a new failure mode, or
behavior actually observed to fail.

## Final UI release gate

Automated checks cover what a reader cannot see. They do not judge whether an answer
sounds like a person. Both gates are required.

1. Start the application with the configured provider.
2. Run this fixed script through the real frontend, in order:
   1. `¿Desde cuándo trabaja Marco en Global Payments?`
   2. `Has Marco worked with FAISS?`
   3. `¿En qué proyectos ha trabajado Marco?`
   4. `What security-related work has Marco done?`
   5. `Dime acerca de la experiencia de Marco`
   6. `Summarize Marco's experience.`
   7. `Resume la experiencia de Marco.`
   8. `Summarize the projects Marco has worked on.`
   9. `What impact did Marco's work have?`
   10. `¿Qué impacto tuvo el trabajo de Marco?`
3. Record each answer, its HTTP outcome, and its sanitized trace.
4. Fail the release if any answer is off-topic, broader than the question, a verbose
   résumé dump, unnatural in the requested language, or inconsistent with the acceptance
   criteria — even when every automated check passed.

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
- Run Uvicorn at `0.0.0.0:${PORT:-8000}` with `--proxy-headers` and `FORWARDED_ALLOW_IPS` set so per-IP rate limiting reads the real client address behind Render's single proxy.
- Configure Render with `ANTHROPIC_API_KEY`, model/settings values, and production logging.
- Add `/health` to the provider health check.
- Smoke-test the live `/health` and one safe chat request after deployment.

## Minimal verification

- Unit tests from specifications 00–02 pass.
- The fixed scenario matrix runs reproducibly using a recorded model/configuration identifier, and the evaluator-contract group passes offline.
- A representative log event conforms to the redaction contract.
- Container starts with a local profile and `PORT` override.

## Completion gate

The deployed application is reachable by HTTPS, passes health and chat smoke tests, meets every hard release gate above, passes the fixed bilingual UI script, and has a demonstrable sanitized observability event. This is the v1 release point.
