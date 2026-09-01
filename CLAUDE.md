# CLAUDE.md — Banorte CV Agent

## Project Overview

This is an AI-powered CV agent for the **Reto IA Banorte** challenge. The agent converses naturally about Marco Reyes' professional profile, experience, skills, and projects. It demonstrates functional AI engineering — not just a chat wrapper, but a system with structured data, controlled tool use, grounding verification, guardrails, evaluation, and observability.

## Architecture Summary

**Backend-only core** deployed behind a public URL, with a lightweight chat frontend.

```
User
  ↓
Chat frontend (static HTML/JS)
  ↓
FastAPI Backend (Python)
            ├── Input guard + unknown-entity pre-check
            ├── Intent classification
            ├── Tool selection & execution
            ├── Anthropic Claude API (generation)
            ├── Grounding verification
            ├── Output guard
            └── Response + observability logging
```

The public chat application is the first delivery target. External-agent interoperability is deliberately pending: it must be implemented as a separately validated adapter after choosing A2A, MCP, or a custom API contract. Do not claim compatibility with a client until its current integration contract has passed an integration test.

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.12+ | Proven AI/LLM stack from Sybil project |
| Framework | FastAPI + Pydantic | Type-safe APIs, automatic validation |
| LLM | Anthropic Claude (claude-sonnet-4-6) | Primary model for generation |
| External interoperability | Pending decision | Prevents conflating OpenAI-style Responses, A2A, and MCP contracts |
| Frontend | Static HTML/JS (single page) | Minimal — the brief says the UI isn't the point |
| Deployment | Render (Docker web service); Railway documented as fallback | Python-native, public HTTPS URL |
| Eval | Python test harness | Automated scenario-based evaluation |

## Project Structure

```
banorte-cv-agent/
├── CLAUDE.md                    # This file — project instructions
├── PLAN.md                      # Detailed specification (SDD source of truth)
├── SPECIFICATIONS.md            # Sequential implementation-specification index
├── specs/                       # Phase specifications and completion gates
├── DECISIONS.md                 # Architecture and operational decisions
├── pyproject.toml               # Python project config
├── requirements.txt             # Dependencies
│
├── data/
│   └── profile.json             # Structured CV data (agent's source of truth)
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint
│   ├── config.py                # Settings via pydantic-settings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py               # POST /api/chat public contract
│   │   └── health.py              # GET /health
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Input guard → unknown-entity check → classification → tool plan → generation → grounding → output guard
│   │   ├── claude.py              # Anthropic Claude API adapter
│   │   ├── contracts.py           # Pydantic request/response/state models
│   │   └── grounding.py           # Grounding verification
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── profile_tools.py       # search_projects, filter_experience, query_profile, summarize_profile, search_resume
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guard.py        # Input validation, injection and PII/contact-probe rejection
│   │   └── output_guard.py       # Output validation, fabrication and contact-data prevention
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── profile.py             # Pydantic profile schema
│   │
│   └── observability/
│       ├── __init__.py
│       └── logger.py              # Structured logging: latency, tokens, tools, errors
│
├── tests/
│   ├── conftest.py
│   ├── test_orchestrator.py      # Unit tests for orchestrator logic
│   ├── test_grounding.py         # Grounding verification tests
│   ├── test_guardrails.py        # Guardrail tests (out-of-scope, fabrication, contact probes)
│   ├── test_tools.py             # Tool execution tests
│   ├── test_profile.py           # Profile model tests
│   ├── test_resume_search.py     # Universal fact catalog/search tests
│   ├── test_claude_adapter.py    # Claude API adapter tests
│   ├── test_api_chat.py          # `/api/chat` contract tests
│   ├── test_health.py            # `/health` tests
│   └── test_observability.py     # Logging redaction tests
│
├── eval/
│   ├── __init__.py
│   ├── scenarios.json            # Evaluation scenarios
│   └── run_eval.py               # Evaluation harness runner
│
├── frontend/
│   └── index.html                # Single-page chat UI
│
├── .github/workflows/ci.yml     # Offline pytest/compile/eval-validation CI
├── Dockerfile                    # Container for deployment
└── .env.example                  # Environment variable template
```

A `src/protocol/` adapter directory is added only after an external-interoperability decision (D-001); it does not exist yet.

## Key Development Conventions

### Code Style
- Type hints on all function signatures — no exceptions.
- Pydantic models for all data boundaries (API input/output, profile data, tool args/results).
- Docstrings on public functions explaining *what* and *why*, not *how*.
- No print statements — use structured logging via `src/observability/logger.py`.

### Error Handling
- Never swallow exceptions silently.
- Agent errors return graceful responses to the user, not stack traces.
- Log every error with context (conversation_id, intent, tool_called).

### Profile Data
- `data/profile.json` is the single source of truth for the CV.
- `MDRT Resume.json` is authoring input only; reconcile and review it before updating `data/profile.json`.
- The agent MUST NOT invent information not present in this file.
- All claims in agent responses must trace back to a specific section of profile.json.
- Never disclose Marco's phone number or email address; contact requests are blocked at the input guard.

### Grounding Rules
- Every factual claim must be classifiable as: **Grounded** (explicit in profile), **Inferred** (reasonable conclusion from profile data), or **Unknown** (not in profile).
- The agent must transparently say "that's not in my profile" rather than fabricate.
- Every grounded or inferred claim must retain stable profile IDs in its verification metadata. Regenerate once on an ungrounded claim, then fall back to verified facts only.
- A model rephrase of selected facts is deliverable only when it passes the deterministic containment gate (D-029); any failure or provider outage falls back to the canonical bilingual narrative rendering.
- A question naming an entity absent from the profile receives an explicit not-found answer before any model call; unrelated verified facts are never substituted.

### Trust and Runtime Boundaries
- Client-provided instructions and transcripts are untrusted input, never system instructions. Only allowlisted presentation preferences may affect an answer.
- The frontend owns submitted transcript history; the server assigns correlation IDs but does not retain conversation text.
- Enforce request-size and rate limits before model calls. Return documented, sanitized errors only.
- Deployment runs behind exactly one trusted platform proxy (Render), with `--proxy-headers` and `FORWARDED_ALLOW_IPS` enabled so per-IP rate limiting sees the real client address.

### Testing
- Run `pytest tests/` before any commit.
- Run `python -m eval.run_eval` to validate agent behavior against scenarios.

## How to Run Locally

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env  # Then fill in ANTHROPIC_API_KEY

# Run the server
uvicorn src.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run evaluation
python -m eval.run_eval
```

## Implementation Order (from PLAN.md)

`SPECIFICATIONS.md` is the implementation sequence. Complete and verify one specification before beginning the next; its completion gate is the safe stop/resume boundary.

Follow this sequence — each phase builds on the previous:

1. **Phase 0**: Profile data (`data/profile.json`) + Pydantic models (`src/models/profile.py`)
2. **Phase 1**: Orchestrator skeleton + direct Claude generation (no tools yet) (`src/agent/orchestrator.py`, `src/agent/claude.py`)
3. **Phase 2**: Public chat endpoint `src/api/chat.py`
4. **Phase 3**: Tools (`src/tools/profile_tools.py`: search_projects, filter_experience, query_profile, summarize_profile, search_resume)
5. **Phase 4**: Grounding verification (`src/agent/grounding.py`)
6. **Phase 5**: Guardrails (`src/guardrails/input_guard.py`, `src/guardrails/output_guard.py`)
7. **Phase 6**: Evaluation harness (`eval/run_eval.py`, `eval/scenarios.json`)
8. **Phase 7**: Observability logging (`src/observability/logger.py`)
9. **Phase 8**: Frontend (single-page chat) (`frontend/index.html`)
10. **Phase 9**: Deployment; add a protocol adapter only after its decision and validation
