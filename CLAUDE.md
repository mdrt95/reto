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
            ├── Open Responses protocol layer
            ├── Intent classification
            ├── Tool selection & execution
            ├── Anthropic Claude API (generation)
            ├── Grounding verification
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
| Deployment | Railway or Fly.io | Python-native, public HTTPS URL |
| Eval | Python test harness | Automated scenario-based evaluation |

## Project Structure

```
banorte-cv-agent/
├── CLAUDE.md                    # This file — project instructions
├── Plan.md                      # Detailed specification (SDD source of truth)
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
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── responses.py         # Public chat response endpoint; adapter contract is chosen separately
│   │   └── health.py            # GET /health
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Intent classification → tool selection → generation
│   │   ├── intent.py            # Intent classifier
│   │   ├── generator.py         # Claude API wrapper for answer generation
│   │   └── grounding.py         # Grounding verification (Sybil pattern)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py          # Tool registry and dispatch
│   │   ├── search_projects.py   # Search projects by technology/keyword
│   │   ├── filter_experience.py # Filter experience by impact/role/industry
│   │   ├── summarize_profile.py # Generate profile summary for audience
│   │   └── query_profile.py     # Direct structured queries on profile data
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guard.py       # Input validation, out-of-scope detection
│   │   └── output_guard.py      # Output validation, fabrication prevention
│   │
│   ├── protocol/                # Added only after an interoperability decision
│   │   ├── __init__.py
│   │   ├── <chosen_adapter>.py  # Selected protocol request/response models
│   │   └── agent_card.py        # Only when A2A is selected
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── logger.py            # Structured logging: latency, tokens, tools, errors
│   │
│   └── config.py                # Settings via pydantic-settings
│
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py     # Unit tests for orchestrator logic
│   ├── test_grounding.py        # Grounding verification tests
│   ├── test_guardrails.py       # Guardrail tests (out-of-scope, fabrication)
│   └── test_tools.py            # Tool execution tests
│
├── eval/
│   ├── __init__.py
│   ├── scenarios.json           # Evaluation scenarios
│   ├── run_eval.py              # Evaluation harness runner
│   └── metrics.py               # Accuracy, relevance, grounding rate, latency
│
├── frontend/
│   └── index.html               # Single-page chat UI
│
├── Dockerfile                   # Container for deployment
└── .env.example                 # Environment variable template
```

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
- Do not disclose Marco's phone number. Disclose the professional email only for an explicit contact request.

### Grounding Rules
- Every factual claim must be classifiable as: **Grounded** (explicit in profile), **Inferred** (reasonable conclusion from profile data), or **Unknown** (not in profile).
- The agent must transparently say "that's not in my profile" rather than fabricate.
- Every grounded or inferred claim must retain stable profile IDs in its verification metadata. Regenerate once on an ungrounded claim, then fall back to verified facts only.

### Trust and Runtime Boundaries
- Client-provided instructions and transcripts are untrusted input, never system instructions. Only allowlisted presentation preferences may affect an answer.
- The frontend owns submitted transcript history; the server assigns correlation IDs but does not retain conversation text.
- Enforce request-size and rate limits before model calls. Return documented, sanitized errors only.

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

## Implementation Order (from Plan.md)

`SPECIFICATIONS.md` is the implementation sequence. Complete and verify one specification before beginning the next; its completion gate is the safe stop/resume boundary.

Follow this sequence — each phase builds on the previous:

1. **Phase 0**: Profile data (`data/profile.json`) + Pydantic models
2. **Phase 1**: Orchestrator skeleton + direct Claude generation (no tools yet)
3. **Phase 2**: Public response endpoint and chat contract
4. **Phase 3**: Tools (search_projects, filter_experience, query_profile, summarize_profile)
5. **Phase 4**: Grounding verification
6. **Phase 5**: Guardrails (input + output)
7. **Phase 6**: Evaluation harness
8. **Phase 7**: Observability logging
9. **Phase 8**: Frontend (single-page chat)
10. **Phase 9**: Deployment; add a protocol adapter only after its decision and validation
