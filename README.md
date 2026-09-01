# Banorte CV Agent

A grounded, public-facing chat assistant for Marco Reyes' professional profile, built for the Reto IA Banorte challenge.

The assistant answers only from a validated local profile, uses specialized and universal typed read-only tools, uses fact/source IDs to select canonical facts for deterministic rendering, and applies input/output privacy guardrails.

## What it does

- Answers questions about experience, projects, skills, education, and languages.
- Uses deterministic profile search and filtering before model generation when a tool is needed.
- Supports normalized English/Spanish resume questions and deterministic verified answers when classification or generation is unavailable.
- Treats fact/source citations as selection signals—not semantic proof—and renders their canonical values deterministically.
- Rejects prompt-injection attempts and out-of-scope requests without sending them to the model.
- Does not disclose the stored phone number; email is only available for an explicit contact request.
- Exposes a small FastAPI contract at `POST /api/chat` and serves a responsive static chat UI at `/`.

## Architecture

The project is a modular Python/FastAPI application:

```text
browser → POST /api/chat → input guard → intent classifier → typed profile tool
        → grounded model response → output guard → sanitized public response
```

`data/profile.json` is the sole runtime source of biographical claims. The in-memory fact catalog and search index are derived from it at runtime. There is intentionally no vector database, embedding pipeline, durable chat memory, or write-capable tool in v1; compact follow-up state is optionally carried by the client.

More detail is documented in [DECISIONS.md](DECISIONS.md) and the sequential build specifications in [SPECIFICATIONS.md](SPECIFICATIONS.md).
For the step-by-step public deployment and challenge integration process, see [GO_LIVE_AND_CHALLENGE_TESTING.md](GO_LIVE_AND_CHALLENGE_TESTING.md).

## Run locally

Requirements: Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The UI and health endpoint run without a provider key. To receive real model responses, add an Anthropic API key to `.env`:

```dotenv
ANTHROPIC_API_KEY=your_key_here
```

Never commit `.env` or an API key.

## API

### `POST /api/chat`

```json
{
  "message": "What security-related work has Marco done?",
  "history": [],
  "preferences": { "verbosity": "concise" },
  "state": null
}
```

A successful response contains opaque request and conversation IDs, the grounded answer, and optional client-carried follow-up state. Provider failures still produce deterministic verified resume answers when possible; validation, rate-limit, unrecoverable provider, and unexpected failures return sanitized `application/problem+json` bodies.

### `GET /health`

Returns the application version and readiness status. Use it for deployment health checks.

## Verify

The verification set is deliberately small and risk-based:

```bash
python -m pytest -q
python -m compileall -q src tests eval
python -m eval.run_eval
```

To run the fixed scenarios against the configured model after setting `ANTHROPIC_API_KEY`:

```bash
python -m eval.run_eval --execute
```

The GitHub Actions workflow runs the offline checks on every push and pull request. It never requires or exposes a provider key.

## Deployment

The included `Dockerfile` runs Uvicorn on `0.0.0.0:${PORT:-8000}`. For a Render Web Service deployment, configure at least:

- `ANTHROPIC_API_KEY`
- `ENVIRONMENT=production`
- an HTTPS health check for `/health`

Run the live evaluation and a deployed chat smoke test before treating a deployment as a v1 release.

## Project structure

```text
src/          FastAPI app, agent workflow, guards, tools, and observability
data/         Validated runtime professional profile
frontend/     Dependency-free accessible chat interface
tests/        Focused risk-based test suite
eval/         Fixed evaluation scenarios and runner
specs/        Sequential implementation specifications
```
