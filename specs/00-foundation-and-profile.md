# 00 — Foundation and Profile

## Goal

Create a runnable, type-safe foundation that loads one approved professional profile and exposes only a health check. No model call, agent behavior, or frontend belongs in this phase.

## Scope

- Create the Python 3.12 project configuration and pinned runtime dependencies: FastAPI, Uvicorn, Pydantic v2, Pydantic Settings, Anthropic SDK, and pytest.
- Create `data/profile.json` by reconciling `MDRT Resume.json`. Preserve all stable IDs; add an empty-list default in models for optional `tags` and `technologies` fields.
- Create Pydantic models for metadata, personal data, language, skills, experience/highlights, projects/highlights, education, and the complete profile.
- Load and validate the profile once during application startup. Fail startup with a clear operator-facing error if it is missing or invalid.
- Add typed settings: `ANTHROPIC_API_KEY`, `MODEL_NAME`, `MODEL_TIMEOUT_SECONDS`, `MAX_INPUT_CHARS`, `MAX_HISTORY_MESSAGES`, `RATE_LIMIT_PER_MINUTE`, `ENVIRONMENT`, and `LOG_LEVEL`.
- Add `GET /health`, returning version and readiness without secrets or profile content.

## Constraints

- `data/profile.json` is the runtime source of truth. Do not load the authoring resume in application code.
- Phone number may be stored because it is source data, but no public response model may expose it.
- All function signatures are typed. Public functions have concise what/why docstrings.
- Settings must fail fast for missing required production values; test and local development may use explicit test settings.

## Minimal verification

- A valid profile fixture loads and exposes stable IDs.
- Invalid required data fails validation at startup.
- `/health` returns `200` and no secret/config value.

## Completion gate

`pytest` passes these focused tests, the service starts locally with a validated profile, and `/health` returns ready. Stop safely here before any agent or API contract work.
