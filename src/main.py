"""FastAPI bootstrap for the Banorte CV Agent."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.claude import create_default_agent_service
from src.api.chat import InMemoryRateLimiter, ProblemDetails, PublicProblem, router as chat_router
from src.api.health import router as health_router
from src.config import Settings
from src.protocol.openai_compat import router as openai_compat_router
from src.protocol.response_store import ResponseStateStore
from src.models.profile import load_profile
from src.observability.logger import TurnLogEvent, configure_logging, log_turn


def create_app(
    settings: Settings | None = None,
    *,
    agent_service: Any | None = None,
    responses_state_store: ResponseStateStore | None = None,
) -> FastAPI:
    """Create the HTTP service and validate its approved profile during startup."""
    runtime_settings = settings or Settings()
    configure_logging(runtime_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        app.state.profile = load_profile(runtime_settings.profile_path)
        app.state.agent_service = agent_service or create_default_agent_service(
            app.state.profile,
            runtime_settings,
        )
        app.state.ready = True
        yield
        app.state.ready = False

    application = FastAPI(title="Banorte CV Agent", version="0.1.0", lifespan=lifespan)
    application.state.settings = runtime_settings
    application.state.rate_limiter = InMemoryRateLimiter()
    application.state.responses_state_store = (
        responses_state_store
        if responses_state_store is not None
        else ResponseStateStore(
            ttl_seconds=runtime_settings.responses_state_ttl_seconds,
            max_entries=runtime_settings.responses_state_max_entries,
        )
    )
    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(openai_compat_router)

    @application.middleware("http")
    async def enforce_request_size(request: Request, call_next: Any) -> Any:
        """Reject oversized chat payloads before parsing or model orchestration."""
        if request.url.path == "/api/chat":
            content_length = request.headers.get("content-length")
            maximum_bytes = runtime_settings.max_input_chars * 2 + 4_096
            if content_length and content_length.isdigit() and int(content_length) > maximum_bytes:
                request_id = f"req_{uuid4().hex}"
                return JSONResponse(
                    status_code=422,
                    content=ProblemDetails(
                        type="https://banorte-cv-agent.invalid/problems/invalid-request",
                        title="Invalid request",
                        status=422,
                        code="invalid_request",
                        request_id=request_id,
                    ).model_dump(),
                    media_type="application/problem+json",
                )
        return await call_next(request)

    @application.exception_handler(PublicProblem)
    async def public_problem_handler(_: Any, error: PublicProblem) -> JSONResponse:
        """Return stable, content-free problem details for expected API failures."""
        log_turn(
            TurnLogEvent(
                request_id=error.request_id,
                conversation_id=f"conv_{uuid4().hex}",
                outcome_code=error.code,
                guardrail_input="pass",
                guardrail_output="pass",
                latency_total_ms=0,
                error_code=error.code,
            )
        )
        return JSONResponse(
            status_code=error.status,
            content=ProblemDetails(
                type=f"https://banorte-cv-agent.invalid/problems/{error.code}",
                title=error.title,
                status=error.status,
                code=error.code,
                request_id=error.request_id,
            ).model_dump(),
            media_type="application/problem+json",
        )

    @application.exception_handler(RequestValidationError)
    async def validation_problem_handler(_: Any, __: RequestValidationError) -> JSONResponse:
        """Sanitize Pydantic request errors instead of exposing schema internals."""
        request_id = f"req_{uuid4().hex}"
        log_turn(
            TurnLogEvent(
                request_id=request_id,
                conversation_id=f"conv_{uuid4().hex}",
                outcome_code="invalid_request",
                guardrail_input="pass",
                guardrail_output="pass",
                latency_total_ms=0,
                error_code="invalid_request",
            )
        )
        return JSONResponse(
            status_code=422,
            content=ProblemDetails(
                type="https://banorte-cv-agent.invalid/problems/invalid-request",
                title="Invalid request",
                status=422,
                code="invalid_request",
                request_id=request_id,
            ).model_dump(),
            media_type="application/problem+json",
        )

    @application.exception_handler(Exception)
    async def unexpected_problem_handler(_: Any, __: Exception) -> JSONResponse:
        """Prevent unexpected implementation or provider details from leaking publicly."""
        request_id = f"req_{uuid4().hex}"
        log_turn(
            TurnLogEvent(
                request_id=request_id,
                conversation_id=f"conv_{uuid4().hex}",
                outcome_code="internal_error",
                guardrail_input="pass",
                guardrail_output="pass",
                latency_total_ms=0,
                error_code="internal_error",
            )
        )
        return JSONResponse(
            status_code=500,
            content=ProblemDetails(
                type="https://banorte-cv-agent.invalid/problems/internal-error",
                title="Internal server error",
                status=500,
                code="internal_error",
                request_id=request_id,
            ).model_dump(),
            media_type="application/problem+json",
        )

    frontend_directory = Path(__file__).resolve().parents[1] / "frontend"
    if frontend_directory.is_dir():
        application.mount("/", StaticFiles(directory=frontend_directory, html=True), name="frontend")
    return application


app = create_app()
