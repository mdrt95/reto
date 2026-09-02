"""First-party public HTTP contract for the bounded CV-agent service."""

import json
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Literal, Protocol
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agent.claude import GenerationUnavailableError
from src.agent.contracts import AgentResponse, ConversationState
from src.observability.logger import TurnLogEvent, log_turn


router = APIRouter(tags=["chat"])


class HistoryItem(BaseModel):
    """A bounded client-owned prior turn used only for current-request context."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, value: str) -> str:
        """Reject blank transcript entries before they can become model context."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        return stripped


class Preferences(BaseModel):
    """Allowlisted presentation preferences that cannot modify system behavior."""

    language: str | None = None
    verbosity: Literal["concise", "standard", "detailed"] | None = None
    model_config = ConfigDict(extra="forbid")


class ChatRequest(BaseModel):
    """Validated first-party request before settings-dependent limit checks."""

    message: str = Field(min_length=1)
    history: list[HistoryItem] = Field(default_factory=list)
    preferences: Preferences | None = None
    state: ConversationState | None = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("message")
    @classmethod
    def require_nonblank_message(cls, value: str) -> str:
        """Normalize the user message while rejecting whitespace-only requests."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    """Public success payload without provider or grounding implementation detail."""

    id: str
    answer: str
    conversation_id: str
    status: Literal["completed"] = "completed"
    state: ConversationState | None = None


class ProblemDetails(BaseModel):
    """Sanitized RFC-7807-inspired public failure payload."""

    type: str
    title: str
    status: int
    code: str
    request_id: str


class PublicProblem(Exception):
    """An expected client-safe problem response raised within a route."""

    def __init__(self, *, status: int, code: str, title: str, request_id: str) -> None:
        self.status = status
        self.code = code
        self.title = title
        self.request_id = request_id


class ChatResponder(Protocol):
    """The narrow core-service port required by the public delivery adapter."""

    def respond(
        self,
        message: str,
        *,
        history: list[object],
        state: ConversationState | None = None,
    ) -> AgentResponse: ...


@dataclass
class InMemoryRateLimiter:
    """Small in-process limiter appropriate only for the stateless v1 demo."""

    requests: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def allow(self, identity: str, *, limit: int) -> bool:
        """Permit no more than the configured number of calls in one rolling minute."""
        now = monotonic()
        entries = self.requests[identity]
        while entries and now - entries[0] >= 60:
            entries.popleft()
        if len(entries) >= limit:
            return False
        entries.append(now)
        return True


def new_request_id() -> str:
    """Generate opaque correlation IDs without encoding client data."""
    return f"req_{uuid4().hex}"


@router.post("/api/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Validate, rate-limit, and delegate one public chat turn to the core service."""
    request_id = new_request_id()
    settings = request.app.state.settings
    if len(payload.message) > settings.max_input_chars:
        raise PublicProblem(status=422, code="invalid_request", title="Invalid request", request_id=request_id)
    history_characters = sum(len(item.content) for item in payload.history)
    if len(payload.history) > settings.max_history_messages or history_characters > settings.max_input_chars:
        raise PublicProblem(status=422, code="invalid_request", title="Invalid request", request_id=request_id)

    client_host = request.client.host if request.client else "unknown"
    limiter: InMemoryRateLimiter = request.app.state.rate_limiter
    if not limiter.allow(client_host, limit=settings.rate_limit_per_minute):
        raise PublicProblem(status=429, code="rate_limited", title="Too many requests", request_id=request_id)

    history = [item.model_dump() for item in payload.history]
    started_at = monotonic()
    responder: ChatResponder = request.app.state.agent_service
    try:
        if payload.state is None:
            result = responder.respond(payload.message, history=history)
        else:
            result = responder.respond(payload.message, history=history, state=payload.state)
    except GenerationUnavailableError as error:
        logging.getLogger("banorte_cv_agent.generation").warning(
            json.dumps({"event": "generation_unavailable", "request_id": request_id, "error_type": type(error).__name__})
        )
        raise PublicProblem(
            status=503,
            code="generation_unavailable",
            title="Answer generation is temporarily unavailable",
            request_id=request_id,
        ) from error

    conversation_id = f"conv_{uuid4().hex}"
    log_turn(
        TurnLogEvent(
            request_id=request_id,
            conversation_id=conversation_id,
            outcome_code="completed",
            intent=result.trace.intent,
            intent_confidence=result.trace.intent_confidence,
            tool_name=result.trace.tool_name,
            tool_result_count=result.trace.tool_result_count,
            guardrail_input=result.trace.guardrail_input,
            guardrail_output=result.trace.guardrail_output,
            grounding_status=result.trace.grounding_status,
            rephrase_outcome=result.trace.rephrase_outcome,
            fallback_reason=result.trace.fallback_reason,
            generator_skipped=result.trace.generator_skipped,
            selection_path=result.trace.selection_path,
            referent_source=result.trace.referent_source,
            referent_correction=result.trace.referent_correction,
            informativeness_outcome=result.trace.informativeness_outcome,
            answer_mode=result.trace.answer_mode,
            rendering_mode=result.trace.rendering_mode,
            evidence_topics=result.trace.evidence_topics,
            synthesis_dimension=result.trace.synthesis_dimension,
            transformation_outcome=result.trace.transformation_outcome,
            final_word_count=result.trace.final_word_count,
            final_sentence_count=result.trace.final_sentence_count,
            selected_fact_count=len(result.trace.selected_fact_ids),
            selected_source_count=len(set(result.trace.selected_source_ids)),
            claim_source_count=len(set(result.trace.claim_source_ids)),
            latency_total_ms=round((monotonic() - started_at) * 1_000),
            model_name=settings.model_name,
        )
    )
    return ChatResponse(
        id=request_id,
        answer=result.answer,
        conversation_id=conversation_id,
        state=result.state,
    )
