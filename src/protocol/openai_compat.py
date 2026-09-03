"""OpenAI Responses API (`POST /v1/responses`) compatibility adapter.

This is the decision D-001 external-interoperability adapter for OpenAI-style
clients. It translates the OpenAI Responses wire contract to and from the same
``AgentService`` the first-party ``/api/chat`` route uses. Nothing about agent,
tool, guardrail, or grounding behavior changes here: the adapter only re-shapes
the HTTP boundary and enforces the same request limits and per-client rate limit.

Trust boundary: client-supplied ``instructions``, sampling parameters, and tool
declarations are accepted and ignored. The message text and prior user/assistant
turns in ``input`` reach the core service as untrusted data.

Continuity (issue #27): ``previous_response_id`` is resolved through a bounded,
process-local store that maps the opaque ``resp_*`` ID to the compact verified
``ConversationState`` the core agent produced on that turn — catalog IDs and enum
values only, never message or answer text. Store keys are namespaced by a
non-reversible tag of the presenting bearer credential, so an ID never resolves
under a different token. An ID that is unknown, expired, malformed, or
cross-token fails closed with a machine-readable ``previous_response_not_found``
error and no provider call; the client may then resend history in ``input``.

Streaming (``stream: true``) returns a ``text/event-stream`` of Responses API
events, but the answer is produced and fully grounding-verified before the first
byte is emitted — the SSE sequence only frames an already-complete answer, so it
carries no time-to-first-token benefit and cannot bypass the verification gate.
"""

import hashlib
import hmac
import json
import logging
from collections.abc import Iterator
from time import monotonic, time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from src.agent.claude import GenerationUnavailableError
from src.agent.contracts import AgentResponse, ConversationState
from src.observability.logger import TurnLogEvent, log_turn
from src.protocol.response_store import ResponseStateStore

router = APIRouter(tags=["openai-compat"])

MODEL_ID = "banorte-cv-agent"
_MODEL_CREATED = 1_700_000_000
_LOGGER = logging.getLogger("banorte_cv_agent.openai_compat")


class _ContentPart(BaseModel):
    """One structured content part of an OpenAI message; non-text parts are dropped."""

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    text: str | None = None


class _InputMessage(BaseModel):
    """One item of a Responses ``input`` array. Non-message items parse but are filtered."""

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    role: str | None = None
    content: str | list[_ContentPart] | None = None


class ResponsesRequest(BaseModel):
    """The subset of the OpenAI Responses request this adapter reads.

    ``extra="ignore"`` is deliberate and the opposite of the strict first-party
    contract: OpenAI clients send many sampling and tooling fields that must not
    cause a rejection.
    """

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    input: str | list[_InputMessage] | None = None
    stream: bool | None = None
    previous_response_id: str | None = None


class _OpenAIError(Exception):
    """An error rendered in the OpenAI top-level ``{"error": {...}}`` envelope."""

    status = 500
    err_type = "api_error"
    code: str | None = None
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        super().__init__(self.message)


class _BadRequest(_OpenAIError):
    status = 400
    err_type = "invalid_request_error"
    message = "The request was invalid."


class _Unauthorized(_OpenAIError):
    status = 401
    err_type = "invalid_request_error"
    code = "invalid_api_key"
    message = "Incorrect API key provided."


class _UnknownPreviousResponse(_OpenAIError):
    status = 404
    err_type = "invalid_request_error"
    code = "previous_response_not_found"
    message = "Previous response not found or expired. Resend prior turns in 'input'."


class _NotConfigured(_OpenAIError):
    status = 503
    err_type = "api_error"
    message = "The OpenAI-compatible endpoint is not configured on this deployment."


class _RateLimited(_OpenAIError):
    status = 429
    err_type = "rate_limit_error"
    message = "Rate limit exceeded. Please retry later."


class _Upstream(_OpenAIError):
    status = 503
    err_type = "api_error"
    message = "Answer generation is temporarily unavailable."


def _error_response(error: _OpenAIError) -> JSONResponse:
    """Render any adapter error as the OpenAI error envelope with a plain JSON type."""
    return JSONResponse(
        status_code=error.status,
        content={
            "error": {
                "message": error.message,
                "type": error.err_type,
                "param": None,
                "code": error.code,
            }
        },
    )


def _require_configured_auth(request: Request) -> None:
    """Enforce the shared-secret bearer token; a missing token disables the endpoint."""
    configured = getattr(request.app.state.settings, "openai_compat_token", None)
    if configured is None:
        raise _NotConfigured()
    header = request.headers.get("authorization", "")
    presented = header[7:] if header.lower().startswith("bearer ") else ""
    if not presented or not hmac.compare_digest(presented, configured.get_secret_value()):
        raise _Unauthorized()


def _flatten_content(content: str | list[_ContentPart] | None) -> str:
    """Reduce string or structured content to a single trimmed text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    fragments = [
        part.text.strip()
        for part in content
        if part.text and (part.type is None or part.type.endswith("text"))
    ]
    return " ".join(fragment for fragment in fragments if fragment).strip()


def _to_core_inputs(payload: ResponsesRequest) -> tuple[str, list[dict[str, str]]]:
    """Map a Responses ``input`` to the core ``(message, history)`` pair.

    System/developer items, tool and reasoning items, and empty turns are dropped.
    The request must resolve to a final user message; anything else is a 400.
    """
    if payload.input is None:
        raise _BadRequest("The 'input' field is required.")

    if isinstance(payload.input, str):
        message = payload.input.strip()
        if not message:
            raise _BadRequest("The 'input' field must not be empty.")
        return message, []

    turns: list[dict[str, str]] = []
    for item in payload.input:
        if item.type not in (None, "message"):
            continue
        if item.role not in ("user", "assistant"):
            continue
        text = _flatten_content(item.content)
        if not text:
            continue
        turns.append({"role": item.role, "content": text})

    if not turns or turns[-1]["role"] != "user":
        raise _BadRequest("The 'input' must contain a final user message.")
    return turns[-1]["content"], turns[:-1]


def _enforce_limits(message: str, history: list[dict[str, str]], request: Request) -> None:
    """Apply the same size ceilings the first-party contract enforces."""
    settings = request.app.state.settings
    if len(message) > settings.max_input_chars:
        raise _BadRequest("The input message exceeds the maximum allowed length.")
    history_characters = sum(len(turn["content"]) for turn in history)
    if len(history) > settings.max_history_messages or history_characters > settings.max_input_chars:
        raise _BadRequest("The input history exceeds the allowed size.")


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _owner_tag(request: Request) -> str:
    """A short non-reversible tag for the adapter credential the caller presented.

    Snapshots are namespaced by this tag so a response ID minted under one bearer
    token never resolves under another — after the operator rotates
    ``OPENAI_COMPAT_TOKEN``, or if per-client tokens are introduced later. Auth
    has already run, so a configured secret is guaranteed to be present.
    """
    configured = request.app.state.settings.openai_compat_token
    secret = configured.get_secret_value().encode("utf-8")
    return hashlib.sha256(secret).hexdigest()[:16]


def _snapshot_key(request: Request, response_id: str) -> str:
    """Namespace a bare ``resp_*`` ID by the presenting credential's owner tag."""
    return f"{_owner_tag(request)}:{response_id}"


def _resolve_prior_state(
    payload: ResponsesRequest, request: Request
) -> ConversationState | None:
    """Turn ``previous_response_id`` into stored verified state, or fail closed.

    A blank or absent ID is a fresh turn. Any non-empty ID that the bounded store
    cannot resolve — unknown, expired, malformed, or minted under a different
    bearer token — is a ``previous_response_not_found`` error raised before the
    core service is called. The client-supplied ID is never logged or echoed.
    """
    previous_id = (payload.previous_response_id or "").strip()
    if not previous_id:
        return None
    store: ResponseStateStore = request.app.state.responses_state_store
    prior_state = store.get(_snapshot_key(request, previous_id))
    if prior_state is None:
        raise _UnknownPreviousResponse()
    return prior_state


def _log_completed(response_id: str, result: AgentResponse, latency_ms: int, model_name: str) -> None:
    trace = result.trace
    log_turn(
        TurnLogEvent(
            request_id=response_id,
            conversation_id=f"conv_{uuid4().hex}",
            route="/v1/responses",
            outcome_code="completed",
            intent=trace.intent,
            tool_name=trace.tool_name,
            guardrail_input=trace.guardrail_input,
            guardrail_output=trace.guardrail_output,
            grounding_status=trace.grounding_status,
            answer_mode=trace.answer_mode,
            rendering_mode=trace.rendering_mode,
            latency_total_ms=latency_ms,
            model_name=model_name,
        )
    )


def _log_error(response_id: str, code: str) -> None:
    log_turn(
        TurnLogEvent(
            request_id=response_id,
            conversation_id=f"conv_{uuid4().hex}",
            route="/v1/responses",
            outcome_code=code,
            guardrail_input="pass",
            guardrail_output="pass",
            latency_total_ms=0,
            error_code=code,
        )
    )


@router.post("/v1/responses", response_model=None)
def create_response(payload: ResponsesRequest, request: Request) -> JSONResponse | StreamingResponse:
    """Answer one OpenAI Responses request by delegating to the core agent service.

    Every fallible step — auth, size limits, rate limit, ``previous_response_id``
    resolution, and the core call — runs to completion before a streaming
    response is chosen, so a rejected request is always a plain JSON error and
    never a truncated event stream. A resolved snapshot reaches the core service
    as ``state``, on the same untrusted footing as ``message`` and ``history``;
    the JSON and SSE paths then frame one identical verified answer.
    """
    response_id = f"resp_{uuid4().hex}"
    started_at = monotonic()
    try:
        _require_configured_auth(request)

        message, history = _to_core_inputs(payload)
        _enforce_limits(message, history, request)

        limiter = request.app.state.rate_limiter
        if not limiter.allow(
            _client_host(request), limit=request.app.state.settings.rate_limit_per_minute
        ):
            raise _RateLimited()

        prior_state = _resolve_prior_state(payload, request)
        agent_service = request.app.state.agent_service
        if prior_state is None:
            result: AgentResponse = agent_service.respond(message, history=history)
        else:
            result = agent_service.respond(message, history=history, state=prior_state)
    except _OpenAIError as error:
        _log_error(response_id, error.code or error.err_type)
        return _error_response(error)
    except GenerationUnavailableError as error:
        _LOGGER.warning(
            json.dumps(
                {
                    "event": "generation_unavailable",
                    "request_id": response_id,
                    "error_type": type(error).__name__,
                }
            )
        )
        _log_error(response_id, "generation_unavailable")
        return _error_response(_Upstream())
    except Exception:  # noqa: BLE001 - boundary: never leak an internal failure shape
        _LOGGER.error(json.dumps({"event": "adapter_error", "request_id": response_id}))
        _log_error(response_id, "internal_error")
        return _error_response(_OpenAIError())

    model_name = payload.model or MODEL_ID
    latency_ms = round((monotonic() - started_at) * 1_000)
    _log_completed(response_id, result, latency_ms, model_name)

    if result.state is not None:
        request.app.state.responses_state_store.put(
            _snapshot_key(request, response_id), result.state
        )

    created_at = int(time())
    message_id = f"msg_{uuid4().hex}"
    if payload.stream:
        return StreamingResponse(
            _stream_events(response_id, message_id, result.answer, model_name, created_at),
            media_type="text/event-stream",
        )
    return JSONResponse(
        _response_object(response_id, message_id, result.answer, model_name, created_at, completed=True)
    )


def _message_item(message_id: str, answer: str, *, completed: bool) -> dict[str, Any]:
    """The single assistant message item, with or without its text populated."""
    return {
        "type": "message",
        "id": message_id,
        "status": "completed" if completed else "in_progress",
        "role": "assistant",
        "content": (
            [{"type": "output_text", "text": answer, "annotations": []}] if completed else []
        ),
    }


def _response_object(
    response_id: str,
    message_id: str,
    answer: str,
    model_name: str,
    created_at: int,
    *,
    completed: bool,
) -> dict[str, Any]:
    """Build the top-level ``response`` object for the JSON reply or an SSE frame."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed" if completed else "in_progress",
        "model": model_name,
        "output": [_message_item(message_id, answer, completed=True)] if completed else [],
        "output_text": answer if completed else "",
        "usage": (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0} if completed else None
        ),
        "error": None,
        "metadata": {},
    }


def _text_chunks(text: str, size: int = 64) -> list[str]:
    """Split the finished answer into ordered fragments that rejoin to the original."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def _stream_events(
    response_id: str, message_id: str, answer: str, model_name: str, created_at: int
) -> Iterator[str]:
    """Frame an already-complete, already-verified answer as Responses API SSE events."""
    sequence = 0

    def frame(event_type: str, payload: dict[str, Any]) -> str:
        nonlocal sequence
        body = {"type": event_type, "sequence_number": sequence, **payload}
        sequence += 1
        return f"event: {event_type}\ndata: {json.dumps(body)}\n\n"

    in_progress = _response_object(
        response_id, message_id, answer, model_name, created_at, completed=False
    )
    completed = _response_object(
        response_id, message_id, answer, model_name, created_at, completed=True
    )

    yield frame("response.created", {"response": in_progress})
    yield frame("response.in_progress", {"response": in_progress})
    yield frame(
        "response.output_item.added",
        {"output_index": 0, "item": _message_item(message_id, answer, completed=False)},
    )
    yield frame(
        "response.content_part.added",
        {
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )
    for chunk in _text_chunks(answer):
        yield frame(
            "response.output_text.delta",
            {"item_id": message_id, "output_index": 0, "content_index": 0, "delta": chunk},
        )
    yield frame(
        "response.output_text.done",
        {"item_id": message_id, "output_index": 0, "content_index": 0, "text": answer},
    )
    yield frame(
        "response.content_part.done",
        {
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": answer, "annotations": []},
        },
    )
    yield frame(
        "response.output_item.done",
        {"output_index": 0, "item": _message_item(message_id, answer, completed=True)},
    )
    yield frame("response.completed", {"response": completed})


@router.get("/v1/models")
def list_models(request: Request) -> JSONResponse:
    """Report the single logical model so OpenAI clients that probe discovery succeed."""
    try:
        _require_configured_auth(request)
    except _OpenAIError as error:
        return _error_response(error)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": _MODEL_CREATED,
                    "owned_by": MODEL_ID,
                }
            ],
        }
    )
