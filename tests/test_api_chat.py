"""Focused contract tests for the first-party public chat endpoint."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.agent.contracts import AgentResponse, AgentTrace
from src.agent.claude import GenerationUnavailableError
from src.config import Settings
from src.main import create_app


class SuccessfulAgent:
    """Small fake that keeps HTTP contract tests independent of model calls."""

    def respond(self, message: str, *, history: list[object]) -> AgentResponse:
        return AgentResponse(answer=f"Verified: {message}", trace=AgentTrace())


class UnavailableAgent:
    """Fake gateway failure used to verify public error sanitization."""

    def respond(self, message: str, *, history: list[object]) -> AgentResponse:
        raise GenerationUnavailableError("provider timeout details must stay private")


class BrokenAgent:
    """Unexpected failure double used to protect the generic public boundary."""

    def respond(self, message: str, *, history: list[object]) -> AgentResponse:
        raise RuntimeError("internal diagnostic that must not reach the client")


@pytest.fixture
def chat_client() -> Iterator[TestClient]:
    """Provide a started app wired to a deterministic successful core service."""
    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(create_app(settings, agent_service=SuccessfulAgent())) as client:
        yield client


def test_chat_returns_the_stable_public_contract(chat_client: TestClient) -> None:
    """A valid first-party request returns no internal trace or profile secrets."""
    response = chat_client.post("/api/chat", json={"message": "What languages does Marco speak?"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("req_")
    assert body["conversation_id"].startswith("conv_")
    assert body["status"] == "completed"
    assert body["answer"].startswith("Verified:")
    assert "trace" not in body


def test_chat_logs_the_informativeness_outcome_without_exposing_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private delivery-floor outcome must reach content-free observability."""
    events: list[object] = []

    class InformativenessFallbackAgent:
        def respond(self, message: str, *, history: list[object]) -> AgentResponse:
            return AgentResponse(
                answer="Please ask about the profile section where that item appears.",
                trace=AgentTrace(
                    informativeness_outcome="fallback",
                    rendering_mode="informativeness_fallback",
                ),
            )

    monkeypatch.setattr("src.api.chat.log_turn", events.append)
    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(
        create_app(settings, agent_service=InformativenessFallbackAgent())
    ) as client:
        response = client.post("/api/chat", json={"message": "Tell me more."})

    assert response.status_code == 200
    assert len(events) == 1
    event = events[0]
    assert getattr(event, "informativeness_outcome") == "fallback"
    assert getattr(event, "rendering_mode") == "informativeness_fallback"
    assert "trace" not in response.json()


def test_chat_logs_the_evidence_topics_a_spanning_answer_drew_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A theme answer that spans topics reports every one to content-free logging."""
    events: list[object] = []

    class SpanningAgent:
        def respond(self, message: str, *, history: list[object]) -> AgentResponse:
            return AgentResponse(
                answer="Sybil and the multi-agent ISV work.",
                trace=AgentTrace(
                    answer_topic="projects",
                    evidence_topics=["experience", "projects"],
                ),
            )

    monkeypatch.setattr("src.api.chat.log_turn", events.append)
    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(create_app(settings, agent_service=SpanningAgent())) as client:
        response = client.post("/api/chat", json={"message": "What is his AI experience?"})

    assert response.status_code == 200
    assert getattr(events[0], "evidence_topics") == ["experience", "projects"]


def test_chat_accepts_and_returns_optional_conversation_state() -> None:
    """State extension stays optional and preserves the original request contract."""
    from src.agent.contracts import ConversationState

    class StatefulAgent:
        def respond(
            self,
            message: str,
            *,
            history: list[object],
            state: ConversationState | None = None,
        ) -> AgentResponse:
            assert state is not None and state.last_topic == "projects"
            return AgentResponse(answer="More", trace=AgentTrace(), state=state)

    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(create_app(settings, agent_service=StatefulAgent())) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "What else?",
                "state": {
                    "last_topic": "projects",
                    "last_source_ids": ["project:proj-sybil"],
                    "last_entities": ["Sybil"],
                    "last_tool": "search_resume",
                    "response_language": "en",
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["state"]["last_topic"] == "projects"


def test_chat_rejects_oversized_conversation_state(chat_client: TestClient) -> None:
    response = chat_client.post(
        "/api/chat",
        json={
            "message": "What else?",
            "state": {
                "last_topic": "projects",
                "last_source_ids": [f"project:{index}" for index in range(21)],
                "last_entities": ["Sybil"],
                "last_tool": "search_resume",
                "response_language": "en",
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_chat_rejects_blank_message_as_problem_details(chat_client: TestClient) -> None:
    """Malformed public input must receive the documented sanitized 422 shape."""
    response = chat_client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "invalid_request"
    assert response.json()["request_id"].startswith("req_")


def test_chat_maps_provider_failure_without_leaking_details() -> None:
    """Provider exceptions must become retry-safe public errors."""
    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(create_app(settings, agent_service=UnavailableAgent())) as client:
        response = client.post("/api/chat", json={"message": "Tell me about Sybil"})

    assert response.status_code == 503
    assert response.json()["code"] == "generation_unavailable"
    assert "provider timeout" not in response.text


def test_chat_sanitizes_unexpected_failures() -> None:
    """Unexpected exceptions must remain a stable public 500 problem response."""
    settings = Settings(environment="test", profile_path="data/profile.json")
    with TestClient(
        create_app(settings, agent_service=BrokenAgent()),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/api/chat", json={"message": "Tell me about Sybil"})

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "internal diagnostic" not in response.text
