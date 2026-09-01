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
