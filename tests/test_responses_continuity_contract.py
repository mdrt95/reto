"""Executable continuation contract for ``POST /v1/responses`` (issue #30).

This module *is* the captured contract: the representative single-turn and
follow-up exchanges the adapter must honor, each pinned as a provider-free test.
Behavioral edge cases live in ``test_openai_compat.py``; this file is the
reference a client integrator reads.

Contract summary
----------------
* ``id`` and message ``id`` are ``resp_`` / ``msg_`` followed by 32 lowercase hex
  characters. Every turn mints a new ``resp_*`` id.
* ``previous_response_id`` resolves to the compact verified ``ConversationState``
  from that turn — verified referents only, never message or answer text.
* A continuation carries the same ``MAX_HISTORY_MESSAGES`` / ``MAX_INPUT_CHARS``
  limits as any other turn; resent ``input`` items form the bounded history and
  the snapshot supplies the referents.
* Snapshots expire after ``RESPONSES_STATE_TTL_SECONDS`` (default 1800), are not
  refreshed on read, and are lost on process restart.
* An unknown, expired, or malformed ``previous_response_id`` fails closed with
  HTTP 404 ``previous_response_not_found`` and no provider call. Client fields
  stay untrusted; guardrails, grounding, privacy, size, and rate limits all run.
"""

import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.agent.contracts import AgentResponse, AgentTrace, ConversationState
from src.config import Settings
from src.main import create_app

TOKEN = "contract-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

RESPONSE_ID = re.compile(r"^resp_[0-9a-f]{32}$")
MESSAGE_ID = re.compile(r"^msg_[0-9a-f]{32}$")


class ContractAgent:
    """A deterministic core double so the documented exchange is reproducible."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._state = ConversationState(
            last_topic="experience",
            last_source_ids=["employment:google"],
            response_language="en",
            focus_source_id="employment:google",
            delivered_fact_ids=["fact:experience:google:role"],
            discussed_topics=["experience"],
            discussed_source_ids=["employment:google"],
        )

    def respond(
        self, message: str, *, history: list[object], state: object | None = None
    ) -> AgentResponse:
        self.calls.append({"message": message, "history": history, "state": state})
        answer = (
            "Marco worked at Google."
            if state is None
            else "He was a senior engineer there."
        )
        return AgentResponse(answer=answer, trace=AgentTrace(), state=self._state)


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        environment="test",
        profile_path="data/profile.json",
        openai_compat_token=TOKEN,
    )
    agent = ContractAgent()
    with TestClient(
        create_app(settings, agent_service=agent), raise_server_exceptions=False
    ) as test_client:
        test_client.contract_agent = agent  # type: ignore[attr-defined]
        yield test_client


def test_single_turn_request_and_response_shape(client: TestClient) -> None:
    request_body = {"model": "banorte-cv-agent", "input": "Where did Marco work?"}

    response = client.post("/v1/responses", headers=AUTH, json=request_body)

    assert response.status_code == 200
    body = response.json()
    assert RESPONSE_ID.match(body["id"])
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "banorte-cv-agent"
    assert MESSAGE_ID.match(body["output"][0]["id"])
    assert body["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "Marco worked at Google.",
        "annotations": [],
    }
    assert body["output_text"] == "Marco worked at Google."
    assert body["error"] is None
    assert client.contract_agent.calls[0]["state"] is None  # type: ignore[attr-defined]


def test_follow_up_with_previous_response_id_only(client: TestClient) -> None:
    first = client.post(
        "/v1/responses", headers=AUTH, json={"input": "Where did Marco work?"}
    )
    previous_response_id = first.json()["id"]

    follow_up = client.post(
        "/v1/responses",
        headers=AUTH,
        json={
            "input": "And his role there?",
            "previous_response_id": previous_response_id,
        },
    )

    assert follow_up.status_code == 200
    body = follow_up.json()
    assert RESPONSE_ID.match(body["id"])
    assert body["id"] != previous_response_id  # every turn mints a new id
    assert body["output_text"] == "He was a senior engineer there."

    resolved = client.contract_agent.calls[1]["state"]  # type: ignore[attr-defined]
    assert resolved is not None
    assert resolved.last_topic == "experience"
    assert resolved.focus_source_id == "employment:google"
    assert client.contract_agent.calls[1]["history"] == []  # type: ignore[attr-defined]


def test_unresolvable_previous_response_id_is_the_documented_error(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/responses",
        headers=AUTH,
        json={
            "input": "And his role there?",
            "previous_response_id": "resp_" + "0" * 32,
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Previous response not found or expired. Resend prior turns in 'input'.",
            "type": "invalid_request_error",
            "param": None,
            "code": "previous_response_not_found",
        }
    }
    assert client.contract_agent.calls == []  # type: ignore[attr-defined]


def test_previous_response_id_with_resent_history_uses_both(client: TestClient) -> None:
    first = client.post(
        "/v1/responses", headers=AUTH, json={"input": "Where did Marco work?"}
    )
    previous_response_id = first.json()["id"]

    follow_up = client.post(
        "/v1/responses",
        headers=AUTH,
        json={
            "previous_response_id": previous_response_id,
            "input": [
                {"role": "user", "content": "Where did Marco work?"},
                {"role": "assistant", "content": "Marco worked at Google."},
                {"role": "user", "content": "And his role there?"},
            ],
        },
    )

    assert follow_up.status_code == 200
    call = client.contract_agent.calls[1]  # type: ignore[attr-defined]
    assert call["message"] == "And his role there?"
    assert call["history"] == [
        {"role": "user", "content": "Where did Marco work?"},
        {"role": "assistant", "content": "Marco worked at Google."},
    ]
    assert call["state"] is not None  # snapshot still supplies referents
