"""Focused tests for the Anthropic adapter's privacy and parsing boundaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.claude import ClaudeIntentClassifier, ClaudeResponseGenerator, profile_prompt_payload
from src.agent.contracts import GenerationUnavailableError, Intent, InvalidStructuredOutputError
from src.config import Settings
from src.models.profile import load_profile


def test_model_payload_never_contains_phone_or_email() -> None:
    """The provider must never receive private contact data, with no disclosure flag."""
    profile = load_profile("data/profile.json")

    payload = profile_prompt_payload(profile)

    assert "phone" not in payload["personal"]
    assert "email" not in payload["personal"]


def test_intent_classifier_accepts_json_wrapped_in_one_markdown_fence() -> None:
    """Claude may fence valid JSON even when instructed to return JSON only."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                text=(
                    "```json\n"
                    '{"intent":"filter_request","confidence":0.95,'
                    '"filter_by":"tag","filter_value":"security"}'
                    "\n```"
                )
            )
        ]
    )
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    decision = classifier.classify("What security work has Marco done?", [])

    assert decision.intent is Intent.FILTER_REQUEST
    assert decision.filter_by == "tag"
    assert decision.filter_value == "security"


def test_intent_classifier_extracts_one_schema_valid_json_object_from_brief_prose() -> None:
    """A single prose-wrapped object remains safe because Pydantic validates it."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='Result:\n{"intent":"summary_request","confidence":0.91}\nDone.')]
    )
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    decision = classifier.classify("Summarize Marco’s experience.", [])

    assert decision.intent is Intent.SUMMARY_REQUEST


def test_intent_classifier_retries_once_after_schema_invalid_http_success() -> None:
    """One malformed successful provider response must not immediately become a 503."""
    client = MagicMock()
    client.messages.create.side_effect = [
        SimpleNamespace(content=[SimpleNamespace(text='{"intent":"unknown","confidence":0.9}')]),
        SimpleNamespace(content=[SimpleNamespace(text='{"intent":"summary_request","confidence":0.9}')]),
    ]
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    decision = classifier.classify("Summarize Marco’s experience.", [])

    assert decision.intent is Intent.SUMMARY_REQUEST
    assert client.messages.create.call_count == 2


def test_response_generator_retries_once_after_schema_invalid_http_success() -> None:
    """Answer generation gets the same bounded structured-output recovery."""
    client = MagicMock()
    valid = (
        '{"text":"Global Payments (EVO Payments México)","claims":['
        '{"text":"Global Payments (EVO Payments México)","kind":"direct",'
        '"source_ids":["experience:exp-global-payments"],'
        '"evidence":["Global Payments (EVO Payments México)"]}]}'
    )
    client.messages.create.side_effect = [
        SimpleNamespace(content=[SimpleNamespace(text='{"text":"missing claims"}')]),
        SimpleNamespace(content=[SimpleNamespace(text=valid)]),
    ]
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    response = generator.generate(
        message="Where has Marco worked?",
        history=[],
        profile=load_profile("data/profile.json"),
        tool_result=None,
        allowed_source_ids={"experience:exp-global-payments"},
    )

    assert response.text == "Global Payments (EVO Payments México)"
    assert client.messages.create.call_count == 2


def test_exhausted_local_validation_raises_typed_structured_output_error() -> None:
    """The service may recover local contract failures without masking provider outages."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='{"intent":"unknown","confidence":0.9}')]
    )
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    with pytest.raises(InvalidStructuredOutputError):
        classifier.classify("Summarize Marco’s experience.", [])

    assert client.messages.create.call_count == 2


def test_provider_failure_is_not_mislabeled_as_local_validation_failure() -> None:
    """Authentication, transport, and provider errors preserve fail-closed semantics."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("provider unavailable")
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    with pytest.raises(GenerationUnavailableError) as captured:
        classifier.classify("Summarize Marco’s experience.", [])

    assert not isinstance(captured.value, InvalidStructuredOutputError)
