"""Focused tests for the Anthropic adapter's privacy and parsing boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.claude import ClaudeIntentClassifier, ClaudeRephraser, ClaudeResponseGenerator
from src.agent.contracts import GenerationUnavailableError, Intent, InvalidStructuredOutputError
from src.config import Settings
from src.models.profile import load_profile
from src.tools.profile_tools import build_resume_fact_catalog


def test_generation_payload_never_contains_the_full_profile_or_contact_data() -> None:
    """The provider must see selected catalog facts only, never the full profile."""
    profile = load_profile("data/profile.json")
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                text=(
                    '{"text":"x","claims":[{"text":"x","kind":"direct",'
                    '"source_ids":["skills"],"evidence":["x"]}]}'
                )
            )
        ]
    )
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )
    catalog = build_resume_fact_catalog(profile)
    allowed_facts = [fact for fact in catalog if fact.source_id == "skills"][:2]

    generator.generate(
        message="What skills does Marco have?",
        history=[],
        allowed_facts=allowed_facts,
        tool_result=None,
        allowed_source_ids={"skills"},
    )

    request_json = json.dumps(client.messages.create.call_args.kwargs)
    assert '"profile"' not in request_json
    assert profile.personal.phone not in request_json
    assert profile.personal.email not in request_json
    for fact in allowed_facts:
        assert fact.text in request_json


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
        allowed_facts=[],
        tool_result=None,
        allowed_source_ids={"experience:exp-global-payments"},
    )

    assert response.text == "Global Payments (EVO Payments México)"
    assert client.messages.create.call_count == 2


def test_generator_skips_second_attempt_when_first_attempt_used_most_of_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow first attempt must not risk a second call that would exceed the timeout."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='{"text":"missing claims"}')]
    )
    times = iter([0.0, 20.0])
    monkeypatch.setattr("src.agent.claude.time.monotonic", lambda: next(times))
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key", model_timeout_seconds=30),
    )

    with pytest.raises(InvalidStructuredOutputError):
        generator.generate(
            message="Where has Marco worked?",
            history=[],
            allowed_facts=[],
            tool_result=None,
            allowed_source_ids=set(),
        )

    assert client.messages.create.call_count == 1


def test_intent_classifier_uses_recent_history_and_puts_message_first() -> None:
    """A new topic in `message` must not inherit intent bias from older history."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='{"intent":"direct_question","confidence":0.9}')]
    )
    classifier = ClaudeIntentClassifier(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )
    history = [{"role": "user", "content": f"turn {i}"} for i in range(6)]

    classifier.classify("platícame sobre sus habilidades", history)

    request_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    payload = json.loads(request_content)

    assert payload["recent_history"] == history[-2:]
    assert request_content.index('"message"') < request_content.index('"recent_history"')


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


def test_generation_caches_a_prefix_no_turn_content_can_change() -> None:
    """One per-turn byte in the cached prefix would silently stop caching for all traffic."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                text=(
                    '{"text":"Global Payments (EVO Payments México)","claims":['
                    '{"text":"Global Payments (EVO Payments México)","kind":"direct",'
                    '"source_ids":["experience:exp-global-payments"],'
                    '"evidence":["Global Payments (EVO Payments México)"]}]}'
                )
            )
        ]
    )
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    generator.generate(
        message="Where has Marco worked?",
        history=[],
        allowed_facts=[],
        tool_result=None,
        allowed_source_ids={"experience:exp-global-payments"},
    )
    generator.generate(
        message="What languages does Marco speak?",
        history=[{"role": "user", "content": "an earlier turn"}],
        allowed_facts=[],
        tool_result=None,
        allowed_source_ids={"personal"},
    )

    first, second = (call.kwargs["system"] for call in client.messages.create.call_args_list)
    assert first == second
    assert first[-1]["cache_control"] == {"type": "ephemeral"}
    assert "Where has Marco worked?" not in json.dumps(first)


def test_rephraser_extracts_text_from_fenced_json_and_never_sends_the_profile() -> None:
    """The rephraser must parse fenced JSON and never leak the full profile or contact data."""
    profile = load_profile("data/profile.json")
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='```json\n{"text": "Marco built the Security Console."}\n```')]
    )
    rephraser = ClaudeRephraser(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )
    catalog = build_resume_fact_catalog(profile)
    facts = [fact for fact in catalog if fact.source_id.endswith("hl-security-console")]

    text = rephraser.rephrase(
        message="Tell me about Marco's security work.",
        facts=facts,
        language="en",
    )

    assert text == "Marco built the Security Console."
    request_json = json.dumps(client.messages.create.call_args.kwargs)
    assert '"profile"' not in request_json
    assert profile.personal.phone not in request_json
    assert profile.personal.email not in request_json


def test_rephraser_provider_failure_raises_generation_unavailable() -> None:
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("provider unavailable")
    rephraser = ClaudeRephraser(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    with pytest.raises(GenerationUnavailableError):
        rephraser.rephrase(message="Tell me about Marco.", facts=[], language="en")


def test_rephraser_retries_once_then_raises_invalid_structured_output() -> None:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="not json at all")]
    )
    rephraser = ClaudeRephraser(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    with pytest.raises(InvalidStructuredOutputError):
        rephraser.rephrase(message="Tell me about Marco.", facts=[], language="en")

    assert client.messages.create.call_count == 2


def test_generator_unwraps_a_single_key_wrapper_object() -> None:
    """A model that mirrors the prompt's `response_format` key must still validate."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                text=(
                    '{"response_format":{"text":"Marco builds AI systems.","claims":['
                    '{"text":"Marco builds AI systems.","kind":"direct",'
                    '"source_ids":["skills"],"fact_ids":["fact:skills:python"]}]}}'
                )
            )
        ],
    )
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    response = generator.generate(
        message="What skills does Marco have?",
        history=[],
        allowed_facts=[],
        tool_result=None,
        allowed_source_ids={"skills"},
    )

    assert response.text == "Marco builds AI systems."
    assert client.messages.create.call_count == 1


def test_generator_raises_immediately_on_truncated_max_tokens_output() -> None:
    """A stop_reason of max_tokens means the JSON is unterminated; do not retry it."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="max_tokens",
        content=[SimpleNamespace(text='{"text":"partial answer","claims":[{"text":"x"')],
    )
    generator = ClaudeResponseGenerator(
        client=client,
        settings=Settings(environment="test", anthropic_api_key="test-key"),
    )

    with pytest.raises(InvalidStructuredOutputError, match="truncated"):
        generator.generate(
            message="Summarize Marco's experience.",
            history=[],
            allowed_facts=[],
            tool_result=None,
            allowed_source_ids=set(),
        )

    assert client.messages.create.call_count == 1
