"""Behavioral contracts for precise, deterministic direct answers."""

import pytest
from fastapi.testclient import TestClient

from src.agent.contracts import (
    ConversationState,
    GenerationUnavailableError,
    Intent,
    IntentDecision,
)
from src.agent.orchestrator import AgentService
from src.config import Settings
from src.main import create_app
from src.models.profile import load_profile


class CompaniesClassifier:
    """Deliberately incompatible coarse projection; current-message evidence must win."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return IntentDecision(
            intent=Intent.DIRECT_QUESTION,
            confidence=0.99,
            profile_field="companies",
        )


class ProviderOutage:
    """One boundary double usable as classifier, generator, and rephraser."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        raise GenerationUnavailableError("provider unavailable")

    def generate(self, **_: object) -> object:
        raise AssertionError("direct answers must not call the generator")

    def rephrase(self, **_: object) -> str:
        raise AssertionError("direct answers must not call the rephraser")


class SummaryClassifier:
    """Stable synthesis classifier used to isolate answer-mode planning."""

    def classify(self, message: str, history: list[object]) -> IntentDecision:
        return IntentDecision(
            intent=Intent.SUMMARY_REQUEST,
            confidence=0.99,
            audience="recruiter",
        )


class UnavailableSynthesisGenerator:
    def generate(self, **_: object) -> object:
        raise GenerationUnavailableError("generator unavailable")


@pytest.fixture
def profile():
    return load_profile("data/profile.json")


@pytest.mark.parametrize(
    (
        "message",
        "topic",
        "requested_field",
        "required_fragments",
        "forbidden_fragments",
        "expected_sources",
        "expected_fact_ids",
    ),
    [
        (
            "¿Desde cuándo trabaja Marco en Global Payments?",
            "experience",
            "start_date",
            ("marzo", "2025", "actual"),
            ("Empleadores según",),
            {"experience:exp-global-payments"},
            {
                "fact:experience:exp-global-payments:start_date",
                "fact:experience:exp-global-payments:current",
            },
        ),
        (
            "Has Marco worked with FAISS?",
            "projects",
            "technology",
            ("FAISS", "SQLite FTS5"),
            ("Programming languages", "AI / LLM"),
            {"project:proj-sybil.highlight:sybil-hl-hybrid"},
            {"fact:project:proj-sybil.highlight:sybil-hl-hybrid"},
        ),
        (
            "¿En qué proyectos ha trabajado Marco?",
            "projects",
            "projects",
            ("Sybil", "RAG"),
            ("Global Payments",),
            {"project:proj-sybil"},
            {"fact:project:proj-sybil"},
        ),
        (
            "What security-related work has Marco done?",
            "experience",
            "tag",
            ("Security Console", "security hardening"),
            ("8-person team", "Assisted Senior Engineers"),
            {
                "experience:exp-global-payments.highlight:hl-security-console",
                "experience:exp-global-payments.highlight:hl-performance",
            },
            {
                "fact:experience:exp-global-payments.highlight:hl-security-console",
                "fact:experience:exp-global-payments.highlight:hl-performance",
            },
        ),
        (
            "Dime acerca de la experiencia de Marco",
            "experience",
            "experience",
            ("Global Payments", "Jr. .NET Developer"),
            ("Empleadores según",),
            {"experience:exp-global-payments"},
            {"fact:experience:exp-global-payments"},
        ),
    ],
)
def test_explicit_direct_questions_use_the_smallest_canonical_fact_set(
    profile,
    message: str,
    topic: str,
    requested_field: str,
    required_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
    expected_sources: set[str],
    expected_fact_ids: set[str],
) -> None:
    outage = ProviderOutage()
    service = AgentService(
        profile=profile,
        classifier=CompaniesClassifier(),
        generator=outage,
        rephraser=outage,
    )

    response = service.respond(
        message,
        history=[{"role": "user", "content": "List Marco's employers."}],
    )

    assert response.trace.answer_mode == "direct"
    assert response.trace.rendering_mode == "canonical"
    assert response.trace.answer_topic == topic
    assert response.trace.requested_field == requested_field
    assert response.trace.generator_skipped is True
    assert set(response.trace.selected_fact_ids) == expected_fact_ids
    assert set(response.trace.selected_source_ids) == expected_sources
    assert set(response.trace.claim_source_ids) == expected_sources
    assert all(fragment.casefold() in response.answer.casefold() for fragment in required_fragments)
    assert all(fragment.casefold() not in response.answer.casefold() for fragment in forbidden_fragments)


def test_provider_outage_returns_http_200_when_direct_canonical_facts_suffice(
    profile,
) -> None:
    outage = ProviderOutage()
    service = AgentService(
        profile=profile,
        classifier=outage,
        generator=outage,
        rephraser=outage,
    )
    settings = Settings(environment="test", profile_path="data/profile.json")

    direct = service.respond(
        "¿Desde cuándo trabaja Marco en Global Payments?", history=[]
    )
    assert direct.trace.fallback_reason == "classifier_unavailable"

    with TestClient(create_app(settings, agent_service=service)) as client:
        response = client.post(
            "/api/chat",
            json={"message": "¿Desde cuándo trabaja Marco en Global Payments?"},
        )

    assert response.status_code == 200
    assert "2025" in response.json()["answer"]
    assert "marzo" in response.json()["answer"].casefold()


@pytest.mark.parametrize(
    ("message", "state", "expected_sources", "forbidden_fragment"),
    [
        (
            "What projects has Marco worked on?",
            None,
            {"project:proj-sybil"},
            "Global Payments",
        ),
        (
            "¿En qué proyectos ha trabajado Marco?",
            None,
            {"project:proj-sybil"},
            "Global Payments",
        ),
        (
            "What security-related work has Marco done?",
            None,
            {
                "experience:exp-global-payments.highlight:hl-security-console",
                "experience:exp-global-payments.highlight:hl-performance",
            },
            "8-person team",
        ),
        (
            "¿Qué trabajo relacionado con seguridad ha hecho Marco?",
            None,
            {
                "experience:exp-global-payments.highlight:hl-security-console",
                "experience:exp-global-payments.highlight:hl-performance",
            },
            "equipo de 8 personas",
        ),
        (
            "What else about Marco’s projects?",
            ConversationState(
                last_topic="experience",
                last_source_ids=["experience:exp-global-payments"],
                last_entities=["Global Payments (EVO Payments México)"],
                last_tool="search_resume",
                response_language="en",
            ),
            {"project:proj-sybil"},
            "Global Payments",
        ),
    ],
)
def test_bilingual_direct_topics_override_classifier_and_history_equivalently(
    profile,
    message: str,
    state: ConversationState | None,
    expected_sources: set[str],
    forbidden_fragment: str,
) -> None:
    outage = ProviderOutage()
    service = AgentService(
        profile=profile,
        classifier=CompaniesClassifier(),
        generator=outage,
        rephraser=outage,
    )

    response = service.respond(message, history=[], state=state)

    assert response.trace.answer_mode == "direct"
    assert response.trace.rendering_mode == "canonical"
    assert set(response.trace.selected_source_ids) == expected_sources
    assert forbidden_fragment.casefold() not in response.answer.casefold()


@pytest.mark.parametrize(
    "message",
    [
        "What is the impact of Sybil?",
        "Explain the significance of Sybil.",
        "Summarize when Marco started at Global Payments.",
    ],
)
def test_synthesis_intent_precedes_every_explicit_direct_branch(
    profile,
    message: str,
) -> None:
    service = AgentService(
        profile=profile,
        classifier=SummaryClassifier(),
        generator=UnavailableSynthesisGenerator(),
    )

    response = service.respond(message, history=[])

    assert response.trace.answer_mode == "synthesis"
    assert response.trace.generator_skipped is False


@pytest.mark.parametrize(
    ("message", "expected_rendering"),
    [
        ("What else?", "clarification"),
        ("What did Marco do at Google?", "canonical_not_found"),
    ],
)
def test_in_scope_boundary_answers_still_resolve_one_typed_mode(
    profile,
    message: str,
    expected_rendering: str,
) -> None:
    outage = ProviderOutage()
    service = AgentService(
        profile=profile,
        classifier=outage,
        generator=outage,
        rephraser=outage,
    )

    response = service.respond(message, history=[])

    assert response.trace.answer_mode == "direct"
    assert response.trace.rendering_mode == expected_rendering
