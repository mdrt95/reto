"""Focused tests for the derived multilingual resume fact index."""

import pytest

from src.models.profile import load_profile
from src.tools.profile_tools import SearchResumeArguments, build_resume_fact_catalog, search_resume


@pytest.mark.parametrize(
    ("query", "topic"),
    [
        ("¿Cuál es tu experiencia?", "experience"),
        ("¿Qué proyectos has construido?", "projects"),
        ("What have you built?", "projects"),
        ("¿Qué tecnologías dominas?", "skills"),
        ("What is your stack?", "skills"),
        ("Tell me about yourself", "summary"),
        ("What projects has Marco worked in?", "projects"),
    ],
)
def test_universal_search_routes_english_and_spanish_topics(query: str, topic: str) -> None:
    profile = load_profile("data/profile.json")

    result = search_resume(profile, SearchResumeArguments(query=query))

    assert result.topic == topic
    assert result.matches
    assert all(match.fact_id.startswith("fact:") for match in result.matches)
    assert all(match.source_id for match in result.matches)


def test_fact_catalog_is_derived_from_profile_values() -> None:
    profile = load_profile("data/profile.json")

    catalog = build_resume_fact_catalog(profile)

    assert any(fact.entity == "Sybil" and "FAISS" in fact.text for fact in catalog)
    assert any(fact.source_id == "education:edu-itcm-ict" for fact in catalog)
    assert not any(fact.topic == "career_preferences" for fact in catalog)


def test_value_fact_ids_do_not_change_when_profile_lists_are_reordered() -> None:
    profile = load_profile("data/profile.json")
    reordered = profile.model_copy(
        update={
            "skills": profile.skills.model_copy(
                update={
                    "programming_languages": list(reversed(profile.skills.programming_languages))
                }
            ),
            "personal": profile.personal.model_copy(
                update={"languages": list(reversed(profile.personal.languages))}
            ),
        }
    )

    original_ids = {fact.text: fact.fact_id for fact in build_resume_fact_catalog(profile)}
    reordered_ids = {fact.text: fact.fact_id for fact in build_resume_fact_catalog(reordered)}

    assert reordered_ids["Python"] == original_ids["Python"]
    assert reordered_ids["Spanish (native)"] == original_ids["Spanish (native)"]


def test_normalization_handles_accents_punctuation_and_synonyms() -> None:
    profile = load_profile("data/profile.json")

    result = search_resume(
        profile,
        SearchResumeArguments(query="¿TECNOLOGÍAS, de recuperación semántica?!"),
    )

    assert result.topic == "skills"
    assert any("Semantic search" in match.text for match in result.matches)


def test_unmatched_named_entity_term_is_reported() -> None:
    """A search term absent from every candidate must be surfaced for boundary wording."""
    profile = load_profile("data/profile.json")

    result = search_resume(profile, SearchResumeArguments(query="experience at Google"))

    assert result.profile_missing is True
    assert result.unmatched_terms == ["google"]


def test_missing_career_preferences_are_explicit() -> None:
    profile = load_profile("data/profile.json")

    result = search_resume(profile, SearchResumeArguments(query="¿Qué tipo de puesto estás buscando?"))

    assert result.topic == "career_preferences"
    assert result.matches == []
    assert result.profile_missing is True


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("tell me about Google", ["Google"]),
        ("Does Marco know Node.js?", []),
        ("Has Marco used ASP.NET Core?", []),
        ("What Is Marco's Role?", []),
        ("Did Marco work at Banorte or Microsoft?", ["Banorte", "Microsoft"]),
        ("Tell me about Sybil and FAISS.", []),
    ],
)
def test_unknown_entity_check_handles_case_dots_and_title_case(message: str, expected: list[str]) -> None:
    """Sentence position, dotted product names, and title-case prose must not skew detection."""
    from src.tools.profile_tools import find_unknown_entities

    profile = load_profile("data/profile.json")

    assert find_unknown_entities(profile, message) == expected


def test_spanish_direct_project_question_renders_the_project_record_narrative() -> None:
    """A broad Spanish project question renders the smallest project record, not siblings."""
    from src.agent.claude import UnavailableClassifier, UnavailableGenerator
    from src.agent.orchestrator import AgentService

    profile = load_profile("data/profile.json")
    service = AgentService(
        profile=profile,
        classifier=UnavailableClassifier(),
        generator=UnavailableGenerator(),
    )

    response = service.respond("¿Qué proyectos has construido?", history=[])

    project = profile.projects[0]
    assert project.narrative is not None
    assert response.answer == project.narrative.es
    assert response.trace.answer_mode == "direct"
    assert response.trace.selected_source_ids == [f"project:{project.id}"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Está bien, platícame sobre sus habilidades", "es"),
        ("Cuéntame sobre sus proyectos", "es"),
        ("What are Marco's skills?", "en"),
        ("Tell me about the stack", "en"),
    ],
)
def test_language_detection_recognizes_common_spanish_request_words(message: str, expected: str) -> None:
    """Spanish requests phrased without question words must still get Spanish rendering."""
    from src.tools.profile_tools import detect_response_language

    assert detect_response_language(message) == expected
